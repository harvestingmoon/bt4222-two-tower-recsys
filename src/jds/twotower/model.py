"""Two-tower model: item tower (content) vs query+user tower (test query + segmented history), cosine score.

  * Item tower is content-based: term bags (name / brand / category), brand/shop/category ids and windowed
    popularity buckets. No item-id embedding: ~80% of displayed items never occur in any history.
  * Query+user tower: the test query shares the item tower's term embedding. Each history event is the item tower
    output + event type + log-time-before-anchor + the query it was issued under (query-tagged history). Each of
    the three history segments (short / mid / long, see data.py) is pooled by its own query-guided attention with a
    null slot (so an empty segment is valid), and the segment sizes enter as a feature.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import C_BRAND, C_CATE, C_ID0, C_NAME, C_POP0, N_POP_BUCKETS, N_TIME_BUCKETS, POP_COLS, SEGMENTS


def masked_mean(e: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    m = m.unsqueeze(-1).to(e.dtype)
    return (e * m).sum(-2) / m.sum(-2).clamp(min=1.0)


class ItemTower(nn.Module):
    def __init__(self, n_terms: int, id_sizes: list[int], tc: dict):
        super().__init__()
        d_term, d_id, d_pop, d = tc["d_term"], tc["d_id"], tc["d_pop"], tc["d"]
        self.term = nn.Embedding(n_terms, d_term, padding_idx=0)
        self.ids = nn.ModuleList([nn.Embedding(n, d_id, padding_idx=0) for n in id_sizes])
        self.pop = nn.ModuleList([nn.Embedding(N_POP_BUCKETS, d_pop, padding_idx=0) for _ in POP_COLS])
        d_in = 3 * d_term + len(id_sizes) * d_id + len(POP_COLS) * d_pop
        self.mlp = nn.Sequential(nn.Linear(d_in, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:             # x: [..., N_FEAT] int
        x = x.long()
        parts = [masked_mean(self.term(x[..., sl]), x[..., sl] > 0) for sl in (C_NAME, C_BRAND, C_CATE)]
        parts += [emb(x[..., C_ID0 + j]) for j, emb in enumerate(self.ids)]
        parts += [emb(x[..., C_POP0 + j]) for j, emb in enumerate(self.pop)]
        return self.mlp(torch.cat(parts, -1))


class SegmentPool(nn.Module):
    """Query-guided attention over one history segment, with an always-present null slot."""

    def __init__(self, d: int):
        super().__init__()
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.q = nn.Linear(d, d, bias=False)
        self.last_attn = None

    def forward(self, q: torch.Tensor, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = h.size(0)
        h = torch.cat([self.null.expand(B, 1, -1).to(h.dtype), h], 1)
        mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=h.device), mask], 1)
        s = torch.einsum("bd,bld->bl", self.q(q), h) / math.sqrt(h.size(-1))
        a = torch.softmax(s.float().masked_fill(~mask, -1e4), -1).to(h.dtype)
        self.last_attn = a.detach()
        return torch.einsum("bl,bld->bd", a, h)


class QueryUserTower(nn.Module):
    def __init__(self, item_tower: ItemTower, tc: dict, use_history: bool = True, use_query: bool = True):
        super().__init__()
        d, d_term = tc["d"], tc["d_term"]
        self.it = item_tower
        self.use_history, self.use_query = use_history, use_query
        self.q_proj = nn.Sequential(nn.Linear(d_term, d), nn.GELU(), nn.Linear(d, d))
        self.hq_proj = nn.Linear(d_term, d)                          # query under which a history event happened
        self.type_emb = nn.Embedding(5, d, padding_idx=0)
        self.time_emb = nn.Embedding(N_TIME_BUCKETS, d, padding_idx=0)
        self.pools = nn.ModuleDict({s: SegmentPool(d) for s in SEGMENTS})
        self.count = nn.Linear(len(SEGMENTS), d)
        self.out = nn.Sequential(nn.Linear((2 + 2 * len(SEGMENTS)) * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def encode_query(self, q_terms: torch.Tensor) -> torch.Tensor:
        return self.q_proj(masked_mean(self.it.term(q_terms.long()), q_terms > 0))

    def forward(self, q_terms: torch.Tensor, ctx: dict, item_feats) -> torch.Tensor:
        """q_terms [B,Tq]; ctx[segment] = {item [B,L] (row index), type, time, query [B,L,Tq], mask, count [B]};
        item_feats(index tensor) -> [.., N_FEAT] feature rows (gathered from the item matrix)."""
        q = self.encode_query(q_terms)
        if not self.use_query:
            q = torch.zeros_like(q)
        parts = [q]
        pooled = []
        for s in SEGMENTS:
            c = ctx[s]
            if self.use_history:
                hq = c["query"].long()
                h = (self.it(item_feats(c["item"])) + self.type_emb(c["type"].long()) + self.time_emb(c["time"].long())
                     + self.hq_proj(masked_mean(self.it.term(hq), hq > 0)))
                pooled.append(self.pools[s](q, h, c["mask"]))
            else:
                pooled.append(torch.zeros_like(q))
        counts = torch.stack([ctx[s]["count"] for s in SEGMENTS], -1).to(q.dtype)
        if not self.use_history:
            counts = torch.zeros_like(counts)
        parts += pooled + [q * p for p in pooled] + [self.count(counts)]
        return self.out(torch.cat(parts, -1))


class TwoTower(nn.Module):
    def __init__(self, n_terms: int, id_sizes: list[int], tc: dict, use_history: bool = True, use_query: bool = True):
        super().__init__()
        self.item = ItemTower(n_terms, id_sizes, tc)
        self.user = QueryUserTower(self.item, tc, use_history=use_history, use_query=use_query)
        self.log_scale = nn.Parameter(torch.tensor(math.log(20.0)))   # 1 / temperature

    def item_vec(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item(x).float(), dim=-1)

    def user_vec(self, q_terms, ctx, item_feats) -> torch.Tensor:
        return F.normalize(self.user(q_terms, ctx, item_feats).float(), dim=-1)

    def scale(self) -> torch.Tensor:
        return self.log_scale.exp().clamp(max=100.0)

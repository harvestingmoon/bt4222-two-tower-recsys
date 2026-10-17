import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jds.config import load_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SEP = "\x18"
DAY = 86400


def _write_raw(raw: Path, n_users: int = 40) -> None:
    """A tiny dataset in the exact raw JDsearch format (labels sorted, positives first, like the real file)."""
    users = ["query\tcandidate_wid_list\tcandidate_label_list\thistory_qry_list\thistory_wid_list\thistory_type_list\thistory_time_list"]
    for u in range(n_users):
        n_pos = 0 if u == 0 else 1 + u % 3                       # user 0 has no positive
        labels = sorted([1 + (j % 3) for j in range(n_pos)], reverse=True) + [0] * (8 - n_pos)
        cands = [100 + (u * 3 + j) % 30 for j in range(8)]
        # history: an order 200 days ago, a cart 30 days ago, two clicks 2 days and 1 hour ago (gap to query: 1 hour)
        qry = ["-1", f"7{SEP}8", f"{u % 5}{SEP}9", ""]
        wid = [100 + u % 30, 101 + u % 29, 102 + u % 28, 103]
        typ = ["ORD", "CART", "CLICK", "CLICK"]
        times = [0, 170 * DAY, 28 * DAY, 2 * DAY - 3600, 3600]   # len = history + 1; total = 200 days
        users.append("\t".join([f"{u % 5}{SEP}{SEP}9", "_".join(map(str, cands)), "_".join(map(str, labels)),
                                "_".join(qry), "_".join(map(str, wid)), "_".join(typ), "_".join(map(str, times))]))
    (raw / "user_behavior_data.txt").write_text("\n".join(users) + "\n")
    prods = ["wid\tname\tbrand_id\tbrand_name\tcate_id_1\tcate_name_1\tcate_id_2\tcate_name_2\tcate_id_3\tcate_name_3\tcate_id_4\tcate_name_4\tshop_id"]
    for w in range(100, 140):
        prods.append("\t".join(map(str, [w, f"{w % 5}{SEP}9{SEP}{w}", w % 4, 50 + w % 4, 1, 60, 2, 61, 3 + w % 2, 62, 10 + w % 3,
                                         f"63{SEP}{w % 3}", w % 6])))
    prods.append("\t".join(map(str, [999, "1", 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])))   # never referenced -> dropped
    (raw / "product_meta_data.txt").write_text("\n".join(prods) + "\n")


@pytest.fixture(scope="session")
def tiny_cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("jds")
    raw = root / "raw"
    raw.mkdir()
    _write_raw(raw)
    cfg = load_config(REPO / "configs" / "smoke.yaml",
                      [f"paths.raw_dir={raw}", f"paths.work_dir={root / 'out'}", "scope=full", "features.chunk_users=15",
                       "features.bm25.idf_docs=100", "tower.term_count_docs=100", "tower.min_term_count=1"])
    from jds.prepare import prepare_full
    prepare_full(cfg)
    return cfg

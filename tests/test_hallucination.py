"""Kiểm thử bảng gán nhãn ảo giác: chọn mẫu, chia việc, ẩn cấu hình, đọc nhãn."""
import csv
from collections import Counter

from rag.hallucination.annotation import (
    KEY_COLUMNS, SHARED, SHEET_COLUMNS, final_labels, load_labels, make_sheets, write_csv)

CONFIGS = ("S0", "S2", "S3")


def _data(n_inc=70, n_ooc=25):
    types = ["fact", "number", "condition", "multi"]
    qs = [{"id": f"q{i}", "question": f"câu {i}", "answer": f"đáp án {i}", "answerable": True,
           "type": types[i % 4], "gold_chunks": ["c0"]} for i in range(n_inc)]
    qs += [{"id": f"o{i}", "question": f"ngoài kho {i}", "answer": "", "answerable": False,
            "type": "out_of_corpus"} for i in range(n_ooc)]
    chunks = [{"id": "c0", "doc_name": "Quy chế", "text": "Điều 1. Nội dung"}]
    gens = [{"qid": q["id"], "config": c, "chunk_ids": [] if c == "S0" else ["c0"],
             "answer": f"trả lời {q['id']} {c}", "k": 0 if c == "S0" else 1}
            for q in qs for c in CONFIGS]
    return qs, gens, chunks


def test_sizes_and_each_item_assigned_once():
    sheets, key = make_sheets(*_data())
    assert len(key) == 60 * 3
    who = Counter(r["nguoi_gan"] for r in key)
    assert who[SHARED] == 30 and who["TV1"] == who["TV2"] == who["TV3"] == 50
    for a, rows in sheets.items():
        assert len(rows) == 80
        assert len({r["ma_muc"] for r in rows}) == 80


def test_every_annotator_gets_every_config():
    _, key = make_sheets(*_data())
    per = Counter((r["nguoi_gan"], r["config"]) for r in key)
    for a in ("TV1", "TV2", "TV3"):
        assert all(16 <= per[(a, c)] <= 17 for c in CONFIGS), per
    assert all(per[(SHARED, c)] == 10 for c in CONFIGS)


def test_sheets_hide_config_and_include_out_of_corpus():
    sheets, key = make_sheets(*_data())
    assert set(sheets["TV1"][0]) == set(SHEET_COLUMNS)
    assert "config" not in SHEET_COLUMNS and "qid" not in SHEET_COLUMNS
    n_ooc = len({r["qid"] for r in key if r["type"] == "out_of_corpus"})
    assert n_ooc >= 15


def test_deterministic_with_seed():
    a = make_sheets(*_data(), seed=1)
    b = make_sheets(*_data(), seed=1)
    c = make_sheets(*_data(), seed=2)
    assert a == b and a != c


def _write_labels(tmp_path, key, votes):
    write_csv(tmp_path / "key.csv", KEY_COLUMNS, key)
    for who, rows in votes.items():
        write_csv(tmp_path / "labels" / f"{who}.csv", ["ma_muc", "nhan", "trich_dan_sai"], rows)


def test_majority_vote_ties_and_final_override(tmp_path):
    key = [{"ma_muc": m, "qid": "q", "config": "S2", "nguoi_gan": SHARED, "type": "fact",
            "answerable": True} for m in ("H001", "H002")]
    _write_labels(tmp_path, key, {
        "TV1": [{"ma_muc": "H001", "nhan": "3", "trich_dan_sai": "x"}, {"ma_muc": "H002", "nhan": "1"}],
        "TV2": [{"ma_muc": "H001", "nhan": "3", "trich_dan_sai": ""}, {"ma_muc": "H002", "nhan": "2"}],
        "TV3": [{"ma_muc": "H001", "nhan": "1", "trich_dan_sai": "x"}, {"ma_muc": "H002", "nhan": "4"}],
    })
    labels, problems = load_labels(tmp_path)
    assert not problems and len(labels) == 6
    final, ties = final_labels(labels, tmp_path)
    assert ties == ["H002"]
    assert final == [{"ma_muc": "H001", "qid": "q", "config": "S2", "label": 3, "wrong_cite": True}]
    # nhóm bàn xong và chốt nhãn cho H002
    with open(tmp_path / "labels" / "final.csv", "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows([["ma_muc", "nhan", "trich_dan_sai"], ["H002", "4", ""]])
    final, ties = final_labels(load_labels(tmp_path)[0], tmp_path)
    assert not ties and {x["ma_muc"]: x["label"] for x in final} == {"H001": 3, "H002": 4}


def test_load_labels_reports_bad_rows(tmp_path):
    key = [{"ma_muc": "H001", "qid": "q", "config": "S0", "nguoi_gan": "TV1", "type": "fact",
            "answerable": True}]
    _write_labels(tmp_path, key, {
        "TV1": [{"ma_muc": "H001", "nhan": "7"}, {"ma_muc": "H999", "nhan": "1"}],
        "TV2": [{"ma_muc": "H001", "nhan": "1"}],   # mục của TV1, không phải của TV2
    })
    labels, problems = load_labels(tmp_path)
    assert labels == [] and len(problems) == 3

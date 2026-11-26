"""Bảng gán nhãn ảo giác (mục 7 kế hoạch) và đọc lại nhãn sau khi gán.

1) Tạo bảng gán nhãn từ câu trả lời mà TV3 sinh ra (generate.py):
     python -m rag.hallucination.annotation make --gen outputs/generations.jsonl

   - Chọn ngẫu nhiên 60 câu hỏi, phân tầng theo loại câu (có đủ câu ngoài kho),
     lấy câu trả lời của 3 cấu hình S0, S2, S3 -> 180 mục.
   - 30 mục cả 3 người cùng gán (để tính Fleiss kappa), 150 mục còn lại chia đều
     ~50 mục/người, cân bằng giữa các cấu hình.
   - Bảng của người gán KHÔNG có tên cấu hình; mã mục được cấp sau khi xáo trộn,
     và mỗi bảng xáo thứ tự riêng. Ánh xạ mã mục -> (câu hỏi, cấu hình) nằm trong key.csv.

   Đầu ra trong results/hallucination/annotation/:
     sheet_TV1.csv, sheet_TV2.csv, sheet_TV3.csv  -> mỗi file nhập vào một tab Google Sheet
     key.csv                                      -> KHÔNG đưa cho người gán xem

2) Sau khi gán: tải từng tab về dạng CSV, lưu vào
     results/hallucination/annotation/labels/<người gán>.csv
   rồi dùng load_labels() / final_labels() (scripts/hallucination_breakdown.py gọi sẵn).
   Nếu nhóm đã bàn và chốt nhãn cho các mục chung, ghi vào labels/final.csv
   (cột ma_muc, nhan, trich_dan_sai); file này được ưu tiên hơn bỏ phiếu đa số.
"""
import argparse
import csv
import glob
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ANN_DIR = Path("results/hallucination/annotation")
DEFAULT_QUESTIONS = "data/rag_test/*.jsonl"
SHEET_COLUMNS = ["ma_muc", "cau_hoi", "dap_an_chuan", "ngu_canh", "cau_tra_loi",
                 "nhan", "trich_dan_sai", "ghi_chu"]
KEY_COLUMNS = ["ma_muc", "qid", "config", "nguoi_gan", "type", "answerable"]
SHARED = "ALL"

LABELS = {
    1: "Đúng, có căn cứ",
    2: "Đúng nhưng không có căn cứ",
    3: "Ảo giác – mâu thuẫn",
    4: "Ảo giác – bịa",
    5: "Từ chối đúng",
    6: "Từ chối sai",
}
HALLUCINATION = {3, 4}
NO_CONTEXT = "(Không có ngữ cảnh: mô hình trả lời mà không được đưa tài liệu)"
OUT_OF_CORPUS = "(Câu hỏi KHÔNG có đáp án trong kho tài liệu)"


# ------------------------------------------------------------------ I/O
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: Excel và Google Sheets đều đọc đúng tiếng Việt
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ chọn mẫu
def qtype(q):
    return q["type"] if q.get("answerable", True) else "out_of_corpus"


def _proportional(groups, n, rng):
    """Lấy n phần tử, chia theo tỉ lệ kích thước nhóm (làm tròn theo phần dư lớn nhất)."""
    total = sum(len(g) for g in groups.values())
    n = min(n, total)
    quota = {t: n * len(g) / total for t, g in groups.items()}
    take = {t: int(v) for t, v in quota.items()}
    for t in sorted(quota, key=lambda t: (quota[t] - take[t], t), reverse=True)[: n - sum(take.values())]:
        take[t] += 1
    out = []
    for t in sorted(groups):
        out += rng.sample(groups[t], take[t])
    return out


def sample_questions(questions, n, min_ooc, rng):
    """Phân tầng theo loại câu; bảo đảm ít nhất min_ooc câu ngoài kho (nếu có đủ)."""
    ooc = [q for q in questions if qtype(q) == "out_of_corpus"]
    inc = [q for q in questions if qtype(q) != "out_of_corpus"]
    n = min(n, len(questions))
    n_ooc = min(len(ooc), max(round(n * len(ooc) / len(questions)), min_ooc))
    n_ooc = max(n_ooc, n - len(inc))
    groups = defaultdict(list)
    for q in inc:
        groups[qtype(q)].append(q)
    return rng.sample(ooc, n_ooc) + _proportional(groups, n - n_ooc, rng)


def assign(items, annotators, n_shared, rng):
    """Gán mỗi mục cho SHARED hoặc đúng một người; cân bằng theo cấu hình.

    Phần chung: lấy xen kẽ các cấu hình. Phần riêng: chia lần lượt TỪNG cấu hình cho
    mọi người (bộ đếm chạy liên tục), để không ai chỉ chấm một cấu hình — nếu không,
    độ khắt khe của từng người sẽ lẫn vào chênh lệch giữa các cấu hình.
    """
    by_cfg = defaultdict(list)
    for it in items:
        by_cfg[it["config"]].append(it)
    cfgs = sorted(by_cfg)
    for c in cfgs:
        rng.shuffle(by_cfg[c])
    shared, i = [], 0
    while len(shared) < n_shared and any(by_cfg.values()):
        c = cfgs[i % len(cfgs)]
        if by_cfg[c]:
            shared.append(by_cfg[c].pop())
        i += 1
    for it in shared:
        it["nguoi_gan"] = SHARED
    j = 0
    for c in cfgs:
        for it in by_cfg[c]:
            it["nguoi_gan"] = annotators[j % len(annotators)]
            j += 1
    return items


# ------------------------------------------------------------------ hiển thị
def format_context(chunk_ids, chunk_index):
    if not chunk_ids:
        return NO_CONTEXT
    parts = []
    for i, cid in enumerate(chunk_ids, start=1):
        c = chunk_index[cid]
        parts.append(f"[{i}] {c.get('doc_name', c.get('doc_id', ''))}\n{c['text']}")
    return "\n\n".join(parts)


def build_items(questions, gens, configs, chunk_index):
    gen = {(str(r["qid"]), r["config"]): r for r in gens}
    items = []
    for q in questions:
        for cfg in configs:
            r = gen[(q["id"], cfg)]
            items.append({
                "qid": q["id"], "config": cfg, "type": qtype(q),
                "answerable": bool(q.get("answerable", True)),
                "cau_hoi": q["question"],
                "dap_an_chuan": q["answer"] if q.get("answerable", True) else OUT_OF_CORPUS,
                "ngu_canh": format_context(r.get("chunk_ids") or [], chunk_index),
                "cau_tra_loi": r["answer"],
            })
    return items


def make_sheets(questions, gens, chunks, configs=("S0", "S2", "S3"), n_questions=60,
                min_ooc=15, n_shared=30, annotators=("TV1", "TV2", "TV3"), seed=2026):
    """Trả về (sheets: {người: [dòng]}, key: [dòng]). Không ghi file."""
    rng = random.Random(seed)
    have = Counter(str(r["qid"]) for r in gens if r["config"] in configs)
    usable = [q for q in questions if have[q["id"]] == len(configs)]
    missing = len(questions) - len(usable)
    if missing:
        print(f"[cảnh báo] bỏ {missing} câu chưa có đủ câu trả lời cho {list(configs)}")
    chosen = sample_questions(usable, n_questions, min_ooc, rng)
    chunk_index = {c["id"]: c for c in chunks}
    items = build_items(chosen, gens, configs, chunk_index)
    rng.shuffle(items)
    for i, it in enumerate(items, start=1):  # mã cấp SAU khi xáo -> không lộ cấu hình
        it["ma_muc"] = f"H{i:03d}"
    assign(items, list(annotators), n_shared, rng)

    sheets = {}
    for a in annotators:
        rows = [{c: it.get(c, "") for c in SHEET_COLUMNS}
                for it in items if it["nguoi_gan"] in (SHARED, a)]
        random.Random(f"{seed}-{a}").shuffle(rows)
        sheets[a] = rows
    key = sorted(({c: it[c] for c in KEY_COLUMNS} for it in items), key=lambda r: r["ma_muc"])
    return sheets, key


# ------------------------------------------------------------------ đọc nhãn
def _parse_label(v):
    v = (v or "").strip()
    return int(v) if v.isdigit() and int(v) in LABELS else None


def _parse_flag(v):
    return (v or "").strip().lower() in {"x", "1", "có", "co", "yes", "true"}


def load_labels(ann_dir=ANN_DIR):
    """Mọi nhãn đã gán: list {ma_muc, qid, config, annotator, shared, label, wrong_cite}."""
    ann_dir = Path(ann_dir)
    key = {r["ma_muc"]: r for r in read_csv(ann_dir / "key.csv")}
    out, problems = [], []
    for path in sorted((ann_dir / "labels").glob("*.csv")):
        if path.stem == "final":
            continue
        for row in read_csv(path):
            k = key.get(row.get("ma_muc", "").strip())
            if k is None:
                problems.append(f"{path.name}: mã mục lạ {row.get('ma_muc')!r}")
                continue
            if k["nguoi_gan"] not in (SHARED, path.stem):
                problems.append(f"{path.name}: {k['ma_muc']} không thuộc bảng của {path.stem}")
                continue
            label = _parse_label(row.get("nhan"))
            if label is None:
                problems.append(f"{path.name}: {k['ma_muc']} chưa có nhãn hợp lệ (1–6)")
                continue
            out.append({"ma_muc": k["ma_muc"], "qid": k["qid"], "config": k["config"],
                        "annotator": path.stem, "shared": k["nguoi_gan"] == SHARED,
                        "label": label, "wrong_cite": _parse_flag(row.get("trich_dan_sai"))})
    return out, problems


def final_labels(labels, ann_dir=ANN_DIR):
    """Một nhãn cho mỗi mục. Mục chung: labels/final.csv nếu có, không thì đa số phiếu."""
    final_path = Path(ann_dir) / "labels" / "final.csv"
    agreed = {}
    if final_path.exists():
        for r in read_csv(final_path):
            if _parse_label(r.get("nhan")):
                agreed[r["ma_muc"].strip()] = (_parse_label(r["nhan"]), _parse_flag(r.get("trich_dan_sai")))
    by_item = defaultdict(list)
    for x in labels:
        by_item[x["ma_muc"]].append(x)
    out, ties = [], []
    for item, xs in sorted(by_item.items()):
        base = {k: xs[0][k] for k in ("ma_muc", "qid", "config")}
        if item in agreed:
            label, cite = agreed[item]
        else:
            votes = Counter(x["label"] for x in xs).most_common()
            if len(votes) > 1 and votes[0][1] == votes[1][1]:
                ties.append(item)
                continue
            label = votes[0][0]
            cite = sum(x["wrong_cite"] for x in xs) * 2 > len(xs)
        out.append({**base, "label": label, "wrong_cite": cite})
    return out, ties


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make", help="tạo bảng gán nhãn")
    mk.add_argument("--gen", required=True, help="file câu trả lời của generate.py (TV3)")
    mk.add_argument("--questions", nargs="+", default=None)
    mk.add_argument("--chunks", default="data/corpus/chunks.jsonl")
    mk.add_argument("--configs", nargs="+", default=["S0", "S2", "S3"])
    mk.add_argument("--n-questions", type=int, default=60)
    mk.add_argument("--min-ooc", type=int, default=15)
    mk.add_argument("--n-shared", type=int, default=30)
    mk.add_argument("--annotators", nargs="+", default=["TV1", "TV2", "TV3"])
    mk.add_argument("--seed", type=int, default=2026)
    mk.add_argument("--out-dir", default=str(ANN_DIR))
    ck = sub.add_parser("check", help="kiểm tra nhãn đã gán (thiếu, sai mã, bất đồng)")
    ck.add_argument("--ann-dir", default=str(ANN_DIR))
    a = ap.parse_args()

    if a.cmd == "make":
        questions = []
        for p in a.questions or sorted(glob.glob(DEFAULT_QUESTIONS)):
            questions += read_jsonl(p)
        sheets, key = make_sheets(questions, read_jsonl(a.gen), read_jsonl(a.chunks), a.configs,
                                  a.n_questions, a.min_ooc, a.n_shared, a.annotators, a.seed)
        out = Path(a.out_dir)
        for who, rows in sheets.items():
            write_csv(out / f"sheet_{who}.csv", SHEET_COLUMNS, rows)
        write_csv(out / "key.csv", KEY_COLUMNS, key)
        n_q = len({r["qid"] for r in key})
        print(f"{n_q} câu × {len(a.configs)} cấu hình = {len(key)} mục; "
              f"loại câu: {dict(Counter(r['type'] for r in key if r['config'] == a.configs[0]))}")
        print("Số mục mỗi người:", {w: len(r) for w, r in sheets.items()},
              f"(trong đó {sum(r['nguoi_gan'] == SHARED for r in key)} mục chung)")
        print(f"Đã ghi vào {out}. KHÔNG gửi key.csv cho người gán.")
    else:
        labels, problems = load_labels(a.ann_dir)
        for p in problems:
            print("-", p)
        final, ties = final_labels(labels, a.ann_dir)
        print(f"{len(labels)} nhãn hợp lệ, {len(final)} mục có nhãn cuối, {len(problems)} vấn đề")
        if ties:
            print("Mục chung hoà phiếu, cần bàn và ghi vào labels/final.csv:", ties)


if __name__ == "__main__":
    main()

"""Phân tích ảo giác theo loại câu hỏi và theo việc retriever có lấy được chunk đúng hay không.

    python scripts/hallucination_breakdown.py --gen outputs/generations.jsonl

Đọc:
- câu trả lời của TV3 (generate.py), bộ câu hỏi data/rag_test/*.jsonl
- nhãn thủ công trong results/hallucination/annotation/labels/*.csv (nếu đã có)

Ghi results/hallucination/BREAKDOWN.md và summary.json (scripts/plot_figures.py vẽ hình từ đây):
- Bảng A (nhãn người): phân bố nhãn và tỉ lệ ảo giác theo cấu hình.
- Bảng B (nhãn người): tỉ lệ ảo giác theo loại câu hỏi × cấu hình.
- Bảng C (nhãn người): retriever lấy trúng (hit) vs trượt (miss) chunk đúng.
- Bảng D (chấm tự động, mọi cấu hình có truy hồi): token-F1 và tỉ lệ từ chối sai theo hit/miss.

Tỉ lệ ảo giác = (nhãn 3 + nhãn 4) / số câu trả lời (mục 7 kế hoạch).
"hit" = ngữ cảnh đưa cho LLM chứa ít nhất một chunk vàng (gold_chunks) của câu hỏi.
"""
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.hallucination.annotation import (  # noqa: E402
    ANN_DIR, DEFAULT_QUESTIONS, HALLUCINATION, LABELS, final_labels, load_labels, qtype, read_jsonl)

OUT_DIR = Path("results/hallucination")
TYPE_ORDER = ["fact", "number", "condition", "multi", "out_of_corpus"]


def pct(a, b):
    return f"{100 * a / b:.1f}% ({a}/{b})" if b else "–"


def rate(a, b):
    return round(a / b, 4) if b else None


def hit_of(gen_row, q):
    """True/False nếu câu có đáp án và cấu hình có ngữ cảnh; None với câu ngoài kho hoặc S0."""
    if not q.get("answerable", True) or not gen_row.get("chunk_ids"):
        return None
    return bool(set(q.get("gold_chunks") or []) & set(gen_row["chunk_ids"]))


# ------------------------------------------------------------------ nhãn người
def human_tables(final, qs, gens):
    rows = []
    for x in final:
        q = qs[x["qid"]]
        rows.append({**x, "type": qtype(q), "answerable": bool(q.get("answerable", True)),
                     "hit": hit_of(gens[(x["qid"], x["config"])], q)})
    cfgs = sorted({r["config"] for r in rows})
    md, summary = [], {}

    md += ["## Bảng A. Phân bố nhãn theo cấu hình (nhãn người)\n",
           "| Cấu hình | n | " + " | ".join(f"{k}. {v}" for k, v in LABELS.items())
           + " | **Tỉ lệ ảo giác** | Từ chối đúng (câu ngoài kho) | Từ chối sai (câu có đáp án) | Trích dẫn sai |",
           "|" + "---|" * (len(LABELS) + 6)]
    for c in cfgs:
        rs = [r for r in rows if r["config"] == c]
        cnt = {k: sum(r["label"] == k for r in rs) for k in LABELS}
        hal = sum(cnt[k] for k in HALLUCINATION)
        ooc = [r for r in rs if not r["answerable"]]
        inc = [r for r in rs if r["answerable"]]
        ref_ok = sum(r["label"] == 5 for r in ooc)
        ref_bad = sum(r["label"] == 6 for r in inc)
        cite = sum(r["wrong_cite"] for r in rs)
        md.append(f"| {c} | {len(rs)} | " + " | ".join(str(cnt[k]) for k in LABELS)
                  + f" | **{pct(hal, len(rs))}** | {pct(ref_ok, len(ooc))} | {pct(ref_bad, len(inc))} "
                  f"| {pct(cite, len(rs))} |")
        summary[c] = {"n": len(rs), "hallucination_rate": rate(hal, len(rs)), "n_hallucination": hal,
                      "refusal_acc_ooc": rate(ref_ok, len(ooc)), "false_refusal": rate(ref_bad, len(inc)),
                      "labels": cnt}

    md += ["\n## Bảng B. Tỉ lệ ảo giác theo loại câu hỏi (nhãn người)\n",
           "| Loại câu | " + " | ".join(cfgs) + " |", "|" + "---|" * (len(cfgs) + 1)]
    types = [t for t in TYPE_ORDER if any(r["type"] == t for r in rows)]
    types += sorted({r["type"] for r in rows} - set(types))
    by_type = {}
    for t in types:
        cells = []
        for c in cfgs:
            rs = [r for r in rows if r["config"] == c and r["type"] == t]
            hal = sum(r["label"] in HALLUCINATION for r in rs)
            cells.append(pct(hal, len(rs)))
            by_type.setdefault(t, {})[c] = rate(hal, len(rs))
        md.append(f"| {t} | " + " | ".join(cells) + " |")

    md += ["\n## Bảng C. Retriever lấy trúng vs trượt chunk đúng (nhãn người, câu có đáp án)\n",
           "| Cấu hình | Nhóm | n | Đúng (nhãn 1+2) | Ảo giác (3+4) | Từ chối sai (6) |",
           "|---|---|---|---|---|---|"]
    hitmiss = {}
    for c in cfgs:
        for name, flag in (("hit", True), ("miss", False)):
            rs = [r for r in rows if r["config"] == c and r["hit"] is flag]
            if not rs:
                continue
            ok = sum(r["label"] in (1, 2) for r in rs)
            hal = sum(r["label"] in HALLUCINATION for r in rs)
            bad = sum(r["label"] == 6 for r in rs)
            md.append(f"| {c} | {name} | {len(rs)} | {pct(ok, len(rs))} | {pct(hal, len(rs))} | {pct(bad, len(rs))} |")
            hitmiss.setdefault(c, {})[name] = {"n": len(rs), "hallucination_rate": rate(hal, len(rs)),
                                               "correct_rate": rate(ok, len(rs))}
    md.append("\nS0 không có ngữ cảnh nên không có hit/miss; S3 (oracle) luôn hit.")
    return md, {"by_config": summary, "by_type": by_type, "hit_miss": hitmiss}


# ------------------------------------------------------------------ chấm tự động
def auto_table(gens, qs):
    from rag.generation.generate import score_row

    groups = defaultdict(list)
    for (qid, cfg), g in gens.items():
        q = qs.get(qid)
        h = hit_of(g, q) if q else None
        if h is None:
            continue
        sample = {"qid": qid, "answer": q["answer"], "in_corpus": True}
        groups[(cfg, "hit" if h else "miss")].append(score_row(g, sample))
    md = ["## Bảng D. Chấm tự động theo hit/miss (câu có đáp án, mọi cấu hình có ngữ cảnh)\n",
          "| Cấu hình | Nhóm | n | token-F1 | Chứa đáp án | Từ chối sai |", "|---|---|---|---|---|---|"]
    out = {}
    for (cfg, name) in sorted(groups):
        xs = groups[(cfg, name)]
        f1 = sum(x["token_f1"] for x in xs) / len(xs)
        gold = sum(x["contains_gold"] for x in xs)
        ref = sum(x["refused"] for x in xs)
        md.append(f"| {cfg} | {name} | {len(xs)} | {100 * f1:.1f} | {pct(gold, len(xs))} | {pct(ref, len(xs))} |")
        out.setdefault(cfg, {})[name] = {"n": len(xs), "token_f1": round(f1, 4),
                                         "false_refusal": rate(ref, len(xs))}
    return md, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="file câu trả lời của generate.py (TV3)")
    ap.add_argument("--questions", nargs="+", default=None)
    ap.add_argument("--ann-dir", default=str(ANN_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args()

    qs = {}
    for p in a.questions or sorted(glob.glob(DEFAULT_QUESTIONS)):
        for q in read_jsonl(p):
            qs[q["id"]] = q
    gens = {(str(r["qid"]), r["config"]): r for r in read_jsonl(a.gen)}
    models = sorted({str(r.get("model_id")) for r in gens.values()})

    md = ["# Phân tích ảo giác theo loại câu hỏi và chất lượng truy hồi\n",
          f"{len(qs)} câu hỏi; câu trả lời từ `{a.gen}` (mô hình: {', '.join(models)}).\n"]
    summary = {"models": models}
    if (Path(a.ann_dir) / "key.csv").exists():
        labels, problems = load_labels(a.ann_dir)
        final, ties = final_labels(labels, a.ann_dir)
        if problems or ties:
            print(f"[cảnh báo] {len(problems)} vấn đề khi đọc nhãn, {len(ties)} mục hoà phiếu "
                  "(chạy: python -m rag.hallucination.annotation check)")
        if final:
            hmd, hs = human_tables(final, qs, gens)
            md += hmd + [""]
            summary["human"] = hs
            md.append(f"_{len(final)} mục có nhãn cuối; {len(ties)} mục chung hoà phiếu chưa tính._\n")
    if "human" not in summary:
        md.append("_Chưa có nhãn thủ công: chỉ có bảng chấm tự động._\n")
    dmd, ds = auto_table(gens, qs)
    md += dmd
    summary["auto_hit_miss"] = ds

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "BREAKDOWN.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(md))
    print(f"\nĐã ghi {out / 'BREAKDOWN.md'} và {out / 'summary.json'}")


if __name__ == "__main__":
    main()

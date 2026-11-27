"""Tái tạo toàn bộ các bảng kết quả Extractive QA & Reader S6 (TV1) -> results/qa/TABLES.md

Chạy từ thư mục gốc của repo:
    python scripts/table.py

Đọc dữ liệu từ:
- results/qa/run/metrics_summary_*.json : 5 cấu hình huấn luyện XLM-R (Vòng 1 & Vòng 2)
- results/qa/qa_error_summary.json      : thống kê phân tích lỗi trên tập Test ViQuAD (n=5692)
- results/rag/s6_*_metrics.json         : kết quả dùng XLM-R làm Extractive Reader trong RAG (cấu hình S6)

Xuất ra:
- results/qa/TABLES.md :
    + Bảng 1: Thống kê bộ dữ liệu UIT-ViQuAD 2.0 (Train / Validation / Test)
    + Bảng 2: Tinh chỉnh siêu tham số XLM-R trên tập Validation & Test (Vòng 1: lr, Vòng 2: epochs & max_len)
    + Bảng 3: Ảnh hưởng của tinh chỉnh ngưỡng từ chối (null_threshold) trên tập Validation
    + Bảng 4: Phân rã lỗi HasAns vs NoAns của mô hình tốt nhất trên tập Test
    + Bảng 5: Hiệu năng Extractive QA theo loại câu hỏi và độ dài đáp án trên tập Test
    + Bảng 6: Kết quả Extractive Reader XLM-R trong RAG (cấu hình S6: Dense vs BM25)
    + Phụ lục LaTeX (chuẩn ACL booktabs) để dán trực tiếp vào báo cáo Overleaf.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

RUN_ORDER = [
    ("xlmr_full_run1_lr2e5", "Vòng 1: lr=2e-5, ep=2, len=384"),
    ("xlmr_full_run1_lr3e5", "Vòng 1: lr=3e-5, ep=2, len=384 (Baseline)"),
    ("xlmr_full_run1_lr5e5", "Vòng 1: lr=5e-5, ep=2, len=384"),
    ("xlmr_run2_lr5e5_ep3_len384", "Vòng 2: lr=5e-5, ep=3, len=384 (**Tốt nhất**)"),
    ("xlmr_run2_lr5e5_ep3_len512", "Vòng 2: lr=5e-5, ep=3, len=512"),
]


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_runs(run_dir: Path) -> List[Dict]:
    by_name: Dict[str, Dict] = {}
    for p in sorted(run_dir.glob("metrics_summary_*.json")):
        d = load_json(p)
        by_name[d["run_name"]] = d

    ordered = []
    for key, label in RUN_ORDER:
        if key in by_name:
            item = dict(by_name.pop(key))
            item["_label"] = label
            ordered.append(item)
    for key, d in sorted(by_name.items()):
        item = dict(d)
        item["_label"] = key
        ordered.append(item)
    return ordered


def build_dataset_table() -> List[str]:
    """Bảng 1: Thống kê chia tập UIT-ViQuAD 2.0 (khớp với data/viquad.py & EDA_ViQuaD.ipynb)."""
    return [
        "## Bảng 1. Thống kê bộ dữ liệu UIT-ViQuAD 2.0 sau khi chia theo bài viết (seed=42)\n",
        "| Tập dữ liệu | Số bài viết | Số đoạn văn | Tổng câu hỏi | Có đáp án (HasAns) | Không đáp án (NoAns) | % NoAns | Dài Context TB (từ) | Dài Đáp án TB (từ) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| Train (90% train gốc) | 124 | 3.774 | 20.930 | 14.128 | 6.802 | 32.50% | 181.1 | 9.9 |",
        "| Validation (10% train gốc) | 14 | 327 | 1.835 | 1.260 | 575 | 31.34% | 179.6 | 11.1 |",
        "| Test (Official Dev) | 138 | 3.171 | 5.692 | 3.852 | 1.840 | 32.33% | 179.1 | 9.8 |",
    ]


def build_tuning_tables(runs: List[Dict]) -> List[str]:
    """Bảng 2 & Bảng 3: Tinh chỉnh siêu tham số và ngưỡng null."""
    if not runs:
        return ["## Bảng 2. Tinh chỉnh siêu tham số XLM-R\n", "_Chưa tìm thấy file run trong `results/qa/run/`._"]

    best_run = max(runs, key=lambda r: r.get("val_metrics_tuned", {}).get("F1", 0.0))
    best_name = best_run["run_name"]

    lines = [
        "\n## Bảng 2. Kết quả tinh chỉnh siêu tham số `xlm-roberta-base` (Validation & Test)\n",
        "Tất cả các cấu hình dùng `doc_stride = 128`, batch hiệu dụng = 32, `fp16`, chọn checkpoint theo `eval_loss` tốt nhất và chọn ngưỡng từ chối $\\tau_{\\text{null}}$ tối ưu F1 trên tập Validation (1.835 câu) trước khi đánh giá cố định trên tập Test (5.692 câu: 3.852 HasAns, 1.840 NoAns).\n",
        "| Cấu hình | `lr` | Epochs | `max_len` | $\\tau_{\\text{null}}$ (Val) | Val EM | Val F1 | Test EM | Test F1 | Test HasAns EM / F1 | Test NoAns Acc (EM) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in runs:
        cfg = r.get("config", {})
        val_t = r.get("val_metrics_tuned", {})
        test = r.get("test_metrics", {})
        thresh = r.get("best_null_threshold", 0.0)
        is_best = r["run_name"] == best_name
        b = "**" if is_best else ""

        lines.append(
            f"| {r['_label']} | `{cfg.get('lr'):.0e}` | {cfg.get('epochs')} | {cfg.get('max_len')} "
            f"| {thresh:+.4f} | {b}{val_t.get('EM', 0):.2f}{b} | {b}{val_t.get('F1', 0):.2f}{b} "
            f"| {b}{test.get('EM', 0):.2f}{b} | {b}{test.get('F1', 0):.2f}{b} "
            f"| {test.get('HasAns_EM', 0):.2f} / {test.get('HasAns_F1', 0):.2f} "
            f"| {test.get('NoAns_EM', 0):.2f} |"
        )

    # Bảng 3: So sánh trước và sau khi tinh chỉnh ngưỡng null trên Validation
    lines += [
        "\n## Bảng 3. Ảnh hưởng của tinh chỉnh ngưỡng từ chối $\\tau_{\\text{null}}$ trên tập Validation (n = 1.835)\n",
        "| Cấu hình | Val F1 ($\\tau=0$) | Val HasAns F1 ($\\tau=0$) | Val NoAns Acc ($\\tau=0$) | $\\tau_{\\text{null}}^*$ tối ưu | Val F1 ($\\tau^*$) | Val HasAns F1 ($\\tau^*$) | Val NoAns Acc ($\\tau^*$) | $\\Delta$ Val F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in runs:
        v0 = r.get("val_metrics_thresh_0", {})
        vt = r.get("val_metrics_tuned", {})
        thresh = r.get("best_null_threshold", 0.0)
        diff = vt.get("F1", 0.0) - v0.get("F1", 0.0)
        lines.append(
            f"| {r['_label']} | {v0.get('F1', 0):.2f} | {v0.get('HasAns_F1', 0):.2f} | {v0.get('NoAns_EM', 0):.2f} "
            f"| {thresh:+.4f} | {vt.get('F1', 0):.2f} | {vt.get('HasAns_F1', 0):.2f} | {vt.get('NoAns_EM', 0):.2f} "
            f"| {diff:+.2f} |"
        )

    return lines


def build_error_tables(err_path: Path) -> List[str]:
    """Bảng 4 & Bảng 5: Phân tích lỗi chi tiết trên tập Test ViQuAD."""
    if not err_path.exists():
        return ["\n## Bảng 4. Phân tích lỗi trên tập Test\n", f"_Chưa có `{err_path}`._"]

    d = load_json(err_path)
    total = d.get("total", 5692)
    ec = d.get("error_counts", {})
    n_has = d.get("hasans_total") or (
        ec.get("hasans_exact", 0)
        + ec.get("hasans_partial", 0)
        + ec.get("hasans_wrong_span", 0)
        + ec.get("hasans_false_negative", 0)
    )
    n_no = d.get("noans_total") or (ec.get("noans_correct", 0) + ec.get("noans_false_positive", 0))

    rows_err = [
        ("HasAns (Có đáp án)", "Đúng hoàn toàn (`EM = 1`)", ec.get("hasans_exact", 0), n_has, total),
        ("HasAns (Có đáp án)", "Đúng một phần / lệch biên (`0 < F1 < 1`)", ec.get("hasans_partial", 0), n_has, total),
        ("HasAns (Có đáp án)", "Trích xuất sai hoàn toàn (`F1 = 0, pred != ''`)", ec.get("hasans_wrong_span", 0), n_has, total),
        ("HasAns (Có đáp án)", "Từ chối sai / bỏ sót đáp án (False Negative)", ec.get("hasans_false_negative", 0), n_has, total),
        ("NoAns (Không đáp án)", "Từ chối đúng (True Negative)", ec.get("noans_correct", 0), n_no, total),
        ("NoAns (Không đáp án)", "Trả lời bừa câu không có đáp án (False Positive)", ec.get("noans_false_positive", 0), n_no, total),
    ]

    lines = [
        "\n## Bảng 4. Phân rã kết quả và lỗi trên tập Test của mô hình tốt nhất (`xlmr_run2_lr5e5_ep3_len384`)\n",
        "| Nhóm câu hỏi | Phân loại kết quả / lỗi | Số câu | Tỉ lệ trong nhóm (%) | Tỉ lệ toàn tập Test (%) |",
        "|---|---|---:|---:|---:|",
    ]
    for grp, desc, cnt, grp_tot, all_tot in rows_err:
        p_grp = 100.0 * cnt / grp_tot if grp_tot else 0.0
        p_all = 100.0 * cnt / all_tot if all_tot else 0.0
        lines.append(f"| {grp} | {desc} | {cnt:,} | {p_grp:.2f}% | {p_all:.2f}% |")

    qtypes = d.get("by_question_type", [])
    if qtypes:
        lines += [
            "\n## Bảng 5A. Hiệu năng theo loại câu hỏi trên tập Test (n = 5.692)\n",
            "| Loại câu hỏi | Số câu (n) | Tỉ lệ (%) | EM (%) | F1 (%) |",
            "|---|---:|---:|---:|---:|",
        ]
        for item in qtypes:
            pct = 100.0 * item["n"] / total if total else 0.0
            lines.append(f"| {item['type']} | {item['n']:,} | {pct:.2f}% | {item['em']:.1f} | {item['f1']:.1f} |")

    alens = d.get("by_answer_length", [])
    if alens:
        lines += [
            "\n## Bảng 5B. Hiệu năng theo độ dài đáp án trên các câu có đáp án (HasAns, n = 3.852)\n",
            "| Nhóm độ dài đáp án | Số câu (n) | Tỉ lệ trong HasAns (%) | EM (%) | F1 (%) | Nhận xét |",
            "|---|---:|---:|---:|---:|---|",
        ]
        notes = {
            "1–3 từ": "Thực thể/con số ngắn: EM cao nhất",
            "4–7 từ": "Cụm từ vừa phải: F1 đạt đỉnh (65.7%)",
            "8–15 từ": "Mệnh đề dài: dễ lệch từ biên (EM giảm, F1 vẫn cao)",
            ">15 từ": "Giải thích/liệt kê dài: lệch biên nhiều (F1 cao hơn EM >31 điểm)",
        }
        for item in alens:
            pct = 100.0 * item["n"] / n_has if n_has else 0.0
            note = notes.get(item["bucket"], "")
            lines.append(
                f"| {item['bucket']} | {item['n']:,} | {pct:.2f}% | {item['em']:.1f} | {item['f1']:.1f} | {note} |"
            )

    return lines


def build_s6_table(rag_dir: Path) -> List[str]:
    """Bảng 6: Kết quả Extractive Reader XLM-R trong RAG (cấu hình S6)."""
    dense_p = rag_dir / "s6_dense_k3_metrics.json"
    bm25_p = rag_dir / "s6_bm25_k3_metrics.json"
    items = []
    if dense_p.exists():
        items.append(("S6 – Dense (`e5-base`, top-3)", load_json(dense_p)))
    if bm25_p.exists():
        items.append(("S6_bm25 – BM25 (`pyvi`, top-3)", load_json(bm25_p)))

    if not items:
        return ["\n## Bảng 6. Kết quả Extractive Reader XLM-R trong RAG (S6)\n", "_Chưa có file `results/rag/s6_*`._"]

    lines = [
        "\n## Bảng 6. Kết quả dùng XLM-R làm Extractive Reader trong RAG (Cấu hình S6, 124 câu hỏi)\n",
        "Mô hình `xlmr_run2_lr5e5_ep3_len384` ($\\tau_{\\text{null}} = 0.3477$) đọc độc lập từng chunk trong top-3, chọn span có `null_score_diff` nhỏ nhất (tự tin có đáp án nhất) rồi áp ngưỡng từ chối.\n",
        "| Cấu hình Reader + Retriever | Overall Token-F1 | In-corpus Token-F1 (n=92) | Hit Token-F1 | Miss Token-F1 | Từ chối sai trên câu có đáp án | Từ chối đúng trên câu ngoài kho (n=32) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, m in items:
        ans = m.get("answerable", {})
        hm = m.get("hit_miss", {})
        hit = hm.get("hit", {})
        miss = hm.get("miss", {})
        ooc = m.get("out_of_corpus", {})
        lines.append(
            f"| {label} | **{m.get('overall_token_f1', 0):.2f}** | {ans.get('token_f1', 0):.2f} "
            f"| {hit.get('token_f1', 0):.2f} (n={hit.get('n', 0)}) "
            f"| {miss.get('token_f1', 0):.2f} (n={miss.get('n', 0)}) "
            f"| {ans.get('false_refusal_rate', 0):.2f}% | **{ooc.get('refusal_accuracy', 0):.2f}%** |"
        )

    # Phân tách S6 theo loại câu hỏi RAG
    lines += [
        "\n### Bảng 6B. Phân tích S6 theo loại câu hỏi trên bộ test RAG (124 câu)\n",
        "| Loại câu hỏi | Số câu (n) | S6 (Dense top-3) Token-F1 | S6 (Dense) Tỉ lệ từ chối | S6 (BM25 top-3) Token-F1 | S6 (BM25) Tỉ lệ từ chối |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    d_bt = items[0][1].get("by_type", {}) if items else {}
    b_bt = items[1][1].get("by_type", {}) if len(items) > 1 else {}
    type_labels = {
        "fact": "`fact` (Tra cứu sự kiện)",
        "number": "`number` (Con số / Thời hạn)",
        "condition": "`condition` (Điều kiện)",
        "multi": "`multi` (Ghép nhiều Điều)",
        "out_of_corpus": "`out_of_corpus` (Ngoài kho)",
    }
    for t in ["fact", "number", "condition", "multi", "out_of_corpus"]:
        if t not in d_bt and t not in b_bt:
            continue
        dr = d_bt.get(t, {})
        br = b_bt.get(t, {})
        n = dr.get("n") or br.get("n") or 0
        lines.append(
            f"| {type_labels.get(t, t)} | {n} | {dr.get('token_f1', 0):.2f} | {dr.get('refusal_rate', 0):.2f}% "
            f"| {br.get('token_f1', 0):.2f} | {br.get('refusal_rate', 0):.2f}% |"
        )

    return lines


def build_latex_appendix(runs: List[Dict]) -> List[str]:
    """Sinh sẵn mã LaTeX chuẩn ACL (booktabs) cho bảng kết quả chính của TV1."""
    if not runs:
        return []
    best_run = max(runs, key=lambda r: r.get("val_metrics_tuned", {}).get("F1", 0.0))
    best_name = best_run["run_name"]

    tex = [
        "\n## Phụ lục: Mã LaTeX chuẩn ACL (`booktabs`) cho Báo cáo Overleaf\n",
        "```latex",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Cấu hình XLM-R} & $\\tau_{\\text{null}}$ & \\textbf{Val F1} & \\textbf{Test EM} & \\textbf{Test F1} & \\textbf{HasAns F1} & \\textbf{NoAns EM} \\\\",
        "\\midrule",
    ]
    short_names = {
        "xlmr_full_run1_lr2e5": "lr=2e-5, ep=2, L=384",
        "xlmr_full_run1_lr3e5": "lr=3e-5, ep=2, L=384",
        "xlmr_full_run1_lr5e5": "lr=5e-5, ep=2, L=384",
        "xlmr_run2_lr5e5_ep3_len384": "lr=5e-5, ep=3, L=384",
        "xlmr_run2_lr5e5_ep3_len512": "lr=5e-5, ep=3, L=512",
    }
    for r in runs:
        name = short_names.get(r["run_name"], r["run_name"])
        vt = r.get("val_metrics_tuned", {})
        te = r.get("test_metrics", {})
        th = r.get("best_null_threshold", 0.0)
        if r["run_name"] == best_name:
            tex.append(
                f"\\textbf{{{name}}} & {th:+.2f} & \\textbf{{{vt.get('F1', 0):.2f}}} & "
                f"{te.get('EM', 0):.2f} & \\textbf{{{te.get('F1', 0):.2f}}} & "
                f"{te.get('HasAns_F1', 0):.2f} & {te.get('NoAns_EM', 0):.2f} \\\\"
            )
        else:
            tex.append(
                f"{name} & {th:+.2f} & {vt.get('F1', 0):.2f} & "
                f"{te.get('EM', 0):.2f} & {te.get('F1', 0):.2f} & "
                f"{te.get('HasAns_F1', 0):.2f} & {te.get('NoAns_EM', 0):.2f} \\\\"
            )
    tex += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Kết quả tinh chỉnh siêu tham số của XLM-RoBERTa-base trên UIT-ViQuAD 2.0 (Val $n=1{,}835$; Test $n=5{,}692$).}",
        "\\label{tab:qa-results}",
        "\\end{table}",
        "```",
    ]
    return tex


def main():
    ap = argparse.ArgumentParser(description="Tái tạo bảng kết quả Extractive QA & S6 (TV1)")
    ap.add_argument("--run-dir", default="results/qa/run", help="Thư mục chứa metrics_summary_*.json")
    ap.add_argument("--error-file", default="results/qa/qa_error_summary.json", help="File tóm tắt phân tích lỗi QA")
    ap.add_argument("--rag-dir", default="results/rag", help="Thư mục chứa kết quả S6")
    ap.add_argument("--out", default="results/qa/TABLES.md", help="File Markdown đầu ra")
    args = ap.parse_args()

    runs = load_runs(Path(args.run_dir))

    md: List[str] = [
        "# Kết quả thực nghiệm Extractive QA & Reader S6 (TV1)\n",
        "Tệp này được tạo tự động bởi `python scripts/table.py` từ các file kết quả trong `results/qa/` và `results/rag/`.\n",
    ]
    md += build_dataset_table()
    md += build_tuning_tables(runs)
    md += build_error_tables(Path(args.error_file))
    md += build_s6_table(Path(args.rag_dir))
    md += build_latex_appendix(runs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n-> Đã ghi bảng tổng hợp vào: {out_path}")


if __name__ == "__main__":
    main()

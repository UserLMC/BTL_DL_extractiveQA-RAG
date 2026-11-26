"""Vẽ hình cho báo cáo (khổ một cột ACL) -> results/figures/*.pdf và *.png

    python scripts/plot_figures.py

Hình 1: Recall@k (k = 1..10) của các retriever    <- results/retrieval/recall_by_k.csv
        (tạo bằng: python -m rag.retrieval.export_topk)
Hình 2: tỉ lệ ảo giác theo cấu hình (nhãn người)  <- results/hallucination/summary.json
        (tạo bằng: python scripts/hallucination_breakdown.py --gen ...); bỏ qua nếu chưa có nhãn.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Bảng màu phân loại cố định theo thứ tự (đã kiểm tra tách biệt cho người mù màu trên nền trắng).
# Cam và xanh ngọc có tương phản < 3:1 với nền nên mỗi đường đều có nhãn trực tiếp + ký hiệu riêng.
SERIES = ["#257de0", "#f47230", "#00b894"]
MARKERS = ["o", "s", "^"]
TEXT = "#333333"
MUTED = "#6b6b6b"
GRID = "#e6e6e6"
WIDTH_IN = 3.15  # một cột ACL

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "figure.facecolor": "white", "axes.facecolor": "white",
    "pdf.fonttype": 42,  # nhúng font TrueType: chữ tiếng Việt chọn/tìm được trong PDF
})


def _axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("->", out / f"{name}.pdf")


def recall_figure(csv_path, out):
    curves = defaultdict(dict)
    n = None
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            curves[r["retriever"]][int(r["k"])] = float(r["recall"])
            n = r["n"]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.3))
    _axes(ax)
    ends = []
    for i, (name, pts) in enumerate(curves.items()):
        ks = sorted(pts)
        ys = [pts[k] for k in ks]
        ax.plot(ks, ys, color=SERIES[i % len(SERIES)], linewidth=1.5, marker=MARKERS[i % len(MARKERS)],
                markersize=4, markeredgecolor="white", markeredgewidth=0.8, label=name)
        ends.append([ys[-1], ks[-1], name])
    y_lo = max(0.0, min(min(p.values()) for p in curves.values()) - 0.1)
    # nhãn cuối đường: giãn theo chiều dọc để không chồng nhau
    gap = (1.0 - y_lo) * 0.075
    ends.sort(reverse=True)
    for j in range(1, len(ends)):
        ends[j][0] = min(ends[j][0], ends[j - 1][0] - gap)
    for y, k, name in ends:
        ax.annotate(name, (k, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7, color=TEXT, annotation_clip=False)
    ax.axvline(3, color=MUTED, linewidth=0.6, linestyle=(0, (2, 2)))
    ax.text(3.1, y_lo + 0.01, "k = 3 (đưa vào LLM)", fontsize=6.5, color=MUTED, va="bottom")
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.7, 10.3)
    ax.set_ylim(y_lo, 1)
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_title(f"Recall@k trên {n} câu có đáp án", fontsize=8, color=TEXT, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 0.08))
    fig.subplots_adjust(right=0.78)
    save(fig, out, "recall_at_k")


def hallucination_figure(summary_path, out):
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    by_cfg = summary.get("human", {}).get("by_config")
    if not by_cfg:
        print("(bỏ qua hình 2: chưa có nhãn thủ công trong", summary_path, ")")
        return
    cfgs = sorted(by_cfg)
    vals = [100 * (by_cfg[c]["hallucination_rate"] or 0) for c in cfgs]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.1))
    _axes(ax)
    ax.bar(range(len(cfgs)), vals, width=0.55, color=SERIES[0], edgecolor="white", linewidth=1.5)
    for i, c in enumerate(cfgs):
        ax.text(i, vals[i] + 1, f"{vals[i]:.1f}%\n({by_cfg[c]['n_hallucination']}/{by_cfg[c]['n']})",
                ha="center", va="bottom", fontsize=6.5, color=TEXT)
    ax.set_xticks(range(len(cfgs)), cfgs)
    ax.set_ylim(0, max(vals + [10]) * 1.35)
    ax.set_ylabel("Tỉ lệ ảo giác (%)")
    ax.set_title("Tỉ lệ ảo giác theo cấu hình (nhãn người)", fontsize=8, color=TEXT, loc="left")
    save(fig, out, "hallucination_by_config")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recall", default="results/retrieval/recall_by_k.csv")
    ap.add_argument("--summary", default="results/hallucination/summary.json")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    out = Path(a.out)
    recall_figure(a.recall, out)
    if Path(a.summary).exists():
        hallucination_figure(a.summary, out)
    else:
        print("(bỏ qua hình 2: chưa có", a.summary, ")")


if __name__ == "__main__":
    main()

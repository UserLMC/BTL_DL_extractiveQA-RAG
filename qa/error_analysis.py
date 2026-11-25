import argparse
import collections
import json
import os
import re
from typing import Dict, List


def classify_question_type(question: str) -> str:
    """Phân loại câu hỏi tiếng Việt dựa trên từ nghi vấn đặc trưng."""
    q = question.lower().strip()
    patterns = [
        ("Khi nào / Thời gian", r"\b(khi nào|bao giờ|năm nào|tháng nào|ngày nào|thời gian nào|thế kỷ nào|lúc nào)\b"),
        ("Bao nhiêu / Con số", r"\b(bao nhiêu|mấy|tỉ lệ|tỷ lệ|số lượng|bao lâu|dài bao nhiêu|cao bao nhiêu)\b"),
        ("Ở đâu / Địa điểm", r"\b(ở đâu|nơi nào|địa điểm nào|quốc gia nào|thành phố nào|tỉnh nào|vùng nào)\b"),
        ("Ai / Con người", r"\b(ai là|là ai|những ai|của ai|với ai|do ai|người nào)\b|^ai\b"),
        ("Tại sao / Nguyên nhân", r"\b(tại sao|vì sao|nguyên nhân|lý do|do đâu)\b"),
        ("Như thế nào / Cách thức", r"\b(như thế nào|ra sao|bằng cách nào|cách nào|thế nào)\b"),
        ("Cái gì / Sự vật", r"\b(cái gì|gì|là gì|điều gì|loại nào|tên gì)\b"),
    ]
    for label, pat in patterns:
        if re.search(pat, q):
            return label
    return "Khác"


def answer_length_bucket(gold_answers: List[str]) -> str:
    if not gold_answers:
        return "NoAns (0 từ)"
    # Lấy độ dài theo số âm tiết của đáp án ngắn nhất trong danh sách chuẩn
    min_len = min(len(a.split()) for a in gold_answers)
    if min_len <= 3:
        return "Ngắn (1–3 từ)"
    elif min_len <= 10:
        return "Trung bình (4–10 từ)"
    else:
        return "Dài (>10 từ)"


def summarize_group(items: List[Dict]) -> Dict:
    n = len(items)
    if n == 0:
        return {"count": 0, "EM": 0.0, "F1": 0.0}
    return {
        "count": n,
        "EM": round(100.0 * sum(x["em"] for x in items) / n, 2),
        "F1": round(100.0 * sum(x["f1"] for x in items) / n, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Phân tích lỗi Extractive QA từ test_predictions.json")
    parser.add_argument("--pred_file", type=str, required=True, help="Đường dẫn tới test_predictions.json")
    parser.add_argument("--out_dir", type=str, default="outputs/error_analysis")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.pred_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    # 1. Phân tích theo loại câu hỏi
    by_qtype = collections.defaultdict(list)
    for r in records:
        by_qtype[classify_question_type(r["question"])].append(r)

    # 2. Phân tích theo độ dài đáp án
    by_len = collections.defaultdict(list)
    for r in records:
        by_len[answer_length_bucket(r["gold_answers"])].append(r)

    # 3. Ma trận lỗi HasAns vs NoAns
    error_buckets = {
        "HasAns_Exact (EM=1)": [],
        "HasAns_Partial (0<F1<1)": [],
        "HasAns_WrongSpan (F1=0, pred!='')": [],
        "HasAns_FalseAbstain (bỏ qua câu có đáp án)": [],
        "NoAns_CorrectAbstain (từ chối đúng)": [],
        "NoAns_FalseAnswer (trả lời bừa câu NoAns)": [],
    }
    for r in records:
        pred = r["final_pred"].strip()
        if r["is_impossible"]:
            if pred == "":
                error_buckets["NoAns_CorrectAbstain (từ chối đúng)"].append(r)
            else:
                error_buckets["NoAns_FalseAnswer (trả lời bừa câu NoAns)"].append(r)
        else:
            if pred == "":
                error_buckets["HasAns_FalseAbstain (bỏ qua câu có đáp án)"].append(r)
            elif r["em"] == 1.0:
                error_buckets["HasAns_Exact (EM=1)"].append(r)
            elif r["f1"] > 0.0:
                error_buckets["HasAns_Partial (0<F1<1)"].append(r)
            else:
                error_buckets["HasAns_WrongSpan (F1=0, pred!='')"].append(r)

    report = {
        "overall": summarize_group(records),
        "by_question_type": {k: summarize_group(v) for k, v in sorted(by_qtype.items())},
        "by_answer_length": {k: summarize_group(v) for k, v in by_len.items()},
        "error_breakdown": {
            k: {"count": len(v), "ratio_pct": round(100.0 * len(v) / len(records), 2)}
            for k, v in error_buckets.items()
        },
    }

    # In bảng gọn ra màn hình
    print("\n=== TỔNG QUAN ===")
    print(report["overall"])
    print("\n=== THEO LOẠI CÂU HỎI ===")
    for k, v in report["by_question_type"].items():
        print(f"  {k:28s} | n={v['count']:4d} | EM={v['EM']:6.2f} | F1={v['F1']:6.2f}")
    print("\n=== THEO ĐỘ DÀI ĐÁP ÁN ===")
    for k, v in report["by_answer_length"].items():
        print(f"  {k:28s} | n={v['count']:4d} | EM={v['EM']:6.2f} | F1={v['F1']:6.2f}")
    print("\n=== PHÂN RÃ LỖI HASANS / NOANS ===")
    for k, v in report["error_breakdown"].items():
        print(f"  {k:45s} | n={v['count']:4d} ({v['ratio_pct']:5.2f}%)")

    # Lưu báo cáo JSON và xuất 5 ví dụ tiêu biểu cho mỗi nhóm lỗi để viết báo cáo
    with open(os.path.join(args.out_dir, "error_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    samples_export = {
        k: [
            {
                "id": x["id"],
                "question": x["question"],
                "gold_answers": x["gold_answers"],
                "final_pred": x["final_pred"],
                "raw_span_pred": x["raw_span_pred"],
                "null_score_diff": x["null_score_diff"],
                "f1": x["f1"],
            }
            for x in v[:5]
        ]
        for k, v in error_buckets.items()
    }
    with open(os.path.join(args.out_dir, "error_examples.json"), "w", encoding="utf-8") as f:
        json.dump(samples_export, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] Đã lưu kết quả phân tích lỗi tại {args.out_dir}/")


if __name__ == "__main__":
    main()
import collections
import string
import unicodedata
from typing import Dict, List, Optional, Tuple


# -------------------------------------------------------------------------
# 1. CHUẨN HÓA TIẾNG VIỆT (NFC + Chữ thường + Bỏ dấu câu + Khoảng trắng)
# -------------------------------------------------------------------------
def normalize_answer(text: str) -> str:
    """
    Chuẩn hóa chuỗi văn bản tiếng Việt theo chuẩn đánh giá SQuAD 2.0 / ViMRC:
    1. Chuẩn hóa Unicode NFC.
    2. Đưa về chữ thường (lowercase).
    3. Loại bỏ toàn bộ dấu câu (ASCII punctuation + Unicode punctuation).
    4. Xóa khoảng trắng thừa.
    """
    if not text:
        return ""

    # Bước 1: Chuẩn hóa Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # Bước 2: Chuyển chữ thường
    text = text.lower()

    # Bước 3: Bỏ toàn bộ ký tự thuộc nhóm dấu câu (Punctuation - ký hiệu 'P' trong Unicode)
    # Thay bằng khoảng trắng nhẹ hoặc xóa trực tiếp để tránh dính từ khi có dấu gạch ngang/ngoặc
    cleaned_chars = []
    for ch in text:
        if ch in string.punctuation or unicodedata.category(ch).startswith("P"):
            cleaned_chars.append(" ")
        else:
            cleaned_chars.append(ch)
    text = "".join(cleaned_chars)

    # Bước 4: Tách và ghép lại để xóa mọi khoảng trắng thừa
    return " ".join(text.split())


def get_tokens(text: str) -> List[str]:
    """Tách chuỗi đã chuẩn hóa thành danh sách các âm tiết (tokens)."""
    if not text:
        return []
    return normalize_answer(text).split()


# -------------------------------------------------------------------------
# 2. TÍNH EXACT MATCH (EM) VÀ TOKEN-LEVEL F1 CHO 1 CẶP ĐÁP ÁN
# -------------------------------------------------------------------------
def compute_exact(a_gold: str, a_pred: str) -> float:
    """Trả về 1.0 nếu 2 chuỗi sau chuẩn hóa giống hệt nhau, ngược lại 0.0."""
    return float(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str) -> float:
    """
    Tính điểm F1 ở mức âm tiết (token-level) theo đúng công thức SQuAD 2.0.
    """
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)

    # Xử lý trường hợp câu không có đáp án (rỗng)
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(gold_toks == pred_toks)

    # Đếm số lượng âm tiết xuất hiện ở cả gold và pred
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(
    metric_fn, prediction: str, ground_truths: List[str]
) -> float:
    """
    Nếu 1 câu hỏi có nhiều đáp án chuẩn (ví dụ tập dev/test có 3 người gán nhãn),
    lấy điểm cao nhất giữa dự đoán và từng đáp án chuẩn.
    Nếu danh sách ground_truths rỗng (câu NoAns), so sánh dự đoán với chuỗi rỗng "".
    """
    if not ground_truths:
        return metric_fn("", prediction)

    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(ground_truth, prediction)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


# -------------------------------------------------------------------------
# 3. ĐÁNH GIÁ TOÀN BỘ TẬP TEST (All, HasAns, NoAns) & CHỌN NGƯỠNG NULL
# -------------------------------------------------------------------------
def evaluate_qa(
    samples: List[Dict],
    predictions: Dict[str, str],
    null_score_diff: Optional[Dict[str, float]] = None,
    null_threshold: float = 0.0,
) -> Dict[str, float]:
    """
    Tính điểm EM và F1 trên toàn tập dữ liệu (All), nhóm có đáp án (HasAns),
    và nhóm không có đáp án (NoAns).

    Tham số:
    - samples: Danh sách các mẫu dữ liệu (đọc từ val.jsonl hoặc test.jsonl).
    - predictions: Dict ánh xạ {id_câu_hỏi: chuỗi_dự_đoán_tốt_nhất}.
    - null_score_diff (Tùy chọn): Dict {id_câu_hỏi: score_null - score_best_span}.
      Nếu null_score_diff[qid] > null_threshold thì ép dự đoán về "" (không có đáp án).
    - null_threshold: Ngưỡng quyết định câu không có đáp án (mặc định 0.0).
    """
    exact_scores = {}
    f1_scores = {}
    has_ans_qids = []
    no_ans_qids = []

    for sample in samples:
        qid = str(sample["id"])
        is_impossible = bool(sample.get("is_impossible", False))

        # Lấy danh sách đáp án chuẩn
        raw_ans = sample.get("answers", {})
        if isinstance(raw_ans, dict):
            gold_answers = [t for t in raw_ans.get("text", []) if normalize_answer(t)]
        elif isinstance(raw_ans, list):
            gold_answers = [a["text"] for a in raw_ans if normalize_answer(a.get("text", ""))]
        else:
            gold_answers = []

        if is_impossible or len(gold_answers) == 0:
            no_ans_qids.append(qid)
            gold_answers = []
        else:
            has_ans_qids.append(qid)

        # Lấy câu trả lời dự đoán của mô hình
        pred_text = predictions.get(qid, "")

        # Nếu có truyền điểm null_score_diff và vượt ngưỡng -> Dự đoán là "" (NoAns)
        if null_score_diff is not None and qid in null_score_diff:
            if null_score_diff[qid] > null_threshold:
                pred_text = ""

        exact_scores[qid] = metric_max_over_ground_truths(
            compute_exact, pred_text, gold_answers
        )
        f1_scores[qid] = metric_max_over_ground_truths(
            compute_f1, pred_text, gold_answers
        )

    total = len(exact_scores)
    if total == 0:
        return {"EM": 0.0, "F1": 0.0, "total": 0}

    results = {
        "EM": round(100.0 * sum(exact_scores.values()) / total, 2),
        "F1": round(100.0 * sum(f1_scores.values()) / total, 2),
        "total": total,
    }

    if has_ans_qids:
        n_has = len(has_ans_qids)
        results["HasAns_EM"] = round(
            100.0 * sum(exact_scores[q] for q in has_ans_qids) / n_has, 2
        )
        results["HasAns_F1"] = round(
            100.0 * sum(f1_scores[q] for q in has_ans_qids) / n_has, 2
        )
        results["HasAns_total"] = n_has

    if no_ans_qids:
        n_no = len(no_ans_qids)
        results["NoAns_EM"] = round(
            100.0 * sum(exact_scores[q] for q in no_ans_qids) / n_no, 2
        )
        results["NoAns_F1"] = round(
            100.0 * sum(f1_scores[q] for q in no_ans_qids) / n_no, 2
        )
        results["NoAns_total"] = n_no

    return results


def find_best_null_threshold(
    val_samples: List[Dict],
    predictions: Dict[str, str],
    null_score_diff: Dict[str, float],
) -> Tuple[float, float]:
    if not null_score_diff:
        return 0.0, 0.0

    # Tiền tính toán điểm khi giữ nguyên dự đoán span và khi ép về rỗng ""
    span_f1, null_f1 = {}, {}
    for sample in val_samples:
        qid = str(sample["id"])
        is_impossible = bool(sample.get("is_impossible", False))
        raw_ans = sample.get("answers", {})
        if isinstance(raw_ans, dict):
            gold_answers = [t for t in raw_ans.get("text", []) if normalize_answer(t)]
        elif isinstance(raw_ans, list):
            gold_answers = [a["text"] for a in raw_ans if normalize_answer(a.get("text", ""))]
        else:
            gold_answers = []

        if is_impossible or len(gold_answers) == 0:
            gold_answers = []

        pred_text = predictions.get(qid, "")
        span_f1[qid] = metric_max_over_ground_truths(compute_f1, pred_text, gold_answers)
        null_f1[qid] = 1.0 if not gold_answers else 0.0

    sorted_diffs = sorted(set(null_score_diff.values()))
    if len(sorted_diffs) > 200:
        step = max(1, len(sorted_diffs) // 200)
        sorted_diffs = sorted_diffs[::step]

    candidate_thresholds = sorted(set([sorted_diffs[0] - 1.0, 0.0] + sorted_diffs))

    best_thresh = 0.0
    best_f1 = -1.0
    total = len(val_samples)

    for thresh in candidate_thresholds:
        current_f1_sum = sum(
            null_f1[qid] if null_score_diff.get(qid, -1e9) > thresh else span_f1[qid]
            for qid in span_f1
        )
        f1_score = round(100.0 * current_f1_sum / total, 2)
        if f1_score > best_f1:
            best_f1 = f1_score
            best_thresh = thresh

    return best_thresh, best_f1
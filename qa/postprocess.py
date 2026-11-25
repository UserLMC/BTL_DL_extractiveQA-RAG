import collections
from typing import Dict, List, Tuple
import numpy as np


def postprocess_qa_predictions(
    examples: List[Dict],
    features: List[Dict],
    raw_predictions: Tuple[np.ndarray, np.ndarray],
    n_best_size: int = 20,
    max_answer_length: int = 50,
) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, List[Dict]]]:
    """
    Hậu xử lý start_logits và end_logits thành đáp án chuỗi tốt nhất và null_score_diff.

    Lưu ý quan trọng (hay bị hỏi khi vấn đáp):
    - Một câu hỏi dài có thể tách thành nhiều cửa sổ (features).
    - Điểm null của câu hỏi là GIÁ TRỊ NHỎ NHẤT (min) của (start_logit[CLS] + end_logit[CLS])
      qua mọi cửa sổ: chỉ cần 1 cửa sổ tin rằng có đáp án thì min_null_score sẽ thấp.
    - null_score_diff = min_null_score - best_non_null_score.
      Nếu null_score_diff > null_threshold -> kết luận câu hỏi không có đáp án ("").
    """
    all_start_logits, all_end_logits = raw_predictions
    assert len(features) == len(all_start_logits) == len(all_end_logits), (
        "Số lượng features và logits không khớp nhau!"
    )

    # Gom chỉ số các cửa sổ theo từng example_id
    features_per_example = collections.defaultdict(list)
    for feat_idx, feat in enumerate(features):
        features_per_example[str(feat["example_id"])].append(feat_idx)

    best_non_null_predictions: Dict[str, str] = {}
    null_score_diffs: Dict[str, float] = {}
    nbest_predictions: Dict[str, List[Dict]] = {}

    for example in examples:
        qid = str(example["id"])
        context = example["context"]
        feature_indices = features_per_example.get(qid, [])

        min_null_score = None
        valid_answers = []

        for feat_idx in feature_indices:
            start_logits = all_start_logits[feat_idx]
            end_logits = all_end_logits[feat_idx]
            offsets = features[feat_idx]["offset_mapping"]
            cls_index = features[feat_idx]["cls_index"]

            # Cập nhật min_null_score qua các cửa sổ của cùng câu hỏi
            feature_null_score = float(start_logits[cls_index] + end_logits[cls_index])
            if min_null_score is None or feature_null_score < min_null_score:
                min_null_score = feature_null_score

            # Lấy top n_best_size chỉ số có logit cao nhất
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Bỏ qua vị trí ngoài phạm vi hoặc không thuộc context (offset là (-1, -1))
                    if (
                        start_index >= len(offsets)
                        or end_index >= len(offsets)
                        or offsets[start_index][0] < 0
                        or offsets[end_index][0] < 0
                    ):
                        continue
                    # Bỏ qua span ngược hoặc quá dài
                    if end_index < start_index or (end_index - start_index + 1) > max_answer_length:
                        continue

                    start_char = offsets[start_index][0]
                    end_char = offsets[end_index][1]
                    text = context[start_char:end_char].strip()
                    if not text:
                        continue

                    valid_answers.append(
                        {
                            "score": float(start_logits[start_index] + end_logits[end_index]),
                            "text": text,
                            "start_char": start_char,
                            "end_char": end_char,
                        }
                    )

        # Sắp xếp ứng viên theo score giảm dần và lọc trùng text
        valid_answers.sort(key=lambda x: x["score"], reverse=True)
        unique_answers = []
        seen_texts = set()
        for ans in valid_answers:
            if ans["text"] not in seen_texts:
                seen_texts.add(ans["text"])
                unique_answers.append(ans)
            if len(unique_answers) >= n_best_size:
                break

        if unique_answers:
            best_answer = unique_answers[0]
        else:
            best_answer = {"text": "", "score": 0.0, "start_char": -1, "end_char": -1}

        best_non_null_predictions[qid] = best_answer["text"]
        null_score_diffs[qid] = (
            float(min_null_score - best_answer["score"])
            if min_null_score is not None
            else 0.0
        )
        nbest_predictions[qid] = unique_answers

    return best_non_null_predictions, null_score_diffs, nbest_predictions


def apply_null_threshold(
    predictions: Dict[str, str],
    null_score_diffs: Dict[str, float],
    null_threshold: float,
) -> Dict[str, str]:
    """Ép dự đoán về chuỗi rỗng '' nếu null_score_diff vượt ngưỡng đã chọn trên validation."""
    final_preds = {}
    for qid, pred_text in predictions.items():
        if null_score_diffs.get(qid, -1e9) > null_threshold:
            final_preds[qid] = ""
        else:
            final_preds[qid] = pred_text
    return final_preds
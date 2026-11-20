from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset


def find_context_token_bounds(sequence_ids: List[Optional[int]]) -> Tuple[int, int]:
    """
    Tìm vị trí token đầu tiên và token cuối cùng thuộc về đoạn văn (context, có sequence_id == 1).
    Trong cả XLM-R và mBERT khi truyền tokenizer(question, context):
    - None: Các token đặc biệt (<s>, </s>, [CLS], [SEP], <pad>)
    - 0: Các token thuộc câu hỏi (question)
    - 1: Các token thuộc đoạn văn (context)
    """
    idx = 0
    while idx < len(sequence_ids) and sequence_ids[idx] != 1:
        idx += 1
    context_start = idx

    while idx < len(sequence_ids) and sequence_ids[idx] == 1:
        idx += 1
    context_end = idx - 1

    return context_start, context_end


def map_char_span_to_token_span(
    offsets: List[Tuple[int, int]],
    context_start: int,
    context_end: int,
    ans_start_char: int,
    ans_end_char: int,
    cls_index: int = 0,
) -> Tuple[int, int]:
    """
    Ánh xạ vị trí ký tự (ans_start_char, ans_end_char) sang vị trí token (start_pos, end_pos).
    Nếu đáp án không nằm TRỌN VẸN trong cửa sổ trượt hiện tại (nằm ngoài hoặc bị cắt giữa 2 cửa sổ),
    trả về (cls_index, cls_index).
    """
    if context_start > context_end:
        return cls_index, cls_index

    window_start_char = offsets[context_start][0]
    window_end_char = offsets[context_end][1]

    # Kiểm tra đáp án có nằm trọn vẹn trong cửa sổ hiện tại không
    if not (window_start_char <= ans_start_char and ans_end_char <= window_end_char):
        return cls_index, cls_index

    # Dịch từ trái sang phải để tìm token chứa ký tự bắt đầu của đáp án
    token_start = context_start
    while token_start <= context_end and offsets[token_start][0] <= ans_start_char:
        token_start += 1
    start_pos = token_start - 1

    # Dịch từ phải sang trái để tìm token chứa ký tự kết thúc của đáp án
    token_end = context_end
    while token_end >= context_start and offsets[token_end][1] >= ans_end_char:
        token_end -= 1
    end_pos = token_end + 1

    # Đảm bảo tính hợp lệ start_pos <= end_pos
    if start_pos > end_pos or start_pos < context_start or end_pos > context_end:
        return cls_index, cls_index

    return start_pos, end_pos


def convert_samples_to_features(
    samples: List[Dict],
    tokenizer,
    max_len: int = 384,
    stride: int = 128,
    use_sliding_window: bool = True,
    is_training: bool = True,
) -> List[Dict]:
    """
    Biến đổi danh sách mẫu dữ liệu QA thành danh sách các cửa sổ đặc trưng (features).
    Tham số use_sliding_window cho phép bật/tắt cửa sổ trượt để phục vụ luôn Thí nghiệm Ablation E2 ở Tuần 4.
    """
    features = []

    for sample in samples:
        q_id = str(sample["id"])
        question = sample["question"].strip()
        context = sample["context"]
        is_impossible = bool(sample.get("is_impossible", False))
        answers = sample.get("answers", {"text": [], "answer_start": []})

        tokenized = tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=max_len,
            stride=stride if use_sliding_window else 0,
            return_overflowing_tokens=use_sliding_window,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Nếu tắt sliding window, chuẩn hóa đầu ra thành danh sách gồm 1 cửa sổ
        input_ids_list = tokenized["input_ids"]
        attn_mask_list = tokenized["attention_mask"]
        offsets_list = tokenized["offset_mapping"]
        if not use_sliding_window:
            input_ids_list = [input_ids_list]
            attn_mask_list = [attn_mask_list]
            offsets_list = [offsets_list]

        for win_idx in range(len(input_ids_list)):
            input_ids = input_ids_list[win_idx]
            attention_mask = attn_mask_list[win_idx]
            offsets = offsets_list[win_idx]
            sequence_ids = (
                tokenized.sequence_ids(win_idx)
                if use_sliding_window
                else tokenized.sequence_ids()
            )

            # Tìm vị trí token CLS (<s> trong XLM-R hoặc [CLS] trong mBERT luôn ở vị trí 0)
            cls_index = (
                input_ids.index(tokenizer.cls_token_id)
                if tokenizer.cls_token_id in input_ids
                else 0
            )
            context_start, context_end = find_context_token_bounds(sequence_ids)

            feature = {
                "example_id": q_id,
                "window_index": win_idx,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "cls_index": cls_index,
            }

            if is_training:
                # Chuẩn hóa việc lấy danh sách text và answer_start từ cả dict lẫn list
                if isinstance(answers, dict):
                    ans_texts = answers.get("text", [])
                    ans_starts = answers.get("answer_start", [])
                elif isinstance(answers, list) and len(answers) > 0:
                    ans_texts = [a["text"] for a in answers if "text" in a]
                    ans_starts = [a["answer_start"] for a in answers if "answer_start" in a]
                else:
                    ans_texts, ans_starts = [], []

                # Gán nhãn start_positions và end_positions
                if is_impossible or len(ans_texts) == 0 or not ans_texts[0].strip():
                    start_pos, end_pos = cls_index, cls_index
                else:
                    raw_text = ans_texts[0]
                    ans_start_char = int(ans_starts[0])

                    # Dịch chuyển ans_start_char và ans_end_char nếu đáp án gốc thừa khoảng trắng
                    lstripped_text = raw_text.lstrip()
                    ans_start_char += len(raw_text) - len(lstripped_text)
                    clean_text = lstripped_text.rstrip()
                    ans_end_char = ans_start_char + len(clean_text)

                    start_pos, end_pos = map_char_span_to_token_span(
                        offsets=offsets,
                        context_start=context_start,
                        context_end=context_end,
                        ans_start_char=ans_start_char,
                        ans_end_char=ans_end_char,
                        cls_index=cls_index,
                    )

                feature["start_positions"] = start_pos
                feature["end_positions"] = end_pos

            # Lưu lại offset_mapping của riêng phần context (các phần khác gán (-1, -1))
            # để bước hậu xử lý (ngày 13/11) không bao giờ trích xuất nhầm token của câu hỏi
            clean_offsets = [
                ( int(offsets[i][0]), int(offsets[i][1]) ) if sequence_ids[i] == 1 else (-1, -1)
                for i in range(len(offsets))
            ]
            feature["offset_mapping"] = clean_offsets

            features.append(feature)

    return features


class ViQuADDataset(Dataset):
    """Dataset PyTorch thuần bọc các features để đưa vào DataLoader huấn luyện ngày 12/11."""

    def __init__(self, features: List[Dict], is_training: bool = True):
        self.features = features
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        feat = self.features[idx]
        item = {
            "input_ids": torch.tensor(feat["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(feat["attention_mask"], dtype=torch.long),
        }
        if self.is_training:
            item["start_positions"] = torch.tensor(feat["start_positions"], dtype=torch.long)
            item["end_positions"] = torch.tensor(feat["end_positions"], dtype=torch.long)
        return item
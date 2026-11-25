import json
import os
import random
import unicodedata
from collections import defaultdict

# ---------------------------------------------------------
# HÀM 1: Chuẩn hóa tiếng Việt về Unicode NFC
# ---------------------------------------------------------
def normalize_nfc(text: str) -> str:
    """
    Ép toàn bộ ký tự tiếng Việt về chuẩn Unicode dựng sẵn (NFC).
    Giúp tránh lỗi lệch vị trí ký tự (answer_start) và lỗi so khớp từ khi chấm điểm EM/F1.
    """
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text)


# ---------------------------------------------------------
# HÀM 2: Đọc file JSON từ Kaggle và chuyển về danh sách mẫu phẳng
# ---------------------------------------------------------
def load_and_flatten_viquad(filepath: str) -> list:
    """
    Đọc file JSON gốc (dạng SQuAD 3 tầng hoặc dạng danh sách phẳng)
    và trả về danh sách các câu hỏi đã được chuẩn hóa NFC.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Nếu file có cấu trúc {"version": ..., "data": [...]} chuẩn SQuAD
    records = raw_data["data"] if isinstance(raw_data, dict) and "data" in raw_data else raw_data
    samples = []

    for idx, item in enumerate(records):
        # TRƯỜNG HỢP 1: File JSON chuẩn SQuAD 3 tầng (Article -> Paragraphs -> QAs)
        if "paragraphs" in item:
            title = normalize_nfc(item.get("title", f"article_{idx}")).strip()
            for para in item["paragraphs"]:
                context = normalize_nfc(para["context"])
                for qa in para["qas"]:
                    samples.append(_process_single_qa(title, context, qa))
        
        # TRƯỜNG HỢP 2: File JSON trên Kaggle đã được làm phẳng sẵn từng dòng
        elif "context" in item and "question" in item:
            title = normalize_nfc(item.get("title", f"article_{idx}")).strip()
            context = normalize_nfc(item["context"])
            samples.append(_process_single_qa(title, context, item))

    return samples


def _process_single_qa(title: str, context: str, qa: dict) -> dict:
    """Xử lý từng cặp câu hỏi - câu trả lời và cập nhật lại vị trí answer_start sau khi NFC."""
    q_id = str(qa.get("id", ""))
    question = normalize_nfc(qa.get("question", "")).strip()
    is_impossible = bool(qa.get("is_impossible", False))

    raw_answers = qa.get("answers", [])
    ans_texts = []
    ans_starts = []

    # Xử lý cả 2 kiểu lưu answers: [{"text": ..., "answer_start": ...}] hoặc {"text": [...], "answer_start": [...]}
    if isinstance(raw_answers, list):
        for ans in raw_answers:
            if ans.get("text", "").strip():
                ans_texts.append(normalize_nfc(ans["text"]))
                ans_starts.append(int(ans["answer_start"]))
    elif isinstance(raw_answers, dict):
        ans_texts = [normalize_nfc(t) for t in raw_answers.get("text", []) if str(t).strip()]
        ans_starts = [int(s) for s in raw_answers.get("answer_start", [])]

    # Nếu không có đáp án nào hoặc có cờ is_impossible=True -> Câu hỏi không có đáp án (NoAns)
    if is_impossible or len(ans_texts) == 0:
        return {
            "id": q_id,
            "title": title,
            "context": context,
            "question": question,
            "is_impossible": True,
            "answers": {"text": [], "answer_start": []}
        }

    # Kiểm tra lại vị trí answer_start trên context đã chuẩn hóa NFC
    valid_texts, valid_starts = [], []
    for text, start in zip(ans_texts, ans_starts):
        if context[start : start + len(text)] == text:
            valid_texts.append(text)
            valid_starts.append(start)
        else:
            # Nếu NFC làm co độ dài chuỗi, tìm lại vị trí xuất hiện của câu trả lời trong context
            new_start = context.find(text)
            if new_start != -1:
                valid_texts.append(text)
                valid_starts.append(new_start)

    return {
        "id": q_id,
        "title": title,
        "context": context,
        "question": question,
        "is_impossible": len(valid_texts) == 0,
        "answers": {"text": valid_texts, "answer_start": valid_starts}
    }


# ---------------------------------------------------------
# HÀM 3: Chia 10% tập Train làm Val THEO BÀI VIẾT (Chống rò rỉ ngữ cảnh)
# ---------------------------------------------------------
def split_train_val_by_article(samples: list, val_ratio: float = 0.1, seed: int = 42):
    """
    Gom các câu hỏi có cùng 'title' (bài viết) vào chung một nhóm,
    sau đó bốc ngẫu nhiên 10% số bài viết sang tập Val.
    """
    # Bước 3.1: Gom nhóm câu hỏi theo tên bài viết (title)
    article_to_samples = defaultdict(list)
    for s in samples:
        article_to_samples[s["title"]].append(s)

    # Bước 3.2: Xáo trộn danh sách bài viết với seed cố định = 42
    all_titles = list(article_to_samples.keys())
    rng = random.Random(seed)
    rng.shuffle(all_titles)

    # Bước 3.3: Lấy 10% bài viết đầu tiên làm tập Val, 90% còn lại làm tập Train
    n_val_articles = max(1, int(round(len(all_titles) * val_ratio)))
    val_titles_set = set(all_titles[:n_val_articles])

    train_samples, val_samples = [], []
    for title, group in article_to_samples.items():
        if title in val_titles_set:
            val_samples.extend(group)
        else:
            train_samples.extend(group)

    return train_samples, val_samples


def save_jsonl(samples: list, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    TRAIN_RAW = "data/raw/train.json"
    DEV_RAW   = "data/raw/dev.json"

    if not os.path.exists(TRAIN_RAW) or not os.path.exists(DEV_RAW):
        print("LỖI: Bạn hãy bỏ 2 file train.json và dev.json tải từ Kaggle vào thư mục data/raw/ trước nhé!")
    else:
        print("1. Đang đọc và chuẩn hóa NFC dữ liệu gốc...")
        full_train = load_and_flatten_viquad(TRAIN_RAW)
        test_samples = load_and_flatten_viquad(DEV_RAW)  # Dev chính thức làm Test cố định

        print("2. Đang tách 10% tập train làm val theo bài viết (seed=42)...")
        train_samples, val_samples = split_train_val_by_article(full_train, val_ratio=0.1, seed=42)

        # Kiểm tra an toàn: Đảm bảo không có bài viết nào xuất hiện ở cả Train và Val
        train_titles = {s["title"] for s in train_samples}
        val_titles   = {s["title"] for s in val_samples}
        assert len(train_titles.intersection(val_titles)) == 0, "Có rò rỉ bài viết giữa Train và Val!"

        save_jsonl(train_samples, "data/processed/train.jsonl")
        save_jsonl(val_samples,   "data/processed/val.jsonl")
        save_jsonl(test_samples,  "data/processed/test.jsonl")

        print("\n=== ĐÃ CHIA XONG DỮ LIỆU ===")
        print(f"- Train: {len(train_titles)} bài viết | {len(train_samples)} câu hỏi -> data/processed/train.jsonl")
        print(f"- Val:   {len(val_titles)} bài viết | {len(val_samples)} câu hỏi -> data/processed/val.jsonl")
        print(f"- Test:  {len({s['title'] for s in test_samples})} bài viết | {len(test_samples)} câu hỏi -> data/processed/test.jsonl")
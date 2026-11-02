"""
Tokenize & map token offset -> start/end position cho Extractive QA (UIT-ViQuAD 2.0).

Hỗ trợ 2 loại tokenizer:
  * Fast tokenizer (XLM-R, mBERT, ...): dùng ``offset_mapping`` có sẵn của HF.
  * Slow tokenizer (PhoBERT): PhoBERT không có fast tokenizer nên KHÔNG trả về
    offset_mapping. Script tự tính offset ký tự cho từng sub-token (BPE ``@@``)
    và (tuỳ chọn) căn chỉnh lại văn bản đã tách từ bằng VnCoreNLP
    (``Phạm_Văn_Đồng``) về toạ độ ký tự của context GỐC.

Mỗi example (question, context) được cắt thành nhiều feature bằng sliding window
(``max_length`` / ``doc_stride``). Với mỗi feature:
  * ``start_positions`` / ``end_positions``: chỉ số token (inclusive) của đáp án.
    Nếu câu hỏi không có đáp án (is_impossible) hoặc đáp án không nằm trọn trong
    window -> trỏ vào token CLS (index 0), đúng chuẩn SQuAD 2.0.
  * ``offset_mapping`` (chỉ cho tập đánh giá): (char_start, char_end) trong context
    gốc cho token thuộc context, ``None`` cho token câu hỏi / special / padding
    -> dùng ở bước hậu xử lý để cắt câu trả lời từ context.
  * ``example_id``: id của example gốc (1 example có thể sinh nhiều feature).

Ví dụ:
    # XLM-R
    python extractive_qa/preprocess_qa.py --model xlm-roberta-base \
        --max_length 384 --doc_stride 128 --output_dir data/processed/xlmr-base --verify

    # PhoBERT (+ tách từ VnCoreNLP, cần Java)
    python extractive_qa/preprocess_qa.py --model vinai/phobert-base-v2 \
        --max_length 256 --doc_stride 64 --segment --vncorenlp_dir ./vncorenlp \
        --output_dir data/processed/phobert-base-v2 --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger("preprocess_qa")

DATASET_NAME = "taidng/UIT-ViQuAD2.0"

Offset = Tuple[int, int]
# Hàm tách từ: nhận chuỗi, trả về list các "từ" (âm tiết nối bằng '_')
Segmenter = Callable[[str], List[str]]


# =============================================================================
# 1. Chuẩn hoá đáp án (char span trong context)
# =============================================================================
def fix_answer_start(context: str, answer_text: str, answer_start: int, window: int = 20) -> int:
    """Sửa ``answer_start`` nếu bị lệch so với context (lỗi gán nhãn thường gặp).

    Thứ tự: đúng vị trí -> tìm lân cận ±window -> lần xuất hiện gần nhất trong context.
    """
    if context[answer_start: answer_start + len(answer_text)] == answer_text:
        return answer_start
    lo = max(0, answer_start - window)
    hi = min(len(context), answer_start + len(answer_text) + window)
    idx = context.find(answer_text, lo, hi)
    if idx != -1:
        return idx
    candidates = [m.start() for m in re.finditer(re.escape(answer_text), context)]
    if candidates:
        return min(candidates, key=lambda c: abs(c - answer_start))
    return answer_start  # không tìm thấy -> giữ nguyên


def get_answer_char_span(
    context: str, answers: Dict[str, Sequence], is_impossible: bool = False
) -> Optional[Offset]:
    """Trả về (start_char, end_char) [end exclusive] của đáp án đầu tiên,
    hoặc ``None`` nếu câu hỏi không có đáp án."""
    if is_impossible or answers is None or len(answers.get("text", [])) == 0:
        return None
    text = answers["text"][0]
    start = int(answers["answer_start"][0])
    # Bỏ khoảng trắng thừa ở 2 đầu đáp án, dịch start tương ứng
    stripped = text.strip()
    if not stripped:
        return None
    start += len(text) - len(text.lstrip())
    start = fix_answer_start(context, stripped, start)
    return start, start + len(stripped)


# =============================================================================
# 2. Char span -> token span (dùng chung cho mọi tokenizer)
# =============================================================================
def char_span_to_token_span(
    offsets: Sequence[Offset],
    context_mask: Sequence[bool],
    start_char: int,
    end_char: int,
    cls_index: int = 0,
) -> Tuple[int, int]:
    """Ánh xạ đoạn ký tự [start_char, end_char) sang (start_token, end_token) inclusive.

    ``offsets[i]`` là (char_start, char_end) của token i trong context gốc;
    ``context_mask[i]`` = True nếu token i thuộc context. Nếu đáp án không nằm
    trọn trong phần context của window -> trả về (cls_index, cls_index).
    """
    ctx_idx = [i for i, m in enumerate(context_mask) if m]
    if not ctx_idx:
        return cls_index, cls_index
    first, last = ctx_idx[0], ctx_idx[-1]

    if offsets[first][0] > start_char or offsets[last][1] < end_char:
        return cls_index, cls_index

    # Token đầu: token cuối cùng có char_start <= start_char
    tok_start = first
    while tok_start <= last and offsets[tok_start][0] <= start_char:
        tok_start += 1
    tok_start -= 1

    # Token cuối: token đầu tiên (tính từ phải) có char_end >= end_char
    tok_end = last
    while tok_end >= first and offsets[tok_end][1] >= end_char:
        tok_end -= 1
    tok_end += 1

    if tok_start > tok_end:  # trường hợp suy biến (offset rỗng, ...)
        return cls_index, cls_index
    return tok_start, tok_end


# =============================================================================
# 3. Offset thủ công cho slow tokenizer (PhoBERT)
# =============================================================================
def align_segmented_words(original: str, seg_words: Sequence[str]) -> List[Tuple[str, List[int]]]:
    """Căn chỉnh các từ đã tách (vd. ``Phạm_Văn_Đồng``, ``(``) về văn bản gốc.

    Trả về list ``(word, charmap)`` với ``charmap[k]`` = vị trí trong ``original``
    của ký tự ``word[k]`` (``-1`` nếu là '_' nối âm tiết hoặc không căn được).
    Bộ tách từ có thể thêm/bớt khoảng trắng (vd. tách dấu câu) nên ta so khớp
    theo ký tự, bỏ qua khoảng trắng.
    """
    out: List[Tuple[str, List[int]]] = []
    pos, n = 0, len(original)
    for word in seg_words:
        cmap: List[int] = []
        for ch in word:
            if ch == "_" and not (pos < n and original[pos] == "_"):
                cmap.append(-1)  # '_' do bộ tách từ thêm vào
                continue
            j = pos
            while j < n and original[j].isspace():
                j += 1
            if j < n and original[j] == ch:
                cmap.append(j)
                pos = j + 1
            else:
                k = original.find(ch, pos)
                if k == -1:
                    cmap.append(-1)
                else:
                    cmap.append(k)
                    pos = k + 1
        out.append((word, cmap))
    return out


class SlowTokenizerAligner:
    """Encode văn bản + tính offset ký tự (theo văn bản gốc) cho slow tokenizer BPE."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, segmenter: Optional[Segmenter] = None,
                 bpe_suffix: str = "@@"):
        self.tok = tokenizer
        self.segmenter = segmenter
        self.bpe_suffix = bpe_suffix
        self._word_cache: Dict[str, List[Tuple[int, int, int]]] = {}
        self._text_cache: Dict[str, Tuple[List[int], List[Offset]]] = {}

    # Không tách từ: vẫn tách dấu câu khỏi chữ (giống output VnCoreNLP mà PhoBERT
    # được pre-train), nhưng giữ số dạng 1.000 / 3,5 / 29-4 thành một khối.
    _WORD_RE = re.compile(r"\w+(?:[.,\-/:]\w+)*|[^\w\s]")

    def _words(self, text: str) -> List[Tuple[str, List[int]]]:
        if self.segmenter is None:
            return [(m.group(), list(range(m.start(), m.end()))) for m in self._WORD_RE.finditer(text)]
        return align_segmented_words(text, self.segmenter(text))

    def _tokenize_word(self, word: str) -> List[Tuple[int, int, int]]:
        """Trả về list (token_id, start_in_word, end_in_word)."""
        if word in self._word_cache:
            return self._word_cache[word]
        pieces = self.tok.tokenize(word)
        ids = self.tok.convert_tokens_to_ids(pieces)
        sfx = self.bpe_suffix
        clean = [p[: -len(sfx)] if p.endswith(sfx) else p for p in pieces]
        if "".join(clean) == word:
            res, p = [], 0
            for tid, c in zip(ids, clean):
                res.append((tid, p, p + len(c)))
                p += len(c)
        else:
            # <unk> hoặc tokenizer chuẩn hoá ký tự -> gán cả từ cho mọi sub-token
            res = [(tid, 0, len(word)) for tid in ids]
        self._word_cache[word] = res
        return res

    def encode(self, text: str) -> Tuple[List[int], List[Offset]]:
        if text in self._text_cache:
            return self._text_cache[text]
        ids: List[int] = []
        offs: List[Offset] = []
        prev_end = 0
        for word, cmap in self._words(text):
            for tid, s, e in self._tokenize_word(word):
                valid = [cmap[k] for k in range(s, e) if cmap[k] >= 0]
                if not valid:
                    valid = [c for c in cmap if c >= 0]
                off = (valid[0], valid[-1] + 1) if valid else (prev_end, prev_end)
                ids.append(tid)
                offs.append(off)
                prev_end = off[1]
        self._text_cache[text] = (ids, offs)
        return ids, offs


def build_vncorenlp_segmenter(save_dir: str) -> Segmenter:
    """Tạo hàm tách từ bằng VnCoreNLP RDRSegmenter (``pip install py_vncorenlp``, cần Java)."""
    import py_vncorenlp  # import trễ: chỉ cần khi dùng PhoBERT + tách từ

    save_dir = os.path.abspath(save_dir)
    if not os.path.exists(os.path.join(save_dir, "VnCoreNLP-1.2.jar")):
        os.makedirs(save_dir, exist_ok=True)
        py_vncorenlp.download_model(save_dir=save_dir)
    cwd = os.getcwd()  # py_vncorenlp đổi working dir khi khởi tạo
    model = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)
    os.chdir(cwd)

    def segment(text: str) -> List[str]:
        return " ".join(model.word_segment(text)).split()

    return segment


# =============================================================================
# 4. Preprocessor chính
# =============================================================================
class ViQuADPreprocessor:
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        max_length: Optional[int] = None,
        doc_stride: int = 128,
        max_query_length: int = 64,
        pad_to_max_length: bool = True,
        segmenter: Optional[Segmenter] = None,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.is_fast = bool(self.tokenizer.is_fast)

        if max_length is None:
            max_length = 256 if "phobert" in model_name.lower() else 384
        model_max = self.tokenizer.model_max_length
        if model_max and model_max < 100_000 and max_length > model_max:
            logger.warning("max_length=%d > model_max_length=%d -> giảm xuống.", max_length, model_max)
            max_length = model_max
        if doc_stride >= max_length // 2:
            raise ValueError(f"doc_stride ({doc_stride}) nên nhỏ hơn max_length/2 ({max_length // 2}).")

        self.max_length = max_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.pad_to_max_length = pad_to_max_length
        self.segmenter = segmenter

        if self.tokenizer.padding_side != "right":
            raise ValueError("Script giả định padding_side='right' (question trước, context sau).")

        self.aligner: Optional[SlowTokenizerAligner] = None
        if not self.is_fast:
            if self.tokenizer.num_special_tokens_to_add(pair=True) != 4:
                raise ValueError("Slow path hiện chỉ hỗ trợ định dạng RoBERTa: <s> Q </s></s> C </s>.")
            self.aligner = SlowTokenizerAligner(self.tokenizer, segmenter)
            if segmenter is None and "phobert" in model_name.lower():
                logger.warning("PhoBERT được pre-train trên văn bản ĐÃ TÁCH TỪ; nên bật --segment "
                               "để đạt kết quả tốt hơn.")
        elif segmenter is not None:
            logger.warning("Fast tokenizer không dùng segmenter -> bỏ qua --segment.")

        logger.info("Tokenizer=%s | fast=%s | max_length=%d | doc_stride=%d",
                    type(self.tokenizer).__name__, self.is_fast, self.max_length, self.doc_stride)

    # ------------------------------------------------------------------ API
    def prepare_train_features(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Feature cho huấn luyện: input_ids, attention_mask, start/end_positions, example_id."""
        return self.prepare_features(examples, with_labels=True, keep_offsets=False)

    def prepare_eval_features(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Feature cho đánh giá: thêm offset_mapping (None ngoài context) để hậu xử lý.
        Có kèm nhãn nếu dữ liệu có ``answers`` (để tính eval loss)."""
        return self.prepare_features(examples, with_labels="answers" in examples, keep_offsets=True)

    def prepare_features(self, examples: Dict[str, List[Any]], with_labels: bool = True,
                         keep_offsets: bool = False) -> Dict[str, List[Any]]:
        if self.is_fast:
            return self._prepare_fast(examples, with_labels, keep_offsets)
        return self._prepare_slow(examples, with_labels, keep_offsets)

    def process_dataset(self, raw: DatasetDict, num_proc: Optional[int] = None) -> DatasetDict:
        if self.segmenter is not None:
            num_proc = None  # đối tượng Java của VnCoreNLP không pickle được
        out = {}
        for split, ds in raw.items():
            fn = self.prepare_train_features if split == "train" else self.prepare_eval_features
            out[split] = ds.map(fn, batched=True, remove_columns=ds.column_names,
                                num_proc=num_proc, desc=f"Tokenize {split}")
            logger.info("%s: %d examples -> %d features", split, len(ds), len(out[split]))
        return DatasetDict(out)

    # ------------------------------------------------------------ helpers
    def _label(self, examples, ex_idx, offsets, ctx_mask, cls_index) -> Tuple[int, int]:
        impossible = bool(examples["is_impossible"][ex_idx]) if "is_impossible" in examples else False
        span = get_answer_char_span(examples["context"][ex_idx], examples["answers"][ex_idx], impossible)
        if span is None:
            return cls_index, cls_index
        return char_span_to_token_span(offsets, ctx_mask, span[0], span[1], cls_index)

    def _cls_index(self, input_ids: List[int]) -> int:
        cls_id = self.tokenizer.cls_token_id
        return input_ids.index(cls_id) if cls_id in input_ids else 0

    # ---------------------------------------------------------- fast path
    def _prepare_fast(self, examples, with_labels, keep_offsets):
        tok = self.tokenizer
        questions = [q.strip() for q in examples["question"]]
        # Cắt câu hỏi quá dài để không chiếm hết chỗ của context
        questions = [
            tok.convert_tokens_to_string(tok.tokenize(q)[: self.max_query_length])
            if len(q) > self.max_query_length * 2 else q
            for q in questions
        ]
        enc = tok(
            questions,
            examples["context"],
            truncation="only_second",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.pad_to_max_length else False,
        )
        sample_map = enc.pop("overflow_to_sample_mapping")
        offset_mapping = enc.pop("offset_mapping")

        result: Dict[str, List[Any]] = {k: v for k, v in enc.items()}
        result["example_id"] = []
        if with_labels:
            result["start_positions"], result["end_positions"] = [], []
        if keep_offsets:
            result["offset_mapping"] = []

        for i, offsets in enumerate(offset_mapping):
            ex_idx = sample_map[i]
            ctx_mask = [sid == 1 for sid in enc.sequence_ids(i)]
            result["example_id"].append(examples["id"][ex_idx])
            if with_labels:
                s, e = self._label(examples, ex_idx, offsets, ctx_mask, self._cls_index(enc["input_ids"][i]))
                result["start_positions"].append(s)
                result["end_positions"].append(e)
            if keep_offsets:
                result["offset_mapping"].append(
                    [list(o) if m else None for o, m in zip(offsets, ctx_mask)])
        return result

    # ---------------------------------------------------------- slow path
    def _prepare_slow(self, examples, with_labels, keep_offsets):
        tok, aligner = self.tokenizer, self.aligner
        cls_id, sep_id, pad_id = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id

        result: Dict[str, List[Any]] = {"input_ids": [], "attention_mask": [], "example_id": []}
        if with_labels:
            result["start_positions"], result["end_positions"] = [], []
        if keep_offsets:
            result["offset_mapping"] = []

        for ex_idx in range(len(examples["id"])):
            q_ids, _ = aligner.encode(examples["question"][ex_idx].strip())
            q_ids = q_ids[: self.max_query_length]
            c_ids, c_offs = aligner.encode(examples["context"][ex_idx])

            # <s> Q </s> </s> C </s>
            max_ctx = self.max_length - len(q_ids) - 4
            step = max_ctx - self.doc_stride  # doc_stride = số token chồng lấn giữa 2 window
            if step <= 0:
                raise ValueError("max_length quá nhỏ so với câu hỏi + doc_stride.")
            ctx_begin = 1 + len(q_ids) + 2

            start = 0
            while True:
                end = min(start + max_ctx, len(c_ids))
                chunk_ids, chunk_offs = c_ids[start:end], c_offs[start:end]

                input_ids = [cls_id] + q_ids + [sep_id, sep_id] + chunk_ids + [sep_id]
                n = len(input_ids)
                ctx_mask = [ctx_begin <= i < ctx_begin + len(chunk_ids) for i in range(n)]
                offsets: List[Optional[Offset]] = [None] * n
                offsets[ctx_begin: ctx_begin + len(chunk_ids)] = chunk_offs
                attn = [1] * n

                if self.pad_to_max_length and n < self.max_length:
                    pad = self.max_length - n
                    input_ids += [pad_id] * pad
                    attn += [0] * pad
                    ctx_mask += [False] * pad
                    offsets += [None] * pad

                result["input_ids"].append(input_ids)
                result["attention_mask"].append(attn)
                result["example_id"].append(examples["id"][ex_idx])
                if with_labels:
                    full_offs = [o if o is not None else (0, 0) for o in offsets]
                    s, e = self._label(examples, ex_idx, full_offs, ctx_mask, cls_index=0)
                    result["start_positions"].append(s)
                    result["end_positions"].append(e)
                if keep_offsets:
                    result["offset_mapping"].append([list(o) if o is not None else None for o in offsets])

                if end >= len(c_ids):
                    break
                start += step
        return result


# =============================================================================
# 5. Kiểm tra căn chỉnh: giải mã lại span từ token -> so với đáp án gốc
# =============================================================================
def verify_alignment(pre: ViQuADPreprocessor, raw: Dataset, show: int = 5) -> Dict[str, Any]:
    feats = raw.map(lambda b: pre.prepare_features(b, with_labels=True, keep_offsets=True),
                    batched=True, remove_columns=raw.column_names, desc="Verify")
    id2ex = {ex["id"]: ex for ex in raw}

    stats = Counter()
    found_answer = set()
    mismatches = []
    for f in feats:
        ex = id2ex[f["example_id"]]
        s, e = f["start_positions"], f["end_positions"]
        stats["features"] += 1
        if s == 0 and e == 0:
            stats["features_cls"] += 1
            continue
        span = get_answer_char_span(ex["context"], ex["answers"], ex.get("is_impossible", False))
        gold = ex["context"][span[0]: span[1]]
        offs = f["offset_mapping"]
        pred = ex["context"][offs[s][0]: offs[e][1]].strip()
        found_answer.add(ex["id"])
        if pred == gold:
            stats["exact"] += 1
        elif gold in pred:
            stats["superset"] += 1  # token chứa cả phần ngoài đáp án (vd. đáp án nằm giữa 1 từ ghép)
            if len(mismatches) < show:
                mismatches.append((gold, pred))
        else:
            stats["wrong"] += 1
            if len(mismatches) < show:
                mismatches.append((gold, pred))

    answerable = [ex for ex in raw if get_answer_char_span(
        ex["context"], ex["answers"], ex.get("is_impossible", False)) is not None]
    stats["examples"] = len(raw)
    stats["examples_answerable"] = len(answerable)
    stats["answerable_lost_by_truncation"] = sum(1 for ex in answerable if ex["id"] not in found_answer)

    labeled = stats["exact"] + stats["superset"] + stats["wrong"]
    logger.info("===== VERIFY =====")
    for k in ["examples", "examples_answerable", "features", "features_cls",
              "exact", "superset", "wrong", "answerable_lost_by_truncation"]:
        logger.info("  %-32s %d", k, stats[k])
    if labeled:
        logger.info("  exact-match rate (features có đáp án): %.2f%%", 100 * stats["exact"] / labeled)
    for gold, pred in mismatches:
        logger.info("  [mismatch] gold=%r | decoded=%r", gold, pred)
    return dict(stats)


# =============================================================================
# 6. CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tokenize UIT-ViQuAD 2.0 cho extractive QA.")
    p.add_argument("--model", default="xlm-roberta-base",
                   help="vd. xlm-roberta-base, xlm-roberta-large, vinai/phobert-base-v2")
    p.add_argument("--dataset", default=DATASET_NAME)
    p.add_argument("--splits", nargs="+", default=None, help="Mặc định: tất cả split")
    p.add_argument("--max_length", type=int, default=None, help="Mặc định 384 (XLM-R) / 256 (PhoBERT)")
    p.add_argument("--doc_stride", type=int, default=128)
    p.add_argument("--max_query_length", type=int, default=64)
    p.add_argument("--no_pad", action="store_true", help="Không pad (dùng DataCollatorWithPadding)")
    p.add_argument("--segment", action="store_true", help="Tách từ bằng VnCoreNLP (cho PhoBERT)")
    p.add_argument("--vncorenlp_dir", default="./vncorenlp")
    p.add_argument("--sample", type=int, default=None, help="Chỉ lấy N example mỗi split (debug)")
    p.add_argument("--num_proc", type=int, default=None)
    p.add_argument("--output_dir", default=None, help="Thư mục lưu DatasetDict đã tokenize")
    p.add_argument("--verify", action="store_true", help="Kiểm tra căn chỉnh trên tập train")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    raw = load_dataset(args.dataset)
    if args.splits:
        raw = DatasetDict({s: raw[s] for s in args.splits})
    if args.sample:
        raw = DatasetDict({s: ds.select(range(min(args.sample, len(ds)))) for s, ds in raw.items()})
    logger.info("Dataset: %s", raw)

    segmenter = build_vncorenlp_segmenter(args.vncorenlp_dir) if args.segment else None
    pre = ViQuADPreprocessor(
        model_name=args.model,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        pad_to_max_length=not args.no_pad,
        segmenter=segmenter,
    )

    if args.verify and "train" in raw:
        verify_alignment(pre, raw["train"])

    if args.output_dir:
        processed = pre.process_dataset(raw, num_proc=args.num_proc)
        processed.save_to_disk(args.output_dir)
        meta = {
            "model": args.model, "dataset": args.dataset, "max_length": pre.max_length,
            "doc_stride": pre.doc_stride, "max_query_length": pre.max_query_length,
            "segmented": segmenter is not None, "fast_tokenizer": pre.is_fast,
            "num_features": {s: len(d) for s, d in processed.items()},
        }
        with open(os.path.join(args.output_dir, "preprocess_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info("Đã lưu vào %s", args.output_dir)


if __name__ == "__main__":
    main()

"""Unit test cho extractive_qa/preprocess_qa.py.

Chạy:  python -m pytest tests/test_preprocess_qa.py -v
(Các test có tokenizer sẽ tải model từ HF Hub lần đầu.)
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "extractive_qa"))
from preprocess_qa import (  # noqa: E402
    ViQuADPreprocessor,
    align_segmented_words,
    char_span_to_token_span,
    fix_answer_start,
    get_answer_char_span,
)

CONTEXT = (
    "Phạm Văn Đồng (1 tháng 3 năm 1906 – 29 tháng 4 năm 2000) là Thủ tướng đầu tiên của nước "
    "Cộng hòa Xã hội chủ nghĩa Việt Nam từ năm 1976. Ông còn có tên gọi là Lâm Bá Kiệt khi làm "
    "Phó chủ nhiệm cơ quan Biện sự xứ tại Quế Lâm (Chủ nhiệm là Hồ Học Lãm)."
)


def make_batch(long_context=False):
    ctx = CONTEXT
    if long_context:  # đẩy đáp án ra xa để buộc sinh nhiều window
        ctx = ("Đây là một câu văn đệm dài để kiểm tra cửa sổ trượt. " * 40) + CONTEXT
    ans = "Lâm Bá Kiệt"
    return {
        "id": ["q1", "q2"],
        "question": ["Tên gọi nào được Phạm Văn Đồng sử dụng ở Quế Lâm?", "Ai là vua đầu tiên?"],
        "context": [ctx, ctx],
        "answers": [{"text": [ans], "answer_start": [ctx.index(ans)]}, {"text": [], "answer_start": []}],
        "is_impossible": [False, True],
    }


# ----------------------------------------------------------------- pure tests
def test_fix_answer_start():
    assert fix_answer_start(CONTEXT, "Lâm Bá Kiệt", CONTEXT.index("Lâm Bá Kiệt") + 3) == CONTEXT.index("Lâm Bá Kiệt")


def test_get_answer_char_span_impossible():
    assert get_answer_char_span(CONTEXT, {"text": [], "answer_start": []}) is None
    assert get_answer_char_span(CONTEXT, {"text": ["x"], "answer_start": [0]}, is_impossible=True) is None


def test_char_span_to_token_span():
    #            CLS   q      SEP    "ab"    "cd"     "ef"
    offsets = [(0, 0), (0, 0), (0, 0), (0, 2), (3, 5), (6, 8)]
    mask = [False, False, False, True, True, True]
    assert char_span_to_token_span(offsets, mask, 3, 5) == (4, 4)
    assert char_span_to_token_span(offsets, mask, 0, 8) == (3, 5)
    assert char_span_to_token_span(offsets, mask, 6, 20) == (0, 0)  # vượt ra ngoài window


def test_align_segmented_words():
    original = "Phạm Văn Đồng (1906)"
    seg = ["Phạm_Văn_Đồng", "(", "1906", ")"]
    out = align_segmented_words(original, seg)
    word, cmap = out[0]
    assert word == "Phạm_Văn_Đồng"
    assert original[cmap[0]: max(cmap) + 1] == "Phạm Văn Đồng"
    assert cmap[4] == -1  # '_' do bộ tách từ thêm vào
    assert original[out[1][1][0]] == "("


# ------------------------------------------------------------ tokenizer tests
def _check(pre, long_context):
    batch = make_batch(long_context)
    feats = pre.prepare_features(batch, with_labels=True, keep_offsets=True)
    n = len(feats["input_ids"])
    assert all(len(x) == pre.max_length for x in feats["input_ids"])
    if long_context:
        assert n > 2, "Context dài phải sinh nhiều feature"

    found = False
    for i in range(n):
        s, e = feats["start_positions"][i], feats["end_positions"][i]
        if feats["example_id"][i] == "q2":
            assert (s, e) == (0, 0)  # câu không có đáp án -> CLS
        elif (s, e) != (0, 0):
            offs = feats["offset_mapping"][i]
            ctx = batch["context"][0]
            assert ctx[offs[s][0]: offs[e][1]].strip() == "Lâm Bá Kiệt"
            found = True
    assert found, "Ít nhất 1 window phải chứa đáp án"


@pytest.mark.parametrize("long_context", [False, True])
def test_xlmr(long_context):
    pre = ViQuADPreprocessor("xlm-roberta-base", max_length=128, doc_stride=32)
    assert pre.is_fast
    _check(pre, long_context)


@pytest.mark.parametrize("long_context", [False, True])
def test_phobert_no_segment(long_context):
    pre = ViQuADPreprocessor("vinai/phobert-base-v2", max_length=128, doc_stride=32)
    assert not pre.is_fast
    _check(pre, long_context)


def test_phobert_fake_segmenter():
    """Giả lập bộ tách từ: nối 'Lâm Bá Kiệt' & 'Phạm Văn Đồng', tách dấu ngoặc."""
    def fake_seg(text):
        t = text.replace("Lâm Bá Kiệt", "Lâm_Bá_Kiệt").replace("Phạm Văn Đồng", "Phạm_Văn_Đồng")
        t = t.replace("(", " ( ").replace(")", " ) ")
        return t.split()

    pre = ViQuADPreprocessor("vinai/phobert-base-v2", max_length=128, doc_stride=32, segmenter=fake_seg)
    _check(pre, long_context=True)

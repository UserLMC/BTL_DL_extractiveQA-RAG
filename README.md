# BTL_DL_extractiveQA-RAG

Hỏi đáp tiếng Việt: Extractive QA (UIT-ViQuAD 2.0) và RAG.

## Cài đặt
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Extractive QA – Tiền xử lý (tokenize + map offset → start/end position)

Script: `extractive_qa/preprocess_qa.py`

```bash
# XLM-R (fast tokenizer, dùng offset_mapping có sẵn)
python extractive_qa/preprocess_qa.py --model xlm-roberta-base \
    --max_length 384 --doc_stride 128 --output_dir data/processed/xlmr-base --verify

# PhoBERT (slow tokenizer, tự tính offset; --segment để tách từ bằng VnCoreNLP, cần Java)
python extractive_qa/preprocess_qa.py --model vinai/phobert-base-v2 \
    --max_length 256 --doc_stride 64 --output_dir data/processed/phobert-base-v2 --verify
```

Dùng trong notebook / script train:
```python
from datasets import load_dataset, load_from_disk
from extractive_qa.preprocess_qa import ViQuADPreprocessor

pre = ViQuADPreprocessor("xlm-roberta-base", max_length=384, doc_stride=128)
raw = load_dataset("taidng/UIT-ViQuAD2.0")
train_feats = raw["train"].map(pre.prepare_train_features, batched=True,
                               remove_columns=raw["train"].column_names)
val_feats = raw["validation"].map(pre.prepare_eval_features, batched=True,
                                  remove_columns=raw["validation"].column_names)
# hoặc: processed = load_from_disk("data/processed/xlmr-base")
```

Kiểm tra: `python -m pytest tests -v`
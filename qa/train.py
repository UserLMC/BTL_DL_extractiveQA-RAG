import argparse
import inspect
import json
import os
import random
import sys
from typing import Dict, List

import numpy as np
import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# Thêm thư mục gốc repo vào sys.path để tránh lỗi ModuleNotFoundError
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from qa.preprocess import ViQuADDataset, convert_samples_to_features


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str, ratio: float = 1.0, seed: int = 42) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    if ratio < 1.0:
        rng = random.Random(seed)
        indices = list(range(len(data)))
        rng.shuffle(indices)
        n_keep = max(20, int(len(data) * ratio))
        selected_indices = sorted(indices[:n_keep])
        data = [data[i] for i in selected_indices]

    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune XLM-R cho bài toán Extractive QA trên UIT-ViQuAD 2.0"
    )
    # Đường dẫn dữ liệu & mô hình
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
    parser.add_argument("--train_file", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/processed/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/xlmr_w1")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Đường dẫn tới thư mục checkpoint-* để train tiếp nếu bị ngắt kết nối",
    )

    # Tham số tiền xử lý
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--no_sliding_window", action="store_true")
    parser.add_argument(
        "--data_ratio",
        type=float,
        default=1.0,
        help="Tỉ lệ dữ liệu sử dụng (0.1 để debug nhanh, 1.0 để train toàn bộ)",
    )

    # Siêu tham số huấn luyện (khớp kế hoạch: lr=3e-5, batch hiệu dụng=32, 3 epoch, fp16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_fp16", action="store_true")

    # Logging & W&B
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--wandb_project", type=str, default="qa-rag-vi")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Cấu hình Weights & Biases
    report_to = "none" if args.no_wandb else "wandb"
    if not args.no_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity

    run_name = args.run_name or (
        f"{args.model_name.split('/')[-1]}-r{args.data_ratio}"
        f"-lr{args.lr}-len{args.max_len}-ep{args.epochs}"
    )

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[INFO] Thiết bị huấn luyện: {device_name}")
    print(f"[INFO] Mô hình: {args.model_name} | Run: {run_name}")
    print(
        f"[INFO] Batch hiệu dụng = {args.train_batch_size} x {args.grad_accum} = "
        f"{args.train_batch_size * args.grad_accum}"
    )

    # 1. Nạp tokenizer và dữ liệu
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_samples = load_jsonl(args.train_file, ratio=args.data_ratio, seed=args.seed)
    val_samples = load_jsonl(args.val_file, ratio=args.data_ratio, seed=args.seed)
    print(f"[INFO] Số mẫu gốc -> Train: {len(train_samples)} | Val: {len(val_samples)}")

    # 2. Chuyển đổi sang cửa sổ đặc trưng (features) bằng preprocess.py tự cài
    use_sliding_window = not args.no_sliding_window
    train_features = convert_samples_to_features(
        samples=train_samples,
        tokenizer=tokenizer,
        max_len=args.max_len,
        stride=args.stride,
        use_sliding_window=use_sliding_window,
        is_training=True,
    )
    val_features = convert_samples_to_features(
        samples=val_samples,
        tokenizer=tokenizer,
        max_len=args.max_len,
        stride=args.stride,
        use_sliding_window=use_sliding_window,
        is_training=True,
    )
    print(
        f"[INFO] Số cửa sổ (features) -> Train: {len(train_features)} | Val: {len(val_features)}"
    )

    train_dataset = ViQuADDataset(train_features, is_training=True)
    val_dataset = ViQuADDataset(val_features, is_training=True)

    # 3. Nạp mô hình Extractive QA
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)

    # Xử lý tương thích tham số eval_strategy giữa các phiên bản transformers
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"

    training_kwargs = {
        "output_dir": args.output_dir,
        eval_key: "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 1,  # Giữ 1 checkpoint gần nhất để tránh đầy dung lượng Google Drive
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "fp16": (not args.no_fp16) and torch.cuda.is_available(),
        "dataloader_pin_memory": torch.cuda.is_available(),
        "logging_steps": args.logging_steps,
        "seed": args.seed,
        "report_to": report_to,
        "run_name": run_name,
    }

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    # 4. Huấn luyện (hỗ trợ resume nếu Colab bị disconnect)
    print("[INFO] Bắt đầu huấn luyện...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    eval_metrics = trainer.evaluate()

    # 5. Lưu mô hình tốt nhất, tokenizer và log kết quả vào output_dir
    best_model_dir = os.path.join(args.output_dir, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    summary = {
        "args": vars(args),
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "num_train_features": len(train_features),
        "num_val_features": len(val_features),
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }
    summary_path = os.path.join(args.output_dir, "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n================ KẾT THÚC HUẤN LUYỆN ================")
    print(f"Best model đã lưu tại : {best_model_dir}")
    print(f"Tóm tắt kết quả       : {summary_path}")
    print(f"Validation Loss tốt nhất: {eval_metrics.get('eval_loss', -1):.4f}")


if __name__ == "__main__":
    main()
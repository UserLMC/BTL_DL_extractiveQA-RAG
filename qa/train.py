import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Hỗ trợ chạy cả `python -m qa.train` lẫn `python qa/train.py`
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from qa.postprocess import apply_null_threshold, postprocess_qa_predictions
from qa.preprocess import ViQuADDataset, convert_samples_to_features
from qa.qa_metrics import (
    compute_exact,
    compute_f1,
    evaluate_qa,
    find_best_null_threshold,
    metric_max_over_ground_truths,
    normalize_answer,
)


def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_detailed_predictions(
    samples: List[Dict],
    raw_preds: Dict[str, str],
    final_preds: Dict[str, str],
    null_diffs: Dict[str, float],
    output_path: str,
) -> None:
    """Lưu chi tiết dự đoán từng mẫu trên tập test để chạy error_analysis.py."""
    records = []
    for s in samples:
        qid = str(s["id"])
        is_imp = bool(s.get("is_impossible", False))
        raw_ans = s.get("answers", {})
        if isinstance(raw_ans, dict):
            golds = [t for t in raw_ans.get("text", []) if normalize_answer(t)]
        elif isinstance(raw_ans, list):
            golds = [a["text"] for a in raw_ans if normalize_answer(a.get("text", ""))]
        else:
            golds = []
        if is_imp or not golds:
            golds = []

        pred_final = final_preds.get(qid, "")
        records.append(
            {
                "id": qid,
                "question": s["question"],
                "context": s["context"],
                "is_impossible": is_imp or (len(golds) == 0),
                "gold_answers": golds,
                "raw_span_pred": raw_preds.get(qid, ""),
                "final_pred": pred_final,
                "null_score_diff": round(null_diffs.get(qid, 0.0), 4),
                "em": metric_max_over_ground_truths(compute_exact, pred_final, golds),
                "f1": round(metric_max_over_ground_truths(compute_f1, pred_final, golds), 4),
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def log_qa_metrics_to_tensorboard(
    tb_run_dir: str,
    args: argparse.Namespace,
    best_threshold: float,
    val_metrics_0: Dict[str, float],
    val_metrics_tuned: Dict[str, float],
    test_metrics: Dict[str, float],
    global_step: int,
) -> None:
    """Ghi toàn bộ chỉ số QA (EM, F1, HasAns, NoAns) và HParams lên TensorBoard."""
    writer = SummaryWriter(log_dir=tb_run_dir)

    writer.add_scalar("qa_val_thresh0/EM", val_metrics_0.get("EM", 0.0), global_step)
    writer.add_scalar("qa_val_thresh0/F1", val_metrics_0.get("F1", 0.0), global_step)

    for key in ["EM", "F1", "HasAns_EM", "HasAns_F1", "NoAns_EM", "NoAns_F1"]:
        if key in val_metrics_tuned:
            writer.add_scalar(f"qa_val_tuned/{key}", val_metrics_tuned[key], global_step)
        if test_metrics and key in test_metrics:
            writer.add_scalar(f"qa_test/{key}", test_metrics[key], global_step)

    writer.add_scalar("qa_val_tuned/best_null_threshold", best_threshold, global_step)

    # Lưu bảng HParams để so sánh trực quan các run trong tab HPARAMS của TensorBoard
    hparam_dict = {
        "model_name": str(args.model_name),
        "lr": float(args.lr),
        "epochs": int(args.epochs),
        "max_len": int(args.max_len),
        "stride": int(args.stride),
        "eff_batch_size": int(args.batch_size * args.grad_accum),
        "debug_fraction": float(args.debug_fraction),
    }
    metric_dict = {
        "hparam/val_EM": float(val_metrics_tuned.get("EM", 0.0)),
        "hparam/val_F1": float(val_metrics_tuned.get("F1", 0.0)),
        "hparam/best_null_threshold": float(best_threshold),
    }
    if test_metrics:
        metric_dict["hparam/test_EM"] = float(test_metrics.get("EM", 0.0))
        metric_dict["hparam/test_F1"] = float(test_metrics.get("F1", 0.0))

    writer.add_hparams(hparam_dict, metric_dict)
    writer.flush()
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune XLM-R cho Extractive QA (UIT-ViQuAD 2.0)")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
    parser.add_argument("--train_file", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/processed/val.jsonl")
    parser.add_argument("--test_file", type=str, default="data/processed/test.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/xlmr_baseline")

    # Thư mục lưu log TensorBoard chung cho tất cả các run
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/tensorboard",
        help="Thư mục gốc lưu log TensorBoard (mỗi run sẽ tạo 1 thư mục con bên trong)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Tên run hiển thị trên TensorBoard (mặc định tự tạo theo siêu tham số)",
    )

    # Siêu tham số theo kế hoạch
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size mỗi GPU")
    parser.add_argument("--grad_accum", type=int, default=4, help="8 * 4 = batch hiệu dụng 32")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # Tham số debug & postprocess
    parser.add_argument("--debug_fraction", type=float, default=1.0, help="Đặt 0.1 để chạy thử 10% dữ liệu")
    parser.add_argument("--n_best_size", type=int, default=20)
    parser.add_argument("--max_answer_length", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_only", action="store_true", help="Chỉ đánh giá từ checkpoint tại --model_name")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Tự đặt tên run rõ ràng để dễ phân biệt trên TensorBoard
    if args.run_name:
        run_name = args.run_name
    else:
        frac_tag = f"_dbg{int(args.debug_fraction * 100)}pct" if args.debug_fraction < 1.0 else ""
        run_name = f"xlmr_lr{args.lr}_ep{args.epochs}_len{args.max_len}_s{args.stride}{frac_tag}"

    tb_run_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(tb_run_dir, exist_ok=True)

    print(f"[INFO] Run name: {run_name}")
    print(f"[INFO] TensorBoard log dir: {tb_run_dir}")
    print(f"[INFO] Nạp tokenizer và model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)

    # 1. Nạp dữ liệu
    train_samples = load_jsonl(args.train_file)
    val_samples = load_jsonl(args.val_file)
    test_samples = load_jsonl(args.test_file) if os.path.exists(args.test_file) else []

    if 0.0 < args.debug_fraction < 1.0:
        train_samples = train_samples[: max(10, int(len(train_samples) * args.debug_fraction))]
        val_samples = val_samples[: max(10, int(len(val_samples) * args.debug_fraction))]
        if test_samples:
            test_samples = test_samples[: max(10, int(len(test_samples) * args.debug_fraction))]
        print(f"[DEBUG] Chế độ {args.debug_fraction*100:.0f}%: train={len(train_samples)}, val={len(val_samples)}")

    # 2. Tiền xử lý sliding window
    print(f"[INFO] Tiền xử lý dữ liệu (max_len={args.max_len}, stride={args.stride})...")
    val_features = convert_samples_to_features(
        val_samples, tokenizer, max_len=args.max_len, stride=args.stride, is_training=True
    )
    val_dataset_loss = ViQuADDataset(val_features, is_training=True)
    val_dataset_pred = ViQuADDataset(val_features, is_training=False)

    train_dataset = None
    if not args.eval_only:
        train_features = convert_samples_to_features(
            train_samples, tokenizer, max_len=args.max_len, stride=args.stride, is_training=True
        )
        train_dataset = ViQuADDataset(train_features, is_training=True)
        print(f"[INFO] Số cửa sổ trượt: train={len(train_features)}, val={len(val_features)}")

    # 3. Cấu hình TrainingArguments với TensorBoard (tương thích cả transformers cũ & mới)
    os.environ["TENSORBOARD_LOGGING_DIR"] = tb_run_dir
    ta_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to=["tensorboard"],
        run_name=run_name,
        dataloader_num_workers=2 if torch.cuda.is_available() else 0,
        seed=args.seed,
    )
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in ta_params:
        ta_kwargs["eval_strategy"] = "epoch"
    else:
        ta_kwargs["evaluation_strategy"] = "epoch"
    if "logging_dir" in ta_params:
        ta_kwargs["logging_dir"] = tb_run_dir

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset_loss,
    )

    # 4. Huấn luyện
    global_step = 0
    if not args.eval_only:
        train_result = trainer.train()
        global_step = int(train_result.global_step)
        best_model_dir = os.path.join(args.output_dir, "best_model")
        trainer.save_model(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)
        print(f"[INFO] Đã lưu mô hình tốt nhất tại: {best_model_dir}")

    # 5. Dự đoán trên Validation & tìm ngưỡng null tốt nhất
    print("[INFO] Đánh giá trên tập Validation và tìm ngưỡng null tối ưu...")
    val_raw_preds = trainer.predict(val_dataset_pred).predictions
    val_non_null_preds, val_null_diffs, _ = postprocess_qa_predictions(
        examples=val_samples,
        features=val_features,
        raw_predictions=val_raw_preds,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
    )

    val_metrics_default = evaluate_qa(
        val_samples, val_non_null_preds, val_null_diffs, null_threshold=0.0
    )
    best_threshold, _ = find_best_null_threshold(
        val_samples, val_non_null_preds, val_null_diffs
    )
    val_metrics_tuned = evaluate_qa(
        val_samples, val_non_null_preds, val_null_diffs, null_threshold=best_threshold
    )

    print(f"[VAL - thresh=0.0]           {val_metrics_default}")
    print(f"[VAL - best_thresh={best_threshold:.4f}] {val_metrics_tuned}")

    summary = {
        "run_name": run_name,
        "tb_log_dir": tb_run_dir,
        "config": vars(args),
        "best_null_threshold": best_threshold,
        "val_metrics_thresh_0": val_metrics_default,
        "val_metrics_tuned": val_metrics_tuned,
    }

    # 6. Đánh giá trên tập Test (dùng ngưỡng null đã chốt trên Validation)
    test_metrics: Dict[str, float] = {}
    if test_samples:
        print("[INFO] Đánh giá trên tập Test bằng ngưỡng null đã chốt trên Validation...")
        test_features = convert_samples_to_features(
            test_samples, tokenizer, max_len=args.max_len, stride=args.stride, is_training=False
        )
        test_dataset_pred = ViQuADDataset(test_features, is_training=False)
        test_raw_preds = trainer.predict(test_dataset_pred).predictions
        test_non_null_preds, test_null_diffs, _ = postprocess_qa_predictions(
            examples=test_samples,
            features=test_features,
            raw_predictions=test_raw_preds,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
        )

        test_metrics = evaluate_qa(
            test_samples, test_non_null_preds, test_null_diffs, null_threshold=best_threshold
        )
        print(f"[TEST - thresh={best_threshold:.4f}]    {test_metrics}")
        summary["test_metrics"] = test_metrics

        final_test_preds = apply_null_threshold(
            test_non_null_preds, test_null_diffs, best_threshold
        )
        save_detailed_predictions(
            samples=test_samples,
            raw_preds=test_non_null_preds,
            final_preds=final_test_preds,
            null_diffs=test_null_diffs,
            output_path=os.path.join(args.output_dir, "test_predictions.json"),
        )

    # 7. Ghi chỉ số QA & HParams vào TensorBoard
    log_qa_metrics_to_tensorboard(
        tb_run_dir=tb_run_dir,
        args=args,
        best_threshold=best_threshold,
        val_metrics_0=val_metrics_default,
        val_metrics_tuned=val_metrics_tuned,
        test_metrics=test_metrics,
        global_step=global_step,
    )

    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Kết quả: {args.output_dir}/metrics_summary.json")
    print(f"[DONE] Xem TensorBoard: tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()
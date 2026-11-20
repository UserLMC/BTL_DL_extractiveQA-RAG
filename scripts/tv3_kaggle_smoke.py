"""
tv3_kaggle_smoke.py - Việc T2 + T5 của TV3: nạp Qwen 4-bit, đo tốc độ sinh, chạy baseline closed-book (S0).

Chạy trên Kaggle/Colab GPU (T4):
    pip install -q -r requirements-rag.txt
    python scripts/tv3_kaggle_smoke.py --backend hf --model Qwen/Qwen2.5-7B-Instruct \
        --questions data/smoke_questions.jsonl --out outputs/smoke_closed_book.jsonl \
        --report outputs/smoke_report.json --wandb-project qa-rag-vi
Chạy thử không cần GPU:  --backend mock

Đầu ra:
  * outputs/smoke_closed_book.jsonl : câu trả lời S0 cho từng câu thử (để đọc bằng mắt: bịa hay từ chối?)
  * outputs/smoke_report.json       : tên mô hình, phiên bản thư viện, GPU, thời gian nạp, độ trễ, token/giây
    -> dán vào báo cáo mục "phiên bản mô hình" và dùng để quyết định giữ bản 7B hay chuyển sang 3B.
Quy tắc quyết định bản 3B: nếu độ trễ trung bình > --slow-threshold giây/câu thì script in khuyến nghị
(kế hoạch: 120 câu x 6 cấu hình ~ 720 lượt sinh).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rag.generation.generate import log_wandb, make_backend, read_jsonl  # noqa: E402
from rag.generation.prompts import build_messages, is_refusal, refusal_kind  # noqa: E402


def env_info() -> Dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("torch", "transformers", "bitsandbytes", "accelerate"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        else:
            info["gpu"] = None
    except Exception:
        info["gpu"] = None
    return info


def run_smoke(questions: List[Dict], backend, out_path: str, warmup: bool = True) -> Dict:
    if warmup:  # lượt đầu thường chậm do biên dịch kernel/cấp phát bộ nhớ, không tính vào thống kê
        backend.generate(build_messages("Xin chào?", closed_book=True))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    rows, lat, toks = [], [], []
    with open(out_path, "w", encoding="utf-8") as f:
        for q in questions:
            msgs = build_messages(q["question"], None, version="v2", closed_book=True)
            t0 = time.time()
            ans, n_tok = backend.generate(msgs)
            dt = time.time() - t0
            lat.append(dt)
            toks.append(n_tok)
            row = {"qid": q["qid"], "question": q["question"], "note": q.get("note", ""), "config": "S0",
                   "answer": ans, "refusal_kind": refusal_kind(ans), "latency_s": round(dt, 3),
                   "n_new_tokens": n_tok}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = len(rows)
    total_t, total_tok = sum(lat), sum(toks)
    return {
        "n_questions": n,
        "latency_mean_s": round(total_t / n, 3) if n else None,
        "latency_max_s": round(max(lat), 3) if lat else None,
        "tokens_per_s": round(total_tok / total_t, 2) if total_t else None,
        "refusal_rate": round(100 * sum(is_refusal(r["answer"]) for r in rows) / n, 1) if n else None,
        "est_hours_720_generations": round(720 * (total_t / n) / 3600, 2) if n else None,
    }


def main(argv=None) -> Dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=os.path.join(ROOT, "data", "smoke_questions.jsonl"))
    ap.add_argument("--backend", choices=["hf", "mock"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--out", default="outputs/smoke_closed_book.jsonl")
    ap.add_argument("--report", default="outputs/smoke_report.json")
    ap.add_argument("--slow-threshold", type=float, default=15.0)
    ap.add_argument("--wandb-project")
    a = ap.parse_args(argv)

    t0 = time.time()
    backend = make_backend(a.backend, a.model, a.max_new_tokens)
    load_s = round(time.time() - t0, 1)
    stats = run_smoke(read_jsonl(a.questions), backend, a.out)
    report = {"model_id": getattr(backend, "model_id", a.model), "backend": a.backend,
              "max_new_tokens": a.max_new_tokens, "decoding": "greedy", "load_time_s": load_s,
              "env": env_info(), **stats}
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if a.backend == "hf" and stats["latency_mean_s"] and stats["latency_mean_s"] > a.slow_threshold:
        print(f"\n[KHUYẾN NGHỊ] {stats['latency_mean_s']}s/câu > {a.slow_threshold}s: "
              f"dùng Qwen/Qwen2.5-3B-Instruct và --max-new-tokens 150 (theo bảng rủi ro của kế hoạch).")
    log_wandb(a.wandb_project, "tv3-smoke", {"model": report["model_id"], "env": report["env"]}, {"smoke": stats})
    return report


if __name__ == "__main__":
    main()

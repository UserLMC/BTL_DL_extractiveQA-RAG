# Kết quả thực nghiệm Extractive QA & Reader S6 (TV1)

Tệp này được tạo tự động bởi `python scripts/table.py` từ các file kết quả trong `results/qa/` và `results/rag/`.

## Bảng 1. Thống kê bộ dữ liệu UIT-ViQuAD 2.0 sau khi chia theo bài viết (seed=42)

| Tập dữ liệu | Số bài viết | Số đoạn văn | Tổng câu hỏi | Có đáp án (HasAns) | Không đáp án (NoAns) | % NoAns | Dài Context TB (từ) | Dài Đáp án TB (từ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train (90% train gốc) | 124 | 3.774 | 20.930 | 14.128 | 6.802 | 32.50% | 181.1 | 9.9 |
| Validation (10% train gốc) | 14 | 327 | 1.835 | 1.260 | 575 | 31.34% | 179.6 | 11.1 |
| Test (Official Dev) | 138 | 3.171 | 5.692 | 3.852 | 1.840 | 32.33% | 179.1 | 9.8 |

## Bảng 2. Kết quả tinh chỉnh siêu tham số `xlm-roberta-base` (Validation & Test)

Tất cả các cấu hình dùng `doc_stride = 128`, batch hiệu dụng = 32, `fp16`, chọn checkpoint theo `eval_loss` tốt nhất và chọn ngưỡng từ chối $\tau_{\text{null}}$ tối ưu F1 trên tập Validation (1.835 câu) trước khi đánh giá cố định trên tập Test (5.692 câu: 3.852 HasAns, 1.840 NoAns).

| Cấu hình | `lr` | Epochs | `max_len` | $\tau_{\text{null}}$ (Val) | Val EM | Val F1 | Test EM | Test F1 | Test HasAns EM / F1 | Test NoAns Acc (EM) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vòng 1: lr=2e-5, ep=2, len=384 | `2e-05` | 2 | 384 | +0.1035 | 37.98 | 53.39 | 38.39 | 52.07 | 39.07 / 59.29 | 36.96 |
| Vòng 1: lr=3e-5, ep=2, len=384 (Baseline) | `3e-05` | 2 | 384 | -0.0762 | 38.75 | 54.75 | 39.69 | 52.95 | 39.95 / 59.55 | 39.13 |
| Vòng 1: lr=5e-5, ep=2, len=384 | `5e-05` | 2 | 384 | +0.3711 | 40.44 | 57.60 | 38.93 | 53.75 | 43.51 / 65.41 | 29.35 |
| Vòng 2: lr=5e-5, ep=3, len=384 (**Tốt nhất**) | `5e-05` | 3 | 384 | +0.3477 | **41.31** | **58.40** | **40.48** | **54.69** | 42.13 / 63.13 | 37.01 |
| Vòng 2: lr=5e-5, ep=3, len=512 | `5e-05` | 3 | 512 | -0.0898 | 41.42 | 56.67 | 41.04 | 53.01 | 38.73 / 56.42 | 45.87 |

## Bảng 3. Ảnh hưởng của tinh chỉnh ngưỡng từ chối $\tau_{\text{null}}$ trên tập Validation (n = 1.835)

| Cấu hình | Val F1 ($\tau=0$) | Val HasAns F1 ($\tau=0$) | Val NoAns Acc ($\tau=0$) | $\tau_{\text{null}}^*$ tối ưu | Val F1 ($\tau^*$) | Val HasAns F1 ($\tau^*$) | Val NoAns Acc ($\tau^*$) | $\Delta$ Val F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vòng 1: lr=2e-5, ep=2, len=384 | 53.09 | 59.38 | 39.30 | +0.1035 | 53.39 | 60.53 | 37.74 | +0.30 |
| Vòng 1: lr=3e-5, ep=2, len=384 (Baseline) | 54.28 | 61.75 | 37.91 | -0.0762 | 54.75 | 61.32 | 40.35 | +0.47 |
| Vòng 1: lr=5e-5, ep=2, len=384 | 56.73 | 64.85 | 38.96 | +0.3711 | 57.60 | 68.49 | 33.74 | +0.87 |
| Vòng 2: lr=5e-5, ep=3, len=384 (**Tốt nhất**) | 58.13 | 64.49 | 44.17 | +0.3477 | 58.40 | 67.19 | 39.13 | +0.27 |
| Vòng 2: lr=5e-5, ep=3, len=512 | 56.13 | 59.76 | 48.17 | -0.0898 | 56.67 | 59.11 | 51.30 | +0.54 |

## Bảng 4. Phân rã kết quả và lỗi trên tập Test của mô hình tốt nhất (`xlmr_run2_lr5e5_ep3_len384`)

| Nhóm câu hỏi | Phân loại kết quả / lỗi | Số câu | Tỉ lệ trong nhóm (%) | Tỉ lệ toàn tập Test (%) |
|---|---|---:|---:|---:|
| HasAns (Có đáp án) | Đúng hoàn toàn (`EM = 1`) | 1,624 | 42.16% | 28.53% |
| HasAns (Có đáp án) | Đúng một phần / lệch biên (`0 < F1 < 1`) | 1,267 | 32.89% | 22.26% |
| HasAns (Có đáp án) | Trích xuất sai hoàn toàn (`F1 = 0, pred != ''`) | 149 | 3.87% | 2.62% |
| HasAns (Có đáp án) | Từ chối sai / bỏ sót đáp án (False Negative) | 812 | 21.08% | 14.27% |
| NoAns (Không đáp án) | Từ chối đúng (True Negative) | 681 | 37.01% | 11.96% |
| NoAns (Không đáp án) | Trả lời bừa câu không có đáp án (False Positive) | 1,159 | 62.99% | 20.36% |

## Bảng 5A. Hiệu năng theo loại câu hỏi trên tập Test (n = 5.692)

| Loại câu hỏi | Số câu (n) | Tỉ lệ (%) | EM (%) | F1 (%) |
|---|---:|---:|---:|---:|
| Khi nào / Thời gian | 405 | 7.12% | 49.4 | 59.9 |
| Ai / Nhân vật | 538 | 9.45% | 46.8 | 55.4 |
| Ở đâu / Địa điểm | 397 | 6.97% | 43.6 | 56.5 |
| Bao nhiêu / Số lượng | 649 | 11.40% | 41.9 | 55.8 |
| Cái gì / Khác | 2,941 | 51.67% | 39.2 | 53.8 |
| Tại sao / Cách thức | 762 | 13.39% | 33.6 | 53.1 |

## Bảng 5B. Hiệu năng theo độ dài đáp án trên các câu có đáp án (HasAns, n = 3.852)

| Nhóm độ dài đáp án | Số câu (n) | Tỉ lệ trong HasAns (%) | EM (%) | F1 (%) | Nhận xét |
|---|---:|---:|---:|---:|---|
| 1–3 từ | 1,249 | 32.42% | 49.0 | 62.7 | Thực thể/con số ngắn: EM cao nhất |
| 4–7 từ | 967 | 25.10% | 47.7 | 65.7 | Cụm từ vừa phải: F1 đạt đỉnh (65.7%) |
| 8–15 từ | 835 | 21.68% | 38.0 | 63.7 | Mệnh đề dài: dễ lệch từ biên (EM giảm, F1 vẫn cao) |
| >15 từ | 801 | 20.79% | 29.2 | 60.5 | Giải thích/liệt kê dài: lệch biên nhiều (F1 cao hơn EM >31 điểm) |

## Bảng 6. Kết quả dùng XLM-R làm Extractive Reader trong RAG (Cấu hình S6, 124 câu hỏi)

Mô hình `xlmr_run2_lr5e5_ep3_len384` ($\tau_{\text{null}} = 0.3477$) đọc độc lập từng chunk trong top-3, chọn span có `null_score_diff` nhỏ nhất (tự tin có đáp án nhất) rồi áp ngưỡng từ chối.

| Cấu hình Reader + Retriever | Overall Token-F1 | In-corpus Token-F1 (n=92) | Hit Token-F1 | Miss Token-F1 | Từ chối sai trên câu có đáp án | Từ chối đúng trên câu ngoài kho (n=32) |
|---|---:|---:|---:|---:|---:|---:|
| S6 – Dense (`e5-base`, top-3) | **37.73** | 20.42 | 23.42 (n=80) | 0.43 (n=12) | 50.00% | **87.50%** |
| S6_bm25 – BM25 (`pyvi`, top-3) | **38.32** | 20.13 | 22.05 (n=84) | 0.00 (n=8) | 50.00% | **90.62%** |

### Bảng 6B. Phân tích S6 theo loại câu hỏi trên bộ test RAG (124 câu)

| Loại câu hỏi | Số câu (n) | S6 (Dense top-3) Token-F1 | S6 (Dense) Tỉ lệ từ chối | S6 (BM25 top-3) Token-F1 | S6 (BM25) Tỉ lệ từ chối |
|---|---:|---:|---:|---:|---:|
| `fact` (Tra cứu sự kiện) | 22 | 31.33 | 50.00% | 35.70 | 40.91% |
| `number` (Con số / Thời hạn) | 29 | 20.67 | 44.83% | 16.36 | 51.72% |
| `condition` (Điều kiện) | 24 | 20.48 | 45.83% | 20.48 | 45.83% |
| `multi` (Ghép nhiều Điều) | 17 | 5.80 | 64.71% | 5.91 | 64.71% |
| `out_of_corpus` (Ngoài kho) | 32 | 87.50 | 87.50% | 90.62 | 90.62% |

## Phụ lục: Mã LaTeX chuẩn ACL (`booktabs`) cho Báo cáo Overleaf

```latex
\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccc}
\toprule
\textbf{Cấu hình XLM-R} & $\tau_{\text{null}}$ & \textbf{Val F1} & \textbf{Test EM} & \textbf{Test F1} & \textbf{HasAns F1} & \textbf{NoAns EM} \\
\midrule
lr=2e-5, ep=2, L=384 & +0.10 & 53.39 & 38.39 & 52.07 & 59.29 & 36.96 \\
lr=3e-5, ep=2, L=384 & -0.08 & 54.75 & 39.69 & 52.95 & 59.55 & 39.13 \\
lr=5e-5, ep=2, L=384 & +0.37 & 57.60 & 38.93 & 53.75 & 65.41 & 29.35 \\
\textbf{lr=5e-5, ep=3, L=384} & +0.35 & \textbf{58.40} & 40.48 & \textbf{54.69} & 63.13 & 37.01 \\
lr=5e-5, ep=3, L=512 & -0.09 & 56.67 & 41.04 & 53.01 & 56.42 & 45.87 \\
\bottomrule
\end{tabular}
\caption{Kết quả tinh chỉnh siêu tham số của XLM-RoBERTa-base trên UIT-ViQuAD 2.0 (Val $n=1{,}835$; Test $n=5{,}692$).}
\label{tab:qa-results}
\end{table}
```

# Đề tài 10 – Hỏi đáp tiếng Việt: Extractive QA & RAG

- **Extractive QA** trên UIT-ViQuAD 2.0 (XLM-R), đánh giá EM/F1 — *TV1*
- **RAG** trên kho quy chế của ĐHQGHN / Trường ĐH Công nghệ: retriever BM25 và dense, đánh giá Recall@k, MRR — *TV2*
- **Sinh câu trả lời bằng LLM và đánh giá ảo giác** — *TV3*

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Mọi lệnh bên dưới chạy từ thư mục gốc của repo.

## Cấu trúc

```
data/
  corpus/sources.csv           # danh sách văn bản + link PDF (PDF không đưa vào repo)
  corpus/ocr_fixes.tsv         # bảng sửa lỗi OCR (áp dụng tự động)
  corpus/manual/               # trang chép tay thay cho OCR (bảng phụ lục)
  corpus/text/*.txt            # văn bản đã làm sạch (đầu ra của parse_pdf)
  corpus/chunks*.jsonl         # các cách chia đoạn
  rag_test/                    # bộ câu hỏi test RAG + hướng dẫn soạn câu hỏi
rag/
  corpus/      download.py, parse_pdf.py, chunk.py
  retrieval/   text.py, bm25.py, dense.py, gold.py, eval_retrieval.py, export_topk.py
  hallucination/  annotation.py (bảng gán nhãn ảo giác, đọc nhãn)
scripts/       retrieval_tables.py, retrieval_errors.py, hallucination_breakdown.py, plot_figures.py
results/       retrieval/ (kết quả từng câu + TABLES.md), hallucination/, figures/
tests/
```

## RAG – Kho tài liệu (TV2)

Kho gồm 6 văn bản, khoảng 43.000 từ, chia thành 224 chunk theo Điều:

| doc_id | Văn bản | Cách lấy chữ |
|---|---|---|
| vnu_3626 | Quy chế đào tạo đại học tại ĐHQGHN (3626/QĐ-ĐHQGHN, 2022) | OCR (scan 53 trang) |
| vnu_32 | Quy chế Công tác sinh viên tại ĐHQGHN (32/QĐ-ĐHQGHN, 2017) | OCR (scan 28 trang) + chép tay phụ lục |
| vnu_1011 | Hướng dẫn tổ chức đào tạo học phần ngoại ngữ, chuẩn đầu ra ngoại ngữ (1011/HD-ĐHQGHN, 2023) | lớp chữ PDF |
| uet_1931 | Quy định xét, cấp học bổng KKHT – Trường ĐHCN (2026) | lớp chữ PDF |
| uet_798 | Quy định khen thưởng sinh viên – Trường ĐHCN (2023) | lớp chữ PDF |
| uet_dc2025 | Điều chỉnh quy định khen thưởng (2025) | lớp chữ PDF |

```bash
python -m rag.corpus.download                     # tải PDF vào data/pdfs/
python -m rag.corpus.parse_pdf                    # PDF -> data/corpus/text/*.txt
python -m rag.corpus.chunk                        # chia theo Điều -> data/corpus/chunks.jsonl
python -m rag.corpus.chunk --mode window --max-words 150   # cho ablation
python -m rag.corpus.chunk --mode window --max-words 300
```

- **OCR**: dùng EasyOCR tiếng Việt (GPU, khoảng 6 giây/trang). Kết quả OCR thô được cache theo từng trang trong `data/cache/ocr/`, nên khi bị ngắt giữa chừng sẽ chạy tiếp từ trang dở. Các lỗi có hệ thống (`dào`→`đào`, `chưong`→`chương`, `/`→`l`, …) được sửa qua `data/corpus/ocr_fixes.tsv`. Danh sách lỗi `d`/`đ` được lập bằng thống kê: những từ chỉ xuất hiện trong bản OCR mà không có trong các văn bản có lớp chữ. Bảng phụ lục khung kỷ luật của `vnu_32` OCR không đọc được nên đã chép tay vào `data/corpus/manual/`.
- **Làm sạch**: bỏ số trang, bỏ header/footer lặp lại, nối các dòng bị ngắt thành đoạn, bỏ phần "Nơi nhận"/chữ ký.
- **Chia đoạn theo Điều**: mỗi Điều là một chunk; Điều dài hơn 300 từ thì cắt thành nhiều phần gối nhau 50 từ, mỗi phần giữ tiêu đề "Điều N." ở đầu. Văn bản không chia Điều (`vnu_1011`) thì chia theo mục cấp 1. Phần Quyết định ban hành và phần Quy chế được tách riêng (`p0`, `p1`) vì đều đánh số từ Điều 1. Phần đính kèm sau chữ ký mà không mở đầu bằng Điều (văn bản sửa đổi `uet_dc2025`) được giữ thành một khối phụ lục. `gold.py` báo lỗi nếu một đoạn trích của bộ câu hỏi không thuộc chunk nào. Mỗi chunk lưu metadata: văn bản, chương, Điều, tiêu đề, cùng vị trí ký tự trong file `.txt`.

## RAG – Truy hồi (TV2)

```bash
python -m rag.retrieval.gold data/rag_test/questions_tv2.jsonl --write    # kiểm tra evidence, điền gold_chunks
python -m rag.retrieval.eval_retrieval --retriever bm25 --errors 10
python -m rag.retrieval.eval_retrieval --retriever dense --model e5-base
python -m rag.retrieval.eval_retrieval --retriever dense --model bkai
python scripts/retrieval_tables.py           # tái tạo mọi bảng -> results/retrieval/TABLES.md
python scripts/retrieval_errors.py results/retrieval/<kết quả>.json      # xuất câu sai để phân tích
```

- **BM25**: `rank_bm25` (Okapi), tách từ bằng pyvi, chữ thường; k1 = 1.5, b = 0.75. Nội dung được đánh chỉ mục gồm tên văn bản, tiêu đề Điều và nội dung.
- **Dense**: bi-encoder qua `sentence-transformers` + FAISS `IndexFlatIP` trên vector đã chuẩn hóa L2, nên inner product chính là cosine. Mô hình `intfloat/multilingual-e5-base` dùng tiền tố `query:` / `passage:`. Mô hình `bkai-foundation-models/vietnamese-bi-encoder` dựa trên PhoBERT, cần tách từ bằng pyvi và giới hạn 256 token.
- **Đánh giá**: Recall@1/3/5/10 và MRR@10, chỉ tính trên câu có đáp án trong kho. Đáp án vàng là **đoạn trích nguyên văn (evidence)**: một chunk được tính là đúng nếu phủ ít nhất 50% đoạn trích. Nhờ vậy cùng một bộ câu hỏi dùng được cho mọi cách chia đoạn. Câu cần nhiều evidence thì Recall@k là tỉ lệ evidence được lấy về.

Định dạng bộ câu hỏi và quy tắc soạn: [data/rag_test/HUONG_DAN_SOAN_CAU_HOI.md](data/rag_test/HUONG_DAN_SOAN_CAU_HOI.md).

### Kết quả truy hồi dùng chung cho phần sinh câu trả lời

```bash
python -m rag.retrieval.export_topk      # -> results/retrieval/retrieval_top10.jsonl + recall_by_k.csv
```

`retrieval_top10.jsonl` chứa top-10 của **mọi** câu hỏi (cả câu ngoài kho) cho `bm25` (S1) và `dense` = e5-base (S2, S4, S5, S6), đúng định dạng mà `rag/generation/generate.py` đọc qua `--retrieval`. `recall_by_k.csv` là Recall@k với k = 1..10 (ablation k).

### Gán nhãn và phân tích ảo giác (phần TV2)

```bash
# 1. Tạo bảng gán nhãn từ câu trả lời của S0, S2, S3
python -m rag.hallucination.annotation make --gen outputs/generations.jsonl
# 2. Mỗi người gán nhãn file sheet_<TVx>.csv của mình (nhập vào Google Sheet), tải về dạng CSV
#    và lưu vào results/hallucination/annotation/labels/<TVx>.csv
python -m rag.hallucination.annotation check        # nhãn thiếu, mã sai, mục chung hoà phiếu
# 3. Tỉ lệ ảo giác theo loại câu hỏi, theo hit/miss của retriever; rồi vẽ hình
python scripts/hallucination_breakdown.py --gen outputs/generations.jsonl
python scripts/plot_figures.py                      # -> results/figures/*.pdf
```

- Chọn 60 câu (phân tầng theo loại, ít nhất 15 câu ngoài kho) × 3 cấu hình = 180 câu trả lời. 30 câu trả lời cả 3 người cùng gán (để tính kappa), 150 câu còn lại chia đều, mỗi người nhận đủ cả 3 cấu hình.
- Bảng của người gán không có tên cấu hình, mã mục được cấp sau khi xáo trộn. Ánh xạ nằm trong `key.csv`: **không mở file này trước khi gán xong**. S0 vẫn nhận ra được vì không có ngữ cảnh, và S3 thường chỉ có 1 đoạn: hai điểm này được ghi vào Limitations.
- Mục chung: lấy nhãn đa số. Mục hoà phiếu thì cả nhóm bàn rồi ghi nhãn chốt vào `labels/final.csv` (cột `ma_muc, nhan, trich_dan_sai`).
- Tỉ lệ ảo giác = (nhãn 3 + nhãn 4) / số câu trả lời. "Hit" nghĩa là ngữ cảnh đưa cho LLM có ít nhất một chunk vàng.

## Extractive QA (TV1)

*(TV1 bổ sung)*

## Sinh câu trả lời & đánh giá ảo giác (TV3)

*(TV3 bổ sung)*

## Kiểm thử

```bash
python -m pytest -q tests
```

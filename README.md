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
  retrieval/   text.py, bm25.py, dense.py, gold.py, eval_retrieval.py
scripts/       retrieval_tables.py, retrieval_errors.py
results/       retrieval/ (kết quả từng câu + TABLES.md)
tests/
```

## RAG – Kho tài liệu (TV2)

Kho gồm 6 văn bản, khoảng 43.000 từ, chia thành 223 chunk theo Điều:

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
- **Chia đoạn theo Điều**: mỗi Điều là một chunk; Điều dài hơn 300 từ thì cắt thành nhiều phần gối nhau 50 từ, mỗi phần giữ tiêu đề "Điều N." ở đầu. Văn bản không chia Điều (`vnu_1011`) thì chia theo mục cấp 1. Phần Quyết định ban hành và phần Quy chế được tách riêng (`p0`, `p1`) vì đều đánh số từ Điều 1. Mỗi chunk lưu metadata: văn bản, chương, Điều, tiêu đề, cùng vị trí ký tự trong file `.txt`.

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

## Extractive QA (TV1)

*(TV1 bổ sung)*

## Sinh câu trả lời & đánh giá ảo giác (TV3)

*(TV3 bổ sung)*

## Kiểm thử

```bash
python -m pytest -q tests
```

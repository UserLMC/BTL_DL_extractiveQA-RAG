# Hướng dẫn soạn bộ câu hỏi test RAG

Mục tiêu là khoảng 120 câu, mỗi người 40 câu: khoảng 100 câu có đáp án trong kho và khoảng 20 câu **không có** trong kho. Bộ câu hỏi dùng chung cho cả đánh giá truy hồi (Recall@k, MRR) và đánh giá ảo giác (S0–S6). Sau buổi review chéo ngày **CN 22/11**, bộ câu hỏi được đóng băng thành `rag_test_v1` và không sửa nữa.

## 1. File và định dạng

Mỗi người viết vào một file riêng: `data/rag_test/questions_tv1.jsonl`, `questions_tv2.jsonl`, `questions_tv3.jsonl`. Mỗi dòng là một JSON:

```json
{"id": "tv1_007", "question": "Điểm trung bình học kỳ thấp đến mức nào thì bị cảnh báo học vụ?",
 "answer": "Dưới 0,80 đối với học kỳ đầu của khóa học, dưới 1,00 đối với các học kỳ tiếp theo.",
 "answerable": true, "type": "number",
 "evidence": [{"doc_id": "vnu_3626", "text": "Điểm trung bình chung học kỳ đạt dưới 0,80 đối với học kỳ đầu của khóa học, dưới 1,00 đối với các học kỳ tiếp theo"}],
 "author": "TV1"}
```

| Trường | Bắt buộc | Ghi chú |
|---|---|---|
| `id` | có | `tv1_001`, `tv2_001`, …, không trùng nhau |
| `question` | có | Viết như sinh viên hỏi thật, **không** chép nguyên câu trong văn bản (xem mục 3) |
| `answer` | có | Đáp án chuẩn, ngắn gọn, đủ ý. Câu ngoài kho thì ghi `"Không tìm thấy thông tin trong tài liệu"` |
| `answerable` | có | `true` / `false` |
| `type` | có | `fact`, `number`, `condition`, `multi` (mục 2); câu ngoài kho ghi `out_of_corpus` |
| `evidence` | có nếu `answerable` | Danh sách đoạn trích. Câu `multi` cần ≥ 2 đoạn trích (thường ở 2 Điều khác nhau) |
| `author` | có | `TV1` / `TV2` / `TV3` |
| `gold_chunks` | **không tự điền** | Script tự sinh từ `evidence` |

### Quy tắc cho `evidence` (quan trọng)
- Chép **nguyên văn** từ `data/corpus/text/<doc_id>.txt` (chính file đã OCR/làm sạch, **không** chép từ PDF). Được phép khác khoảng trắng và hoa/thường, ngoài ra phải khớp từng ký tự.
- Trích **ngắn nhưng đủ** để trả lời: thường 1–2 câu (khoảng 15–60 từ). Đừng chép cả Điều.
- Nếu đáp án nằm trong một văn bản sửa đổi (ví dụ `uet_dc2025` sửa Điều 3 của `uet_798`) thì evidence lấy từ **văn bản còn hiệu lực**.
- Nếu thấy lỗi OCR trong đoạn cần trích, báo TV2 để sửa trong `data/corpus/ocr_fixes.tsv`. **Không** tự sửa tay file `.txt`, vì sửa tay sẽ bị ghi đè khi chạy lại.

Vì sao dùng đoạn trích thay vì id chunk? Chunk sẽ thay đổi khi làm ablation (chia theo Điều hoặc cửa sổ 150/300 từ). Đoạn trích thì không đổi, nên một chunk được tính là đúng nếu nó phủ ≥ 50% đoạn trích.

## 2. Bốn loại câu (+ câu ngoài kho)

| `type` | Mô tả | Ví dụ |
|---|---|---|
| `fact` | Tra cứu một sự kiện hoặc định nghĩa | "Học phần tiên quyết là gì?" |
| `number` | Đáp án là con số, mốc thời gian, tỉ lệ | "Khóa luận tốt nghiệp có bao nhiêu tín chỉ?" |
| `condition` | Điều kiện, trường hợp, quy trình | "Điều kiện để được xét học bổng loại giỏi là gì?" |
| `multi` | Cần ghép thông tin từ ≥ 2 Điều/văn bản | "Sinh viên đạt danh hiệu Xuất sắc có được học bổng loại xuất sắc không, mức bao nhiêu?" |
| `out_of_corpus` | Hỏi về chủ đề hợp lý với sinh viên nhưng kho **không** có | "Ký túc xá Mỹ Đình giá phòng bao nhiêu một tháng?" |

Tỉ lệ mỗi người nhắm tới (40 câu): khoảng 10 `fact`, 10 `number`, 8 `condition`, 5 `multi`, 7 `out_of_corpus`. TV3 phụ trách nhiều câu ngoài kho hơn (khoảng 20 câu) theo kế hoạch, nên hai người còn lại có thể giảm phần này.

Câu **"bẫy"** (nên có vài câu): câu ngoài kho nhưng nghe rất giống nội dung có trong kho, ví dụ hỏi mức học bổng của trường khác, hỏi quy chế đào tạo thạc sĩ (kho chỉ có quy chế đại học), hoặc hỏi về một năm học không được quy định. Những câu này đo xem LLM có từ chối đúng không.

## 3. Viết câu hỏi tốt

- **Diễn đạt lại** câu hỏi, không chép cụm từ dài trong văn bản. Nếu chép nguyên văn, BM25 sẽ thắng dễ dàng và kết quả so sánh với dense mất ý nghĩa. Nên dùng từ đồng nghĩa và cách nói của sinh viên ("bị đuổi học" thay cho "buộc thôi học", "học lại" thay cho "đăng ký học lại").
- Mỗi câu hỏi chỉ có **một** cách hiểu, và đáp án kiểm chứng được bằng evidence.
- Không hỏi những gì phải suy luận ngoài văn bản ("Em có nên…?").
- Rải đều các văn bản và các chương; không dồn 10 câu vào cùng một Điều.

## 4. Phân công văn bản (tránh trùng)

| Người | Văn bản chính |
|---|---|
| TV1 | `vnu_3626` (Quy chế đào tạo), nửa sau: từ Chương IV trở đi |
| TV2 | `vnu_3626` nửa đầu (Chương I–III), `vnu_1011` (ngoại ngữ), `uet_1931` (học bổng KKHT) |
| TV3 | `vnu_32` (Công tác sinh viên, rèn luyện, kỷ luật), `uet_798` + `uet_dc2025` (khen thưởng); khoảng 20 câu ngoài kho |

Câu `multi` được phép ghép văn bản ngoài phần của mình.

## 5. Kiểm tra trước khi nộp

```bash
python -m rag.retrieval.gold data/rag_test/questions_tv2.jsonl          # báo lỗi nếu không tìm thấy evidence
python -m rag.retrieval.gold data/rag_test/questions_tv2.jsonl --write  # điền gold_chunks
```

Review chéo (CN 22/11): TV1 kiểm câu của TV2, TV2 kiểm TV3, TV3 kiểm TV1. Người kiểm tra xác nhận: đáp án đúng chưa, evidence có thật sự trả lời được câu hỏi không, câu ngoài kho có thật sự không có trong kho không (tìm thử bằng Ctrl+F trong các file `.txt`).

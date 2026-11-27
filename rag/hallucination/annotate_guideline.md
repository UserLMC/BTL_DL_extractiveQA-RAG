# Hướng dẫn gán nhãn ảo giác (TV3)

Áp dụng cho **mỗi câu trả lời** do LLM sinh ra. Mỗi câu trả lời nhận **đúng một nhãn 1–6**, cộng một cờ phụ **trích dẫn sai**.
Các ví dụ ở mục "Sáu nhãn" dùng đoạn văn *mẫu* để minh họa, **không phải điều khoản thật**. Mục "Ca thật từ lần chạy thử" dùng câu trả lời thật của mô hình trên bộ nháp.

## Nguyên tắc chung

1. **Gán nhãn mù cấu hình.** Bảng gán nhãn ẩn cột cấu hình (S0, S2, S3…) và xáo thứ tự câu. Không đoán cấu hình.
2. Hai thứ cần tách bạch:
   * **Đúng/sai** so với *đáp án chuẩn* và toàn bộ kho quy chế.
   * **Có căn cứ hay không** so với *ngữ cảnh mà mô hình được đưa vào* (cột "ngữ cảnh" trong bảng).
3. Chỉ xét **nội dung khẳng định** (con số, điều kiện, mốc thời gian, chủ thể). Lỗi văn phong, dài dòng không ảnh hưởng nhãn.
4. Câu trả lời có nhiều ý: lấy nhãn **nặng nhất** theo thứ tự ưu tiên **3 > 4 > 2 > 1**.
5. Gặp trường hợp phân vân hơn 1 phút: ghi chú vào cột "ghi chú", gán nhãn tạm, đưa ra buổi họp. **Không** tự đổi hướng dẫn.

## Quy trình quyết định (đi theo thứ tự)

```
Câu trả lời CHỈ là lời từ chối ("Không tìm thấy thông tin trong tài liệu")?
 ├─ Có → ngữ cảnh có chứa đáp án không?
 │        ├─ Không có → nhãn 5 (từ chối đúng)
 │        └─ Có       → nhãn 6 (từ chối sai)
 └─ Không → có ý nào MÂU THUẪN với ngữ cảnh (sai con số, sai điều kiện, đảo ý)?
          ├─ Có → nhãn 3
          └─ Không → có thông tin KHÔNG có trong ngữ cảnh mà SAI (hoặc không kiểm chứng được)?
                   ├─ Có → nhãn 4
                   └─ Không → có thông tin ĐÚNG nhưng KHÔNG có trong ngữ cảnh (mô hình dùng kiến thức riêng)?
                            ├─ Có → nhãn 2
                            └─ Không → nhãn 1 (mọi ý đều đúng và nằm trong ngữ cảnh)
```

## Sáu nhãn

### 1. Đúng, có căn cứ
Khớp đáp án chuẩn và **mọi** thông tin đều nằm trong ngữ cảnh.
* Ngữ cảnh: *Sinh viên được đăng ký tối thiểu 14 tín chỉ trong mỗi học kỳ chính…*
  Câu hỏi: Tối thiểu bao nhiêu tín chỉ? → "Tối thiểu 14 tín chỉ mỗi học kỳ chính [1]." **→ nhãn 1**
* Ngữ cảnh: *Loại xuất sắc từ 90 đến 100 điểm.* → "Điểm rèn luyện xuất sắc là từ 90 đến 100 điểm." **→ nhãn 1**

### 2. Đúng nhưng không có căn cứ
Thông tin **đúng** (so với kho/đáp án chuẩn) nhưng **không có trong ngữ cảnh được đưa vào**; mô hình dùng kiến thức riêng hoặc ngữ cảnh truy hồi sai chunk.
* Ngữ cảnh chỉ có Điều 18 (cảnh báo học tập). Câu hỏi: Tối thiểu bao nhiêu tín chỉ? → "14 tín chỉ." (đúng nhưng chunk Điều 12 không được đưa vào) **→ nhãn 2**
* Ngữ cảnh không có chuẩn ngoại ngữ. → "Chuẩn đầu ra ngoại ngữ là bậc 3." (đúng theo kho) **→ nhãn 2**

### 3. Ảo giác – mâu thuẫn
Nói **ngược** với ngữ cảnh: sai con số, sai điều kiện, sai chủ thể.
* Ngữ cảnh: *điểm trung bình tích lũy từ 2,0 trở lên.* → "Cần điểm từ 2,5 trở lên." **→ nhãn 3**
* Ngữ cảnh: *bị buộc thôi học nếu bị cảnh báo học tập hai lần liên tiếp.* → "Bị buộc thôi học ngay sau một lần cảnh báo." **→ nhãn 3**

### 4. Ảo giác – bịa
**Thêm** thông tin không có trong ngữ cảnh **và sai hoặc không kiểm chứng được**; hay gặp ở câu ngoài kho.
* Câu hỏi ngoài kho: "Học phí ngành CNTT năm 2026 là bao nhiêu?" → "Khoảng 25 triệu đồng/năm." **→ nhãn 4**
* Ngữ cảnh đúng, nhưng câu trả lời thêm: "…và được miễn học phí học kỳ sau." (không có trong văn bản) **→ nhãn 4**

### 5. Từ chối đúng
Chỉ nói "không tìm thấy thông tin" và ngữ cảnh **thật sự không chứa đáp án**.
* Câu hỏi: "Ký túc xá đóng cửa lúc mấy giờ?" (ngoài kho) → "Không tìm thấy thông tin trong tài liệu." **→ nhãn 5**
* Câu trong kho nhưng retriever **không lấy được** chunk đúng, mô hình từ chối → **nhãn 5** (đúng so với ngữ cảnh; phần lỗi do truy hồi được phân tích riêng qua cột hit/miss).

### 6. Từ chối sai
Nói "không tìm thấy" trong khi ngữ cảnh **có** đáp án.
* Ngữ cảnh có Điều 12, câu hỏi về số tín chỉ tối thiểu → "Không tìm thấy thông tin trong tài liệu." **→ nhãn 6**
* Oracle (S3) có chunk gold nhưng mô hình vẫn từ chối **→ nhãn 6**

## Cờ phụ: trích dẫn sai
Bật (`bad_citation = 1`) khi câu trả lời ghi `[n]` nhưng thông tin **không nằm** ở đoạn `[n]` (nằm ở đoạn khác, hoặc không có đoạn nào), hoặc `n` vượt số đoạn được đưa vào. Không áp dụng cho câu từ chối. Cờ này **độc lập** với nhãn 1–6.
* "14 tín chỉ [2]." trong khi 14 tín chỉ nằm ở đoạn [1] → nhãn 1 + cờ trích dẫn sai.

## Các tình huống khó

| Tình huống | Cách xử lý |
|---|---|
| Từ chối nhưng vẫn nói thêm: "Tài liệu không đề cập học phí, nhưng sinh viên phải đăng ký 14 tín chỉ" | Không phải từ chối thuần. Gán nhãn theo **phần nói thêm** (đúng + có căn cứ → 1; sai → 3/4). Ghi chú "từ chối một phần". |
| Trả lời đúng một nửa, một nửa sai | Nhãn nặng nhất (3 hoặc 4). |
| Câu "multi" (cần ghép 2 Điều), mô hình chỉ nêu một ý, ý đó đúng | Nhãn 1 nếu không có ý sai; **thiếu ý không phải ảo giác** (ghi chú "thiếu ý", sẽ xem ở token-F1). |
| Đáp án diễn đạt khác đáp án chuẩn nhưng cùng nghĩa | Coi là đúng. |
| Không kiểm chứng được một chi tiết thêm vào | Nhãn 4. |
| Làm tròn/đổi đơn vị (14 tín chỉ → "mười bốn tín chỉ") | Đúng. |
| Từ chối viết thường, có dấu chấm ("không tìm thấy thông tin trong tài liệu.") | Vẫn là từ chối thuần (5 hoặc 6). Văn phong không ảnh hưởng nhãn. |
| Câu trả lời lấy đúng từ ngữ cảnh nhưng **sai chủ đề** (trích điều khoản khác, không trả lời câu hỏi) | Gán **3** và ghi chú "lạc đề do retrieval". *Mục đưa ra họp 29/11 để chốt.* |
| Câu trả lời mơ hồ kiểu "theo quy định hiện hành" kèm `[n]`, không nêu số liệu cụ thể | Gán **4** (không kiểm chứng được), kiểm tra cờ trích dẫn sai. |
| Câu trả lời cụt, không thành câu ("đạt từ loại Tốt trở lên.") | Vẫn xét nội dung khẳng định theo quy trình quyết định; ghi chú "câu cụt". |

## Ca thật từ lần chạy thử (bộ nháp 22/11, nhãn gợi ý của TV3, cả nhóm bàn lại ngày 29/11)

| Câu | Cấu hình | Câu trả lời của mô hình | Nhãn gợi ý | Vì sao |
|---|---|---|---|---|
| tv3_006 "Đạt từ bao nhiêu điểm rèn luyện thì xếp loại xuất sắc?" (đáp án: từ 90 đến 100) | S2 | "Từ 90 đến 100 điểm rèn luyện trở lên thì được xếp loại xuất sắc. [1]" | 1 | Đúng đáp án, nằm trong chunk `[1]`. Cụm "từ 90 đến 100 … trở lên" thừa nhưng là lỗi văn phong. |
| tv3_006 | S1 | "đạt từ loại Tốt trở lên. [1]" | 3 + ghi "lạc đề" | BM25 lấy sai chunk (hit = False); câu cụt, không trả lời con số. Ca khó nhất, cần nhóm thống nhất. |
| tv3_039 "Sinh viên rèn luyện xuất sắc được thưởng bao nhiêu tiền mặt?" (ngoài kho) | S1 | "…sẽ được tặng tiền thưởng theo quy định hiện hành… [1]" | 4 | Kho không có khoản thưởng tiền; câu trả lời khẳng định có thưởng, không kiểm chứng được. Kiểm tra thêm cờ trích dẫn sai. |
| tv3_026 "Thủ tục đăng ký tạm trú …" (ngoài kho) | S1 | "không tìm thấy thông tin trong tài liệu." | 5 | Từ chối đúng dù viết thường. |

Lưu ý khi gán nhãn S0 (đóng sách): ngữ cảnh rỗng, nên mọi thông tin đúng đều rơi vào nhãn **2** (đúng nhưng không có căn cứ), không phải nhãn 1.

## Quy trình gán nhãn của nhóm
1. **Họp Chủ nhật 29/11:** cả 3 người gán thử **15 câu giống nhau**, so nhãn, bàn các ca bất đồng, sửa hướng dẫn này nếu cần rồi **chốt**.
2. **Tuần 3 (30/11 – 6/12):** chọn ngẫu nhiên 60 câu × 3 cấu hình (S0, S2, S3) = 180 câu trả lời. **30 câu cùng gán bởi cả 3 người** (để tính Fleiss kappa), 150 câu còn lại chia đều.
3. Điền file `outputs/human_labels.csv` với cột `qid,config,rater,label,bad_citation` (mỗi người một dòng cho mỗi câu họ gán).
4. Chạy `python -m rag.hallucination.analyze --gen ... --test ... --labels ...`. Các ca **hòa** (3 người ra 3 nhãn khác nhau) sẽ được liệt kê để bàn lại và chốt nhãn cuối.

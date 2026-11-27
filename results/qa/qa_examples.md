# Ví dụ Phân tích lỗi Extractive QA (xlmr_run2_lr5e5_ep3_len384)

## 1. Ba ví dụ dự đoán ĐÚNG tiêu biểu

### Đúng 1 – HasAns khớp hoàn toàn (đáp án ngắn)
- **ID:** `uit_019995` | **Loại câu:** Ai / Nhân vật | **EM:** 100.0 | **F1:** 100.0 | **null_score_diff:** `-2.1201`
- **Câu hỏi:** Ai được nhìn nhận phổ biến là một trong các nhà văn quan trọng nhất của thế kỷ XX?
- **Đáp án chuẩn:** James Joyce
- **Dự đoán XLM-R:** James Joyce
- **Ngữ cảnh (trích):** Trong thế kỷ XX, Ireland có bốn người đoạt giải Nobel văn học là George Bernard Shaw, William Butler Yeats, Samuel Beckett và Seamus Heaney. Dù không được giải Nobel, song James Joyce được nhìn nhận phổ biến là một trong các nhà văn quan trọng nhất của thế kỷ XX. Tiểu thuyết Ulysses vào năm 1922 của ông được cho là một trong số các tác phẩm quan trọng nhất của văn học hiện đại chủ nghĩa và cuộc đời ông được kỷ niệm thường niên vào ngày 16 tháng 6...

### Đúng 2 – HasAns khớp hoàn toàn (đáp án dài)
- **ID:** `uit_012103` | **Loại câu:** Cái gì / Khác | **EM:** 100.0 | **F1:** 100.0 | **null_score_diff:** `-2.6951`
- **Câu hỏi:** Hệ thống đứt gãy trên mặt đá vôi ở Phong Nha - Kẻ Bàng đã tạo điều kiện cho điều gì xảy ra?
- **Đáp án chuẩn:** nước dễ thấm vào các khối đá làm tăng khả năng hòa tan
- **Dự đoán XLM-R:** nước dễ thấm vào các khối đá làm tăng khả năng hòa tan
- **Ngữ cảnh (trích):** Hệ thống đứt gãy chằng chịt trên mặt đá vôi tại khu vực Phong Nha-Kẻ Bàng đã tạo điều kiện cho nước dễ thấm vào các khối đá làm tăng khả năng hòa tan do trong nước có chứa các chất axít có phản ứng với đá vôi (cácbonat canxi). Hệ thống hang động hùng vĩ của Phong Nha được tạo ra do quá trình các khe nứt kiến tạo, sau đó là quá trình phong hoá vật lý và hoá học đã gặm mòn, hoà tan, rửa trôi qua hàng triệu năm. Trong điều kiện nhiệt đới ẩm, quá trì...

### Đúng 3 – NoAns từ chối đúng (không trả lời bừa)
- **ID:** `uit_023459` | **Loại câu:** Tại sao / Cách thức | **EM:** 100.0 | **F1:** 100.0 | **null_score_diff:** `1.4912`
- **Câu hỏi:** Giáo sư Charité đã nhận định như thế nào về trường hợp thủ nghiệm thành công phương pháp ghép tủy xương giúp khỏi HIV?
- **Đáp án chuẩn:** (Không có đáp án)
- **Dự đoán XLM-R:** (Từ chối / Rỗng)
- **Ngữ cảnh (trích):** Ngày 16-17/12/2010, nhiều báo chí Việt Nam đưa tin các bác sĩ tại đại học y khoa Charité ở Berlin, Đức đã thử nghiệm thành công điều trị HIV bằng phương pháp ghép tủy xương người khác Tuy nhiên đây mới chỉ là một trường hợp cá biệt, hơn nữa việc không còn phát hiện HIV trong máu bệnh nhân không có nghĩa rằng anh ta đã khỏi bệnh (HIV có thể tồn tại ẩn mình trong tế bào nội tạng trước khi sinh sôi bùng phát trở lại sau mấy năm). Theo lời giải thích...

## 2. Ba ví dụ dự đoán SAI điển hình

### Sai 1 – NoAns bị trả lời bừa (False Positive)
- **ID:** `uit_002796` | **Loại câu:** Cái gì / Khác | **EM:** 0.0 | **F1:** 0.0 | **null_score_diff:** `-0.5115`
- **Câu hỏi:** Các quốc gia Thổ Nhĩ Kỳ nào tiếp giáp với biên giới của châu Âu?
- **Đáp án chuẩn:** (Không có đáp án)
- **Dự đoán XLM-R:** Hy Lạp và Bulgaria
- **Ngữ cảnh (trích):** Phần thuộc châu Âu của Thổ Nhĩ Kỳ là Đông Thrace, có biên giới với Hy Lạp và Bulgaria. Phần thuộc châu Á của Thổ Nhĩ Kỳ là Anatolia, gồm một cao nguyên trung ương có độ cao lớn cùng các đồng bằng duyên hải hẹp, nằm giữa các dãy núi Köroğlu và Pontic tại phía bắc và Taurus tại phía nam. Miền đông của Thổ Nhĩ Kỳ có cảnh quan đồi núi hơn và là nguồn của các sông như Euphrates, Tigris và Aras, và có điểm cao nhất Thổ Nhĩ Kỳ là Núi Ararat với cao độ 5...

### Sai 2 – HasAns bị bỏ qua / từ chối sai (False Negative)
- **ID:** `uit_021775` | **Loại câu:** Cái gì / Khác | **EM:** 0.0 | **F1:** 0.0 | **null_score_diff:** `0.8336`
- **Câu hỏi:** Kế hoạch về Bức tường đối với phía Tây được gọi là gì?
- **Đáp án chuẩn:** những biện pháp cứng rắn
- **Dự đoán XLM-R:** (Từ chối / Rỗng)
- **Ngữ cảnh (trích):** Tuy quân đội Đồng Minh phía Tây đã có thông tin về kế hoạch của "những biện pháp cứng rắn" nhằm để phong tỏa Tây Berlin thông qua những người tin cậy, nhưng họ vẫn ngạc nhiên về thời điểm cụ thể và quy mô của rào cản này. Do quyền ra vào Tây Berlin không bị cắn xén nên phương Tây đã không can thiệp bằng quân sự. Cơ quan tình báo Liên bang Đức (Bundesnachrichtendienst - BND) cũng đã nhận được nhiều thông tin tương tự ngay từ giữa tháng 7. Sau khi ...

### Sai 3 – HasAns cắt lệch biên đáp án (Partial Match)
- **ID:** `uit_000270` | **Loại câu:** Cái gì / Khác | **EM:** 0.0 | **F1:** 57.63 | **null_score_diff:** `-3.1056`
- **Câu hỏi:** Những lợi ích nào có được từ sự khác nhau giữa toán tử hiện và ẩn?
- **Đáp án chuẩn:** sử dụng nhiều hơn hằng số có phạm vi rộng, 'uốn nắn' các thanh ghi liên tục (lưu giá trị hằng số khác đè lên giá trị đã có của thanh ghi) và rất nhiều ưu điểm vượt trội khác
- **Dự đoán XLM-R:** cho phép sử dụng nhiều hơn hằng số có phạm vi rộng, 'uốn nắn' các thanh ghi liên tục
- **Ngữ cảnh (trích):** Không phải tất cả các máy tính hay chỉ thị đơn lẻ đều có toán hạng hiện (rõ ràng). Một máy tính thanh chứa có sự kết hợp giữa toán hạng trái và kết quả tính toán lưu trong một thanh chứa ẩn đối với hầu hết các chỉ thị đại số. Một số kiến trúc khác (như 8086 hay x86) có phiên bản sử dụng thanh chứa của các chỉ thị thông dụng, và thanh chứa được xem như là một trong những thanh ghi tổng quát nhất của chỉ thị dài. Trong khi đó, một máy tính ngăn xếp...

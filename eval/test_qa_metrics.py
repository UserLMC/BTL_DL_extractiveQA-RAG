import unittest
import unicodedata
from eval.qa_metrics import (
    normalize_answer,
    compute_exact,
    compute_f1,
    evaluate_qa,
    find_best_null_threshold,
)


class TestQAMetrics(unittest.TestCase):
    def test_unicode_nfc_normalization(self):
        """Kiểm tra cùng một từ ở dạng tổ hợp (NFD) và dựng sẵn (NFC) phải khớp EM = 1.0."""
        s_nfc = unicodedata.normalize("NFC", "Đại học Quốc gia Hà Nội")
        s_nfd = unicodedata.normalize("NFD", "Đại học Quốc gia Hà Nội")
        self.assertNotEqual(s_nfc, s_nfd)  # Trước khi chuẩn hóa thì khác nhau
        self.assertEqual(normalize_answer(s_nfc), normalize_answer(s_nfd))
        self.assertEqual(compute_exact(s_nfc, s_nfd), 1.0)

    def test_punctuation_and_case(self):
        """Kiểm tra xóa dấu câu tiếng Việt (ngoặc kép, chấm, phẩy) và chữ hoa/thường."""
        gold = "“Quyết định số 3626/QĐ-ĐHQGHN”."
        pred = "quyết định số 3626 qđ đhqghn"
        self.assertEqual(compute_exact(gold, pred), 1.0)
        self.assertEqual(compute_f1(gold, pred), 1.0)

    def test_hand_calculated_f1(self):
        """Kiểm tra ví dụ tính tay: Gold 6 từ, Pred 5 từ, trùng 4 từ -> F1 = 8/11."""
        gold = "Đại học Quốc gia Hà Nội"
        pred = "trường Đại học Quốc gia"
        self.assertEqual(compute_exact(gold, pred), 0.0)
        self.assertAlmostEqual(compute_f1(gold, pred), 8.0 / 11.0, places=5)

    def test_no_answer_handling(self):
        """Kiểm tra xử lý câu không có đáp án (NoAns)."""
        # Dự đoán đúng là không có đáp án ("")
        self.assertEqual(compute_exact("", ""), 1.0)
        self.assertEqual(compute_f1("", ""), 1.0)
        # Câu không có đáp án nhưng mô hình lại sinh ra chữ -> 0 điểm
        self.assertEqual(compute_exact("", "20 tín chỉ"), 0.0)
        self.assertEqual(compute_f1("", "20 tín chỉ"), 0.0)

    def test_evaluate_qa_overall_hasans_noans(self):
        """Kiểm tra hàm tổng hợp điểm All, HasAns, NoAns và ngưỡng null_threshold."""
        samples = [
            {
                "id": "q1",
                "is_impossible": False,
                "answers": {"text": ["Hà Nội", "Thủ đô Hà Nội"]},
            },
            {
                "id": "q2",
                "is_impossible": True,
                "answers": {"text": []},
            },
        ]
        preds = {
            "q1": "hà nội.",   # Khớp hoàn toàn với đáp án chuẩn đầu tiên -> EM=1, F1=1
            "q2": "",          # Từ chối đúng câu NoAns -> EM=1, F1=1
        }
        res = evaluate_qa(samples, preds)
        self.assertEqual(res["EM"], 100.0)
        self.assertEqual(res["F1"], 100.0)
        self.assertEqual(res["HasAns_F1"], 100.0)
        self.assertEqual(res["NoAns_F1"], 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
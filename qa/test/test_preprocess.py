import json
import os
import unittest
from transformers import AutoTokenizer
from qa.preprocess import convert_samples_to_features


class TestQAPreprocess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        train_path = "data/processed/train.jsonl"
        if os.path.exists(train_path):
            with open(train_path, "r", encoding="utf-8") as f:
                all_samples = [json.loads(line) for line in f]
            # Lấy 15 câu có đáp án và 5 câu không có đáp án = 20 ví dụ kiểm tay
            has_ans = [s for s in all_samples if not s["is_impossible"]][:15]
            no_ans = [s for s in all_samples if s["is_impossible"]][:5]
            cls.samples_20 = has_ans + no_ans
        else:
            cls.samples_20 = []

    def test_20_real_examples_offset_and_cls(self):
        """
        Kiểm tra trên 20 ví dụ thực tế:
        1. Câu NoAns phải có start_positions == 0 và end_positions == 0 (nhãn CLS).
        2. Câu HasAns phải có ít nhất 1 cửa sổ khôi phục lại đúng chuỗi đáp án gốc qua offset_mapping.
        """
        self.assertEqual(len(self.samples_20), 20, "Cần có file data/processed/train.jsonl để lấy 20 mẫu!")
        features = convert_samples_to_features(
            self.samples_20, self.tokenizer, max_len=384, stride=128, is_training=True
        )

        # Gom các cửa sổ theo example_id
        by_id = {}
        for feat in features:
            by_id.setdefault(feat["example_id"], []).append(feat)

        for idx, sample in enumerate(self.samples_20):
            qid = str(sample["id"])
            win_feats = by_id[qid]
            context = sample["context"]

            if sample["is_impossible"]:
                # Mọi cửa sổ của câu NoAns đều phải gán nhãn về CLS (0, 0)
                for wf in win_feats:
                    self.assertEqual(wf["start_positions"], wf["cls_index"])
                    self.assertEqual(wf["end_positions"], wf["cls_index"])
            else:
                gold_ans = sample["answers"]["text"][0]
                recovered_spans = []
                for wf in win_feats:
                    s_idx = wf["start_positions"]
                    e_idx = wf["end_positions"]
                    if s_idx != wf["cls_index"] or e_idx != wf["cls_index"]:
                        char_start = wf["offset_mapping"][s_idx][0]
                        char_end = wf["offset_mapping"][e_idx][1]
                        recovered_spans.append(context[char_start:char_end])

                self.assertTrue(
                    len(recovered_spans) >= 1,
                    f"Mẫu HasAns thứ {idx} (id={qid}) không tìm được span trên bất kỳ cửa sổ nào!",
                )
                # Chuỗi cắt ra từ offset_mapping phải chứa trọn vẹn đáp án chuẩn
                self.assertIn(gold_ans.strip(), recovered_spans[0])

        print(f"\n[PASS] Đã kiểm tra thành công 20 ví dụ thực tế ({len(features)} cửa sổ trượt)!")

    def test_600_token_sliding_window_and_boundary_split(self):
        """
        Kiểm tra bài toán lý thuyết: Đoạn văn ~600 token chia cửa sổ 384, stride 128.
        Khi đáp án bị cắt một nửa ở rìa cửa sổ 1 nhưng nằm trọn trong phần chồng lấp của cửa sổ 2:
        - Cửa sổ 1 (bị cắt nửa đáp án) phải gán nhãn CLS (0, 0).
        - Cửa sổ 2 (chứa trọn đáp án nhờ stride=128) phải gán nhãn đúng vị trí token.
        """
        # Tạo đoạn văn dài khoảng 600 token
        prefix_words = "Quy chế đào tạo đại học tại Đại học Quốc gia Hà Nội quy định về tín chỉ. " * 22
        target_answer = "sinh viên phải tích lũy tối thiểu 120 tín chỉ và đạt chuẩn ngoại ngữ B1"
        suffix_words = " Các học phần tự chọn được đăng ký vào đầu mỗi học kỳ chính thức." * 20

        context = prefix_words + target_answer + suffix_words
        ans_start = context.index(target_answer)

        synthetic_sample = [
            {
                "id": "synth_600",
                "title": "Test_Sliding_Window",
                "context": context,
                "question": "Điều kiện xét tốt nghiệp của sinh viên là gì?",
                "is_impossible": False,
                "answers": {"text": [target_answer], "answer_start": [ans_start]},
            }
        ]

        feats = convert_samples_to_features(
            synthetic_sample, self.tokenizer, max_len=384, stride=128, is_training=True
        )
        # Đoạn ~600 token phải sinh ra ít nhất 2 cửa sổ trượt
        self.assertGreaterEqual(len(feats), 2)

        # Kiểm tra bất kỳ cửa sổ nào không chứa trọn đáp án đều phải trỏ về (0, 0)
        valid_windows = 0
        for wf in feats:
            s_pos, e_pos = wf["start_positions"], wf["end_positions"]
            if s_pos == 0 and e_pos == 0:
                continue
            valid_windows += 1
            c_start = wf["offset_mapping"][s_pos][0]
            c_end = wf["offset_mapping"][e_pos][1]
            self.assertEqual(context[c_start:c_end], target_answer)

        self.assertGreaterEqual(valid_windows, 1)
        print(f"[PASS] Đã kiểm tra đoạn 600 token: chia thành {len(feats)} cửa sổ, khôi phục chính xác đáp án!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
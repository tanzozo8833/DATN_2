import unittest
from src.utils.metrics import calculate_wer

class TestMetrics(unittest.TestCase):
    def test_wer_calculation(self):
        # Case 1: Khớp hoàn toàn
        ref = [[5, 6, 7]]
        hyp = [[5, 6, 7]]
        self.assertEqual(calculate_wer(hyp, ref), 0.0)

        # Case 2: Sai 1 từ (Substitution)
        hyp = [[5, 9, 7]]
        self.assertAlmostEqual(calculate_wer(hyp, ref), 1/3)

        # Case 3: Thừa 1 từ (Insertion)
        hyp = [[5, 6, 7, 8]]
        self.assertAlmostEqual(calculate_wer(hyp, ref), 1/3)

        print("[v] Logic tính WER chính xác.")

if __name__ == '__main__':
    unittest.main()
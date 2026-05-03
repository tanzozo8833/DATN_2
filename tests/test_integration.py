import unittest
import subprocess
import os
import pickle
import torch

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Tạo môi trường giả lập tối thiểu để chạy thử main.py."""
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("configs", exist_ok=True)
        os.makedirs("weights", exist_ok=True)

        # Mock Dictionary
        with open("data/processed/gloss2ids.pkl", "wb") as f:
            pickle.dump({'<s>': 0, '<pad> 1': 1, 'hust': 2}, f)

        # Mock Data (2 mẫu cho mỗi tập)
        mock_data = {
            's1': {'keypoint': torch.randn(32, 77, 3), 'gloss': 'HUST'},
            's2': {'keypoint': torch.randn(32, 77, 3), 'gloss': 'HUST'}
        }
        for p in ["train_77.pkl", "dev_77.pkl"]:
            with open(f"data/processed/{p}", "wb") as f:
                pickle.dump(mock_data, f)

        # Mock Config (77 points chia 5 luồng)
        with open("configs/base_config.yaml", "w") as f:
            f.write("""
data:
  streams:
    body: {indices: [0,1,2,3,4,5,6,7,8]}
    left_hand: {indices: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
    right_hand: {indices: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
    mouth: {indices: [0,1,2,3,4,5,6,7,8,9,10,11]}
    face: {indices: [0,1,2,3,4,5,6,7,8,9,10,11,12,13]}
""")

    def test_dry_run(self):
        """Kiểm tra xem chạy main.py có lỗi cú pháp hay runtime cơ bản không."""
        # Chạy main.py nhưng giới hạn epoch=1 để test nhanh
        # Ta có thể mock train_params trực tiếp nếu cần, hoặc chạy qua CLI
        print("\n[*] Đang chạy thử tích hợp hệ thống (Dry run)...")
        try:
            # Thay đổi tham số trực tiếp trong code để test nhanh
            from main import main as run_main
            run_main() 
            self.assertTrue(os.path.exists("weights/light_mska_best.pth"))
            print("[v] Test tích hợp thành công. Hệ thống sẵn sàng huấn luyện thực tế.")
        except Exception as e:
            self.fail(f"Chạy tích hợp thất bại: {str(e)}")

if __name__ == '__main__':
    unittest.main()
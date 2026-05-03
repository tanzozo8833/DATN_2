

import unittest
import os
import pickle
import torch
from torch.utils.data import DataLoader
from src.datasets.slr_dataset import SLRDataset, slr_collate_fn

class TestSLRDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Tạo dữ liệu giả lập (Mock data) để test mà không cần file thật.
        """
        cls.test_dict_path = "tests/mock_gloss2ids.pkl"
        cls.test_data_path = "tests/mock_test_77.pkl"
        os.makedirs("tests", exist_ok=True)

        # 1. Tạo từ điển giả lập
        mock_dict = {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, 'DRUCK': 4, 'TIEF': 5, 'KOMMEN': 6}
        with open(cls.test_dict_path, 'wb') as f:
            pickle.dump(mock_dict, f)

        # 2. Tạo dữ liệu keypoint giả lập (2 mẫu với số khung hình T khác nhau)
        # Mẫu 1: 10 khung hình
        # Mẫu 2: 15 khung hình
        cls.mock_data = {
            'sample1': {
                'keypoint': torch.randn(10, 77, 3),
                'gloss': 'DRUCK TIEF KOMMEN'
            },
            'sample2': {
                'keypoint': torch.randn(15, 77, 3),
                'gloss': 'TIEF KOMMEN UNKNOWN_WORD'
            }
        }
        with open(cls.test_data_path, 'wb') as f:
            pickle.dump(cls.mock_data, f)

    @classmethod
    def tearDownClass(cls):
        """Xóa dữ liệu giả lập sau khi test xong."""
        if os.path.exists(cls.test_dict_path):
            os.remove(cls.test_dict_path)
        if os.path.exists(cls.test_data_path):
            os.remove(cls.test_data_path)

    def test_dataset_initialization(self):
        """Kiểm tra Dataset có load đúng số lượng mẫu không."""
        dataset = SLRDataset(self.test_data_path, self.test_dict_path)
        self.assertEqual(len(dataset), 2)
        print("[v] Dataset load đúng số lượng mẫu.")

    def test_gloss_to_id_conversion(self):
        """Kiểm tra chuyển đổi từ chữ sang ID, bao gồm cả từ lạ (<unk>)."""
        dataset = SLRDataset(self.test_data_path, self.test_dict_path)
        
        # Test mẫu 1: DRUCK(4) TIEF(5) KOMMEN(6)
        _, label1 = dataset[0]
        self.assertTrue(torch.equal(label1, torch.tensor([4, 5, 6])))

        # Test mẫu 2: TIEF(5) KOMMEN(6) UNKNOWN(3)
        _, label2 = dataset[1]
        self.assertTrue(torch.equal(label2, torch.tensor([5, 6, 3])))
        print("[v] Chuyển đổi Gloss sang ID chính xác (bao gồm xử lý <unk>).")

    def test_collate_fn_padding(self):
        """Kiểm tra logic padding và trả về độ dài cho CTC Loss."""
        dataset = SLRDataset(self.test_data_path, self.test_dict_path)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=slr_collate_fn)

        for batch in dataloader:
            padded_kpts, padded_labels, in_lens, tgt_lens = batch

            # Kiểm tra shape của keypoints sau padding [Batch=2, T_max=15, 77, 3]
            self.assertEqual(padded_kpts.shape, (2, 15, 77, 3))
            
            # Kiểm tra độ dài đầu vào thực tế
            self.assertTrue(torch.equal(in_lens, torch.tensor([10, 15])))
            
            # Kiểm tra độ dài nhãn thực tế
            self.assertTrue(torch.equal(tgt_lens, torch.tensor([3, 3])))

            # Kiểm tra giá trị padding trong keypoints (mẫu 1 từ frame 10-14 phải bằng 0)
            self.assertEqual(padded_kpts[0, 10:, :, :].sum(), 0)
            
            # Kiểm tra giá trị padding trong labels (phải bằng ID của <pad> là 1)
            # Vì cả 2 mẫu đều dài 3 nên không có padding label trong batch này, 
            # nhưng ta kiểm tra format tổng quát.
            self.assertEqual(padded_labels.shape[0], 2)

        print("[v] Hàm collate_fn padding và tính toán độ dài sequence chính xác.")

if __name__ == '__main__':
    unittest.main()
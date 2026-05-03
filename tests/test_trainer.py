import unittest
import torch
import yaml
from torch.utils.data import DataLoader
from src.datasets.slr_dataset import SLRDataset, slr_collate_fn
from src.models.light_mska import LightMSKA
from src.utils.augmentation import Augmentor
from src.trainer import SLRTrainer
import os
import pickle

class TestSLRTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tạo dữ liệu giả lập để test trainer
        cls.config_path = "configs/base_config.yaml"
        cls.dict_path = "tests/mock_dict.pkl"
        cls.data_path = "tests/mock_data.pkl"
        
        # 1. Mock Dictionary
        mock_dict = {'<s>': 0, '<pad>': 1, 'word1': 2}
        with open(cls.dict_path, 'wb') as f:
            pickle.dump(mock_dict, f)

        # 2. Mock Data: 4 mẫu để tạo batch [Batch=2]
        mock_data = {}
        for i in range(4):
            mock_data[f's{i}'] = {
                'keypoint': torch.randn(64, 77, 3), # 64 frames
                'gloss': 'word1'
            }
        with open(cls.data_path, 'wb') as f:
            pickle.dump(mock_data, f)

        cls.config = {
            'device': 'cpu',
            'lr': 1e-4,
            'epochs': 1
        }

    def test_train_step(self):
        # Khởi tạo các thành phần
        dataset = SLRDataset(self.data_path, self.dict_path)
        loader = DataLoader(dataset, batch_size=2, collate_fn=slr_collate_fn)
        
        model = LightMSKA(self.config_path, num_classes=3)
        augmentor = Augmentor()
        
        trainer = SLRTrainer(model, loader, loader, augmentor, self.config)
        
        # Chạy 1 epoch train
        avg_loss = trainer.train_epoch(1)
        
        self.assertGreater(avg_loss, 0)
        print(f"[v] Huấn luyện 1 epoch thành công. Loss trung bình: {avg_loss:.4f}")

    @classmethod
    def tearDownClass(cls):
        # Dọn dẹp file rác
        for p in [cls.dict_path, cls.data_path]:
            if os.path.exists(p): os.remove(p)

if __name__ == '__main__':
    unittest.main()
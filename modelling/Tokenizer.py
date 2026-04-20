import pickle

class GlossTokenizer:
    def __init__(self, dict_path):
        """
        dict_path: Đường dẫn tới file gloss2ids.pkl
        """
        with open(dict_path, 'rb') as f:
            self.gloss2id = pickle.load(f)
        
        # Tạo từ điển ngược để giải mã (số -> chữ) khi cần xem kết quả
        self.id2gloss = {v: k for k, v in self.gloss2id.items()}
        
        self.pad_id = self.gloss2id.get('<pad>', 1)
        self.unk_id = self.gloss2id.get('<unk>', 3)
        self.vocab_size = len(self.gloss2id)
        self.blank_id = 0
        if 0 not in self.id2gloss:
            print("CẢNH BÁO: ID 0 chưa có trong từ điển. Đang thiết lập ID 0 là <blank>")
            self.id2gloss[0] = '<blank>'

    def encode(self, gloss_string):
        """Biến chuỗi Gloss thành danh sách ID số"""
        tokens = []
        for word in gloss_string.split():
            # Nếu từ không có trong từ điển thì dùng ID của <unk>
            tokens.append(self.gloss2id.get(word, self.unk_id))
        return tokens

    def decode(self, id_list):
        """Biến danh sách ID số thành chuỗi chữ (Dùng để xem mô hình dự đoán gì)"""
        words = []
        for i in id_list:
            if i in [self.blank_id, self.gloss2id.get('<s>'), self.gloss2id.get('</s>'), self.pad_id]:
                continue
            words.append(self.id2gloss.get(i, '<unk>'))
        return " ".join(words)
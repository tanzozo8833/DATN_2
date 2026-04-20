import pickle

file_path = './gloss2ids.pkl'

with open(file_path, 'rb') as f:
    gloss_dict = pickle.load(f)

print(f"Tổng số từ khóa (gloss): {len(gloss_dict)}")
# Xem 5 mục đầu tiên trong từ điển
for i, (k, v) in enumerate(gloss_dict.items()):
    if i < 5:
        print(f"Gloss: {k} -> ID: {v}")
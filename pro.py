import pickle
with open('data/processed/gloss2ids.pkl', 'rb') as f:
    d = pickle.load(f)
    print(f"Ví dụ từ điển: {list(d.keys())[:10]}") # Xem nó là chữ hoa hay thường

with open('data/processed/train_77.pkl', 'rb') as f:
    raw = pickle.load(f)
    first_key = list(raw.keys())[0]
    print(f"Ví dụ nhãn gốc: '{raw[first_key]['gloss']}'")
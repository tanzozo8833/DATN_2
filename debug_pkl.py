"""Debug nhanh: in cấu trúc gloss2ids + 1 sample test_77.pkl + train_77.pkl."""
import pickle
import yaml

with open('configs/config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

# 1. gloss2ids
with open(cfg['data']['dict_path'], 'rb') as f:
    gloss2ids = pickle.load(f)

print('=' * 60)
print('GLOSS2IDS')
print('=' * 60)
print(f'  type     : {type(gloss2ids)}')
print(f'  size     : {len(gloss2ids)}')
items = list(gloss2ids.items())[:10]
print(f'  first 10 : {items}')
key_types = set(type(k).__name__ for k in list(gloss2ids.keys())[:50])
val_types = set(type(v).__name__ for v in list(gloss2ids.values())[:50])
print(f'  key types: {key_types}')
print(f'  val types: {val_types}')

# 2. test pkl
print('\n' + '=' * 60)
print('TEST_77.PKL')
print('=' * 60)
with open(cfg['data']['test_path'], 'rb') as f:
    test_raw = pickle.load(f)
print(f'  type     : {type(test_raw)}')
print(f'  size     : {len(test_raw)}')
sample_key = list(test_raw.keys())[0]
sample = test_raw[sample_key]
print(f'  sample key   : {sample_key!r}')
print(f'  sample type  : {type(sample)}')
print(f'  sample.keys(): {list(sample.keys()) if hasattr(sample, "keys") else "N/A"}')
gloss = sample['gloss']
print(f'  gloss type   : {type(gloss)}')
print(f'  gloss value  : {gloss!r}')
if hasattr(gloss, '__iter__') and not isinstance(gloss, str):
    items = list(gloss)
    if items:
        print(f'  gloss item[0]: {items[0]!r} (type={type(items[0]).__name__})')

# 3. train pkl (so sánh)
print('\n' + '=' * 60)
print('TRAIN_77.PKL (để so sánh)')
print('=' * 60)
with open(cfg['data']['train_path'], 'rb') as f:
    train_raw = pickle.load(f)
sample_key = list(train_raw.keys())[0]
sample = train_raw[sample_key]
gloss = sample['gloss']
print(f'  gloss type   : {type(gloss)}')
print(f'  gloss value  : {gloss!r}')
if hasattr(gloss, '__iter__') and not isinstance(gloss, str):
    items = list(gloss)
    if items:
        print(f'  gloss item[0]: {items[0]!r} (type={type(items[0]).__name__})')

# 4. Test xem các gloss text trong test có trong gloss2ids không
print('\n' + '=' * 60)
print('VOCAB COVERAGE CHECK trên test')
print('=' * 60)
all_test_tokens = []
for s in test_raw.values():
    g = s['gloss']
    if isinstance(g, str):
        all_test_tokens.extend(g.split())
    else:
        all_test_tokens.extend(g)

if all_test_tokens and isinstance(all_test_tokens[0], str):
    in_vocab = sum(1 for t in all_test_tokens if t in gloss2ids)
    print(f'  total tokens : {len(all_test_tokens)}')
    print(f'  in gloss2ids : {in_vocab} ({in_vocab/len(all_test_tokens)*100:.1f}%)')
    missing = [t for t in all_test_tokens[:20] if t not in gloss2ids]
    print(f'  first missing (in test order, top 20): {missing}')
elif all_test_tokens:
    print(f'  test gloss đã là int: max={max(all_test_tokens)}, min={min(all_test_tokens)}')

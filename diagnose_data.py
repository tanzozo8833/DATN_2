from collections import Counter
import pickle
import torch
import yaml
from tqdm import tqdm

def diagnose_data():
    with open("configs/base_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    with open("data/processed/gloss2ids.pkl", 'rb') as f:
        gloss_dict = pickle.load(f)

    print(f"Dictionary size: {len(gloss_dict)}")
    print(f"Sample entries: {list(gloss_dict.items())[:10]}")

    ids = sorted([v for k, v in gloss_dict.items()])
    print(f"ID range: {min(ids)} to {max(ids)}")

    with open("data/processed/train_77.pkl", 'rb') as f:
        train_data = pickle.load(f)

    print(f"\nTrain samples: {len(train_data)}")

    for i, (key, sample) in enumerate(list(train_data.items())[:5]):
        print(f"\n--- Sample {i+1}: {key} ---")
        print(f"Keypoint shape: {sample['keypoint'].shape}")
        print(f"Gloss type: {type(sample['gloss'])}")
        print(f"Gloss value: {sample['gloss']}")

        if isinstance(sample['gloss'], str):
            print("❌ ERROR: Gloss is still a STRING!")
        elif isinstance(sample['gloss'], list):
            print(f"✓ Gloss is a list of IDs: {sample['gloss'][:10]}...")
        else:
            print(f"❌ ERROR: Unknown gloss format: {type(sample['gloss'])}")

    with open("data/processed/dev_77.pkl", 'rb') as f:
        dev_data = pickle.load(f)

    print(f"\nDev samples: {len(dev_data)}")

    label_counter = Counter()

    for key, sample in tqdm(dev_data.items(), desc="Counting labels"):
        if isinstance(sample['gloss'], list):
            for label_id in sample['gloss']:
                label_counter[label_id] += 1

    print(f"\nTop 20 most frequent labels in dev set:")
    for label_id, count in label_counter.most_common(20):
        gloss = [k for k, v in gloss_dict.items() if v == label_id]
        gloss_name = gloss[0] if gloss else f"ID:{label_id}"
        print(f"  {gloss_name} (ID:{label_id}): {count} times")

    num_classes = len(gloss_dict)
    print(f"\nModel num_classes should be: {num_classes}")

if __name__ == "__main__":
    diagnose_data()

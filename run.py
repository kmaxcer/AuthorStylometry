import json
import numpy as np
from pathlib import Path
from collections import Counter

base_dir = Path(__file__).parent
datasets_dir = base_dir / 'data' / 'datasets'

ml_dir = datasets_dir / 'ml'
y_train = np.load(ml_dir / 'y.npy')
y_test = np.load(ml_dir / 'y_test.npy')

print("Train:", Counter(y_train))
print("Test:", Counter(y_test))

with open(datasets_dir / 'author_labels.json', 'r') as f:
    labels = json.load(f)

print("\nАвторы:")
for label, name in labels['label_to_author'].items():
    train_count = sum(1 for y in y_train if y == int(label))
    test_count = sum(1 for y in y_test if y == int(label))
    print(f"  {name}: train={train_count}, test={test_count}")
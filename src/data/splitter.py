"""
Разделение данных по произведениям
"""

from typing import List, Tuple


def split_by_works(file_ids: List[str], authors: List[str], test_ratio: float = 0.2) -> Tuple[List[int], List[int]]:
    author_files = {}
    for file_id, author in zip(file_ids, authors):
        if author not in author_files:
            author_files[author] = []
        if file_id not in author_files[author]:
            author_files[author].append(file_id)

    train_files = []
    test_files = []

    for author, files in author_files.items():
        n_files = len(files)
        if n_files == 1:
            train_files.extend(files)
        else:
            n_test = max(1, int(n_files * test_ratio))
            test_files.extend(files[:n_test])
            train_files.extend(files[n_test:])

    train_set = set(train_files)
    train_indices = []
    test_indices = []

    for idx, file_id in enumerate(file_ids):
        if file_id in train_set:
            train_indices.append(idx)
        else:
            test_indices.append(idx)

    return train_indices, test_indices
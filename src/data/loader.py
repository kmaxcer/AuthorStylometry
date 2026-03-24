"""
Загрузка окон и метаданных
"""

import json
from pathlib import Path
from typing import List, Tuple


def load_all_windows(windows_dir: Path) -> Tuple[List[str], List[str], List[str]]:
    texts = []
    authors = []
    file_ids = []

    for corpus_name in ['corpus_19_1', 'corpus_19_2']:
        metadata_path = windows_dir / corpus_name / 'windows_metadata.json'
        if not metadata_path.exists():
            continue

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        windows_folder = windows_dir / corpus_name

        for window_info in metadata['windows']:
            window_file = windows_folder / window_info['window_file']
            if not window_file.exists():
                continue

            with open(window_file, 'r', encoding='utf-8') as f:
                texts.append(f.read())

            authors.append(window_info['author'])
            file_ids.append(window_info['file_id'])

    return texts, authors, file_ids
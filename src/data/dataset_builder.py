"""
Сборка датасетов из текстовых окон
"""

import json
import logging
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict

import numpy as np

from loader import load_all_windows
from splitter import split_by_works
from src.features.base_features import BaseFeatureExtractor
from src.features.ngram_features import NgramExtractor
from src.features.pos_features import PosFeatureExtractor

logger = logging.getLogger(__name__)


class DatasetBuilder:
    def __init__(self, window_size: int = 1000, ngram_n: int = 3, ngram_top_k: int = 500,
                 nn_vocab_size: int = 10000, use_pos: bool = True):
        self.window_size = window_size
        self.nn_vocab_size = nn_vocab_size
        self.use_pos = use_pos
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.windows_dir = self.base_dir / 'data' / 'windows'
        self.datasets_dir = self.base_dir / 'data' / 'datasets'

        self.base_extractor = BaseFeatureExtractor()
        self.ngram_extractor = NgramExtractor(n=ngram_n, top_k=ngram_top_k)
        self.pos_extractor = PosFeatureExtractor() if use_pos else None

    def _get_author_labels(self, authors: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
        unique = sorted(set(authors))
        author_to_label = {a: i for i, a in enumerate(unique)}
        label_to_author = {i: a for a, i in author_to_label.items()}
        logger.info("Найдено авторов: %d (%s)", len(unique), ', '.join(unique))
        return author_to_label, label_to_author

    def _build_feature_names(self) -> List[str]:
        base_names = self.base_extractor.get_feature_names()
        ngram_names = [f'ngram_{ng}' for ng in self.ngram_extractor.vocab]
        names = base_names + ngram_names
        if self.pos_extractor:
            names += self.pos_extractor.get_feature_names()
        return names

    def _extract_ml_features(self, text: str) -> np.ndarray:
        base = self.base_extractor.extract(text)
        ngrams = self.ngram_extractor.extract(text)
        parts = [base, ngrams]
        if self.pos_extractor:
            parts.append(self.pos_extractor.extract(text))
        return np.concatenate([np.array(p, dtype=np.float32) for p in parts])

    def _encode_nn_sequences(self, texts: List[str], char_to_idx: Dict[str, int]) -> np.ndarray:
        sequences = []
        for text in texts:
            seq = [char_to_idx.get(c, 1) for c in text[:self.window_size]]
            pad_len = self.window_size - len(seq)
            if pad_len > 0:
                seq.extend([0] * pad_len)
            sequences.append(seq)
        return np.array(sequences, dtype=np.int32)

    def _build_char_vocab(self, texts: List[str]) -> Dict[str, int]:
        char_counts = Counter()
        for text in texts:
            char_counts.update(text)

        most_common = char_counts.most_common(self.nn_vocab_size - 2)
        char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for char, _ in most_common:
            char_to_idx[char] = len(char_to_idx)

        logger.info("Символьный словарь: %d символов", len(char_to_idx))
        return char_to_idx

    def _save_ml_dataset(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         feature_names: List[str]):
        ml_dir = self.datasets_dir / 'ml'
        ml_dir.mkdir(parents=True, exist_ok=True)

        np.save(ml_dir / 'X.npy', X_train)
        np.save(ml_dir / 'y.npy', y_train)
        np.save(ml_dir / 'X_test.npy', X_test)
        np.save(ml_dir / 'y_test.npy', y_test)

        with open(ml_dir / 'feature_names.json', 'w', encoding='utf-8') as f:
            json.dump(feature_names, f, ensure_ascii=False, indent=2)

        logger.info("ML датасет: train %s, test %s, признаков %d",
                     X_train.shape, X_test.shape, X_train.shape[1])

    def _save_nn_dataset(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         char_to_idx: Dict[str, int]):
        nn_dir = self.datasets_dir / 'nn'
        nn_dir.mkdir(parents=True, exist_ok=True)

        np.save(nn_dir / 'X_sequences.npy', X_train)
        np.save(nn_dir / 'y_labels.npy', y_train)
        np.save(nn_dir / 'X_test_sequences.npy', X_test)
        np.save(nn_dir / 'y_test_labels.npy', y_test)

        with open(nn_dir / 'char_to_idx.json', 'w', encoding='utf-8') as f:
            json.dump(char_to_idx, f, ensure_ascii=False, indent=2)

        logger.info("NN датасет: train %s, test %s, словарь %d",
                     X_train.shape, X_test.shape, len(char_to_idx))

    def _save_labels(self, author_to_label: Dict[str, int], label_to_author: Dict[int, str]):
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        path = self.datasets_dir / 'author_labels.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'author_to_label': author_to_label,
                'label_to_author': {str(k): v for k, v in label_to_author.items()},
                'n_classes': len(author_to_label),
            }, f, ensure_ascii=False, indent=2)

    def run(self, test_ratio: float = 0.2) -> bool:
        texts, authors, file_ids = load_all_windows(self.windows_dir)
        if not texts:
            logger.error("Окна не найдены в %s", self.windows_dir)
            return False

        author_to_label, label_to_author = self._get_author_labels(authors)
        labels = np.array([author_to_label[a] for a in authors])

        train_idx, test_idx = split_by_works(file_ids, authors, test_ratio)

        train_texts = [texts[i] for i in train_idx]
        train_labels = labels[train_idx]
        test_texts = [texts[i] for i in test_idx]
        test_labels = labels[test_idx]

        logger.info("Train: %d окон, Test: %d окон", len(train_texts), len(test_texts))

        self._save_labels(author_to_label, label_to_author)

        self.ngram_extractor.build_vocab(train_texts)

        if self.pos_extractor:
            logger.info("Извлечение POS-признаков (pymorphy2)...")

        X_train_ml = np.array([self._extract_ml_features(t) for t in train_texts])
        X_test_ml = np.array([self._extract_ml_features(t) for t in test_texts])
        feature_names = self._build_feature_names()

        if X_train_ml.shape[1] != len(feature_names):
            logger.warning("Несовпадение: признаков %d, имён %d.",
                           X_train_ml.shape[1], len(feature_names))

        self._save_ml_dataset(X_train_ml, train_labels, X_test_ml, test_labels, feature_names)

        char_to_idx = self._build_char_vocab(train_texts)
        X_train_nn = self._encode_nn_sequences(train_texts, char_to_idx)
        X_test_nn = self._encode_nn_sequences(test_texts, char_to_idx)

        self._save_nn_dataset(X_train_nn, train_labels, X_test_nn, test_labels, char_to_idx)

        logger.info("Сборка датасетов завершена. Директория: %s", self.datasets_dir)
        return True


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    builder = DatasetBuilder(window_size=1000)
    success = builder.run(test_ratio=0.2)
    if not success:
        raise SystemExit("Ошибка сборки датасетов")


if __name__ == "__main__":
    main()

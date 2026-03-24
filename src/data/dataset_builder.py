"""
Сборка датасетов из текстовых окон
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter

from src.data.loader import load_all_windows
from src.data.splitter import split_by_works
from src.features.base_features import BaseFeatureExtractor
from src.features.ngram_features import NgramExtractor


class DatasetBuilder:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.windows_dir = self.base_dir / 'data' / 'windows'
        self.datasets_dir = self.base_dir / 'data' / 'datasets'
        self.author_to_label = {}
        self.label_to_author = {}

        self.base_extractor = BaseFeatureExtractor()
        self.ngram_extractor = NgramExtractor(n=3, top_k=500)

    def _get_author_labels(self, authors: List[str]):
        unique_authors = sorted(set(authors))
        self.author_to_label = {author: idx for idx, author in enumerate(unique_authors)}
        self.label_to_author = {idx: author for author, idx in self.author_to_label.items()}

    def _build_ngram_vocab(self, texts: List[str]):
        self.ngram_extractor.build_vocab(texts)

    def _extract_ml_features(self, text: str) -> List[float]:
        base = self.base_extractor.extract(text)
        ngrams = self.ngram_extractor.extract(text)
        return base + ngrams

    def _create_ml_dataset(self, texts: List[str], labels: List[int], output_dir: Path):
        X = [self._extract_ml_features(t) for t in texts]
        X = np.array(X)
        y = np.array(labels)

        ml_dir = output_dir / 'ml'
        ml_dir.mkdir(parents=True, exist_ok=True)

        np.save(ml_dir / 'X.npy', X)
        np.save(ml_dir / 'y.npy', y)

        feature_names = [
            'word_count', 'char_count', 'sentence_count', 'avg_word_length', 'avg_sentence_length',
            'type_token_ratio', 'hapax_ratio',
            'punct_dot', 'punct_comma', 'punct_exclam', 'punct_quest', 'punct_colon', 'punct_semicolon',
            'punct_hyphen', 'punct_dash', 'punct_ellipsis', 'punct_quote', 'punct_lparen', 'punct_rparen',
            'punct_lfquote', 'punct_rfquote', 'punct_apostrophe', 'punct_density',
            'stopword_and', 'stopword_v', 'stopword_ne', 'stopword_na', 'stopword_ya',
            'stopword_chto', 'stopword_on', 'stopword_s', 'stopword_a', 'stopword_k',
            'freq_o', 'freq_e', 'freq_a', 'freq_i', 'freq_n', 'freq_t', 'freq_s', 'freq_r', 'freq_v', 'freq_l',
            'upper_ratio', 'digit_ratio', 'dialogue_dashes', 'word_length_std', 'punct_diversity'
        ]
        for i in range(len(self.ngram_extractor.vocab)):
            feature_names.append(f'ngram_{self.ngram_extractor.vocab[i]}')

        with open(ml_dir / 'feature_names.json', 'w', encoding='utf-8') as f:
            json.dump(feature_names, f, ensure_ascii=False, indent=2)

    def _create_nn_dataset(self, texts: List[str], labels: List[int], output_dir: Path, vocab_size: int = 10000):
        char_counts = Counter()
        for text in texts:
            char_counts.update(text)

        most_common = char_counts.most_common(vocab_size - 2)
        char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for char, _ in most_common:
            char_to_idx[char] = len(char_to_idx)

        sequences = []
        for text in texts:
            seq = [char_to_idx.get(c, 1) for c in text[:self.window_size]]
            if len(seq) < self.window_size:
                seq.extend([0] * (self.window_size - len(seq)))
            sequences.append(seq)

        X = np.array(sequences, dtype=np.int32)
        y = np.array(labels)

        nn_dir = output_dir / 'nn'
        nn_dir.mkdir(parents=True, exist_ok=True)

        np.save(nn_dir / 'X_sequences.npy', X)
        np.save(nn_dir / 'y_labels.npy', y)

        with open(nn_dir / 'char_to_idx.json', 'w', encoding='utf-8') as f:
            json.dump(char_to_idx, f, ensure_ascii=False, indent=2)

    def run(self, test_ratio: float = 0.2):
        texts, authors, file_ids = load_all_windows(self.windows_dir)
        if not texts:
            return False

        self._get_author_labels(authors)

        train_idx, test_idx = split_by_works(file_ids, authors, test_ratio)

        train_texts = [texts[i] for i in train_idx]
        train_authors = [authors[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]

        train_labels = [self.author_to_label[a] for a in train_authors]
        test_labels = [self.author_to_label[a] for a in test_authors]

        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        with open(self.datasets_dir / 'author_labels.json', 'w', encoding='utf-8') as f:
            json.dump({
                'author_to_label': self.author_to_label,
                'label_to_author': self.label_to_author,
                'n_classes': len(self.author_to_label)
            }, f, ensure_ascii=False, indent=2)

        self._build_ngram_vocab(train_texts)

        self._create_ml_dataset(train_texts, train_labels, self.datasets_dir)

        ml_dir = self.datasets_dir / 'ml'
        X_test = [self._extract_ml_features(t) for t in test_texts]
        np.save(ml_dir / 'X_test.npy', np.array(X_test))
        np.save(ml_dir / 'y_test.npy', np.array(test_labels))

        self._create_nn_dataset(train_texts, train_labels, self.datasets_dir)

        nn_dir = self.datasets_dir / 'nn'
        with open(nn_dir / 'char_to_idx.json', 'r', encoding='utf-8') as f:
            char_to_idx = json.load(f)

        test_sequences = []
        for text in test_texts:
            seq = [char_to_idx.get(c, 1) for c in text[:self.window_size]]
            if len(seq) < self.window_size:
                seq.extend([0] * (self.window_size - len(seq)))
            test_sequences.append(seq)

        np.save(nn_dir / 'X_test_sequences.npy', np.array(test_sequences))
        np.save(nn_dir / 'y_test_labels.npy', np.array(test_labels))

        return True


def main():
    builder = DatasetBuilder(window_size=1000)
    builder.run(test_ratio=0.2)


if __name__ == "__main__":
    main()
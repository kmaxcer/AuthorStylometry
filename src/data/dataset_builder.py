"""
Модуль для сборки датасетов из текстовых окон
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple
from collections import Counter


class DatasetBuilder:

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.windows_dir = self.base_dir / 'data' / 'windows'
        self.datasets_dir = self.base_dir / 'data' / 'datasets'
        self.author_to_label = {}
        self.label_to_author = {}

    def load_all_windows(self) -> Tuple[List[str], List[str], List[str]]:
        texts = []
        authors = []
        file_ids = []

        for corpus_name in ['corpus_19_1', 'corpus_19_2']:
            metadata_path = self.windows_dir / corpus_name / 'windows_metadata.json'

            if not metadata_path.exists():
                continue

            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            windows_folder = self.windows_dir / corpus_name

            for window_info in metadata['windows']:
                window_file = windows_folder / window_info['window_file']

                if not window_file.exists():
                    continue

                with open(window_file, 'r', encoding='utf-8') as f:
                    text = f.read()

                texts.append(text)
                authors.append(window_info['author'])
                file_ids.append(window_info['file_id'])

        unique_authors = sorted(set(authors))
        self.author_to_label = {author: idx for idx, author in enumerate(unique_authors)}
        self.label_to_author = {idx: author for author, idx in self.author_to_label.items()}

        return texts, authors, file_ids

    def split_by_works(self, file_ids: List[str], authors: List[str],
                       test_ratio: float = 0.2) -> Tuple[List[int], List[int]]:
        author_files = {}
        for file_id, author in zip(file_ids, authors):
            if author not in author_files:
                author_files[author] = []
            if file_id not in author_files[author]:
                author_files[author].append(file_id)

        train_files = []
        test_files = []

        for author, files in author_files.items():
            n_test = max(1, int(len(files) * test_ratio))
            test_files.extend(files[:n_test])
            train_files.extend(files[n_test:])

        train_indices = []
        test_indices = []

        for idx, file_id in enumerate(file_ids):
            if file_id in train_files:
                train_indices.append(idx)
            else:
                test_indices.append(idx)

        return train_indices, test_indices

    def extract_ml_features(self, text: str) -> List[float]:
        features = []

        words = text.split()
        chars = len(text)
        sentences = [s for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]

        n_words = len(words)
        n_chars = chars
        n_sentences = len(sentences) if sentences else 1

        features.append(n_words)
        features.append(n_chars)
        features.append(n_sentences)
        features.append(n_chars / n_words if n_words > 0 else 0)
        features.append(n_words / n_sentences)

        unique_words = set(w.lower() for w in words)
        features.append(len(unique_words) / n_words if n_words > 0 else 0)

        word_counts = Counter(w.lower() for w in words)
        hapax = sum(1 for c in word_counts.values() if c == 1)
        features.append(hapax / n_words if n_words > 0 else 0)

        punct_marks = ['.', ',', '!', '?', ':', ';', '-', '—', '…', '"', '(', ')', '«', '»', "'"]
        total_punct = 0

        for mark in punct_marks:
            count = text.count(mark)
            features.append(count)
            total_punct += count

        features.append(total_punct / n_chars if n_chars > 0 else 0)

        stop_words = ['и', 'в', 'не', 'на', 'я', 'что', 'он', 'с', 'а', 'к']
        word_freq = Counter(w.lower() for w in words)

        for sw in stop_words:
            features.append(word_freq.get(sw, 0) / n_words if n_words > 0 else 0)

        russian_letters = ['о', 'е', 'а', 'и', 'н', 'т', 'с', 'р', 'в', 'л']
        text_lower = text.lower()

        for letter in russian_letters:
            features.append(text_lower.count(letter) / n_chars if n_chars > 0 else 0)

        upper_count = sum(1 for c in text if c.isupper())
        features.append(upper_count / n_chars if n_chars > 0 else 0)

        digit_count = sum(1 for c in text if c.isdigit())
        features.append(digit_count / n_chars if n_chars > 0 else 0)

        dialogue_dashes = len(re.findall(r'^-|^—', text, re.MULTILINE))
        features.append(dialogue_dashes)

        word_lengths = [len(w) for w in words]
        features.append(np.std(word_lengths) if word_lengths else 0)

        punct_set = set(c for c in text if c in punct_marks)
        features.append(len(punct_set) / len(punct_marks))

        return features

    def create_ml_dataset(self, texts: List[str], labels: List[int], output_dir: Path):
        X = []
        for text in texts:
            X.append(self.extract_ml_features(text))

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

        with open(ml_dir / 'feature_names.json', 'w', encoding='utf-8') as f:
            json.dump(feature_names, f, ensure_ascii=False, indent=2)

    def create_nn_dataset(self, texts: List[str], labels: List[int],
                          output_dir: Path, vocab_size: int = 10000, max_len: int = None):
        if max_len is None:
            max_len = self.window_size

        char_counts = Counter()
        for text in texts:
            char_counts.update(text)

        most_common = char_counts.most_common(vocab_size - 2)
        char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for char, _ in most_common:
            char_to_idx[char] = len(char_to_idx)

        sequences = []
        for text in texts:
            seq = [char_to_idx.get(c, 1) for c in text[:max_len]]
            if len(seq) < max_len:
                seq.extend([0] * (max_len - len(seq)))
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
        texts, authors, file_ids = self.load_all_windows()

        if not texts:
            return False

        train_idx, test_idx = self.split_by_works(file_ids, authors, test_ratio)

        train_texts = [texts[i] for i in train_idx]
        train_authors = [authors[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        test_authors = [authors[i] for i in test_idx]

        train_labels = [self.author_to_label[a] for a in train_authors]
        test_labels = [self.author_to_label[a] for a in test_authors]

        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        with open(self.datasets_dir / 'author_labels.json', 'w', encoding='utf-8') as f:
            json.dump({
                'author_to_label': self.author_to_label,
                'label_to_author': self.label_to_author,
                'n_classes': len(self.author_to_label)
            }, f, ensure_ascii=False, indent=2)

        self.create_ml_dataset(train_texts, train_labels, self.datasets_dir)

        ml_dir = self.datasets_dir / 'ml'
        X_test = np.array([self.extract_ml_features(t) for t in test_texts])
        np.save(ml_dir / 'X_test.npy', X_test)
        np.save(ml_dir / 'y_test.npy', np.array(test_labels))

        self.create_nn_dataset(train_texts, train_labels, self.datasets_dir)

        nn_dir = self.datasets_dir / 'nn'

        with open(nn_dir / 'char_to_idx.json', 'r', encoding='utf-8') as f:
            char_to_idx = json.load(f)

        test_sequences = []
        max_len = self.window_size
        for text in test_texts:
            seq = [char_to_idx.get(c, 1) for c in text[:max_len]]
            if len(seq) < max_len:
                seq.extend([0] * (max_len - len(seq)))
            test_sequences.append(seq)

        np.save(nn_dir / 'X_test_sequences.npy', np.array(test_sequences))
        np.save(nn_dir / 'y_test_labels.npy', np.array(test_labels))

        return True


def main():
    builder = DatasetBuilder(window_size=1000)
    builder.run(test_ratio=0.2)


if __name__ == "__main__":
    main()
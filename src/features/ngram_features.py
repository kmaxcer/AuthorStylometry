"""
Модуль для работы с символьными n-граммами
"""

from collections import Counter
from typing import List


class NgramExtractor:
    def __init__(self, n: int = 3, top_k: int = 500):
        self.n = n
        self.top_k = top_k
        self.vocab = []

    def build_vocab(self, texts: List[str]):
        all_ngrams = []
        for text in texts:
            for i in range(len(text) - self.n + 1):
                all_ngrams.append(text[i:i + self.n])

        counter = Counter(all_ngrams)
        self.vocab = [ngram for ngram, _ in counter.most_common(self.top_k)]

    def extract(self, text: str) -> List[float]:
        features = []
        total = len(text) - self.n + 1
        if total <= 0 or not self.vocab:
            return [0.0] * len(self.vocab)

        ngram_counts = Counter()
        for i in range(total):
            ngram_counts[text[i:i + self.n]] += 1

        for ngram in self.vocab:
            features.append(ngram_counts.get(ngram, 0) / total)

        return features
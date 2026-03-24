"""
Базовые признаки для стилометрии (48 штук)
"""

import re
import numpy as np
from collections import Counter
from typing import List


class BaseFeatureExtractor:
    def extract(self, text: str) -> List[float]:
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
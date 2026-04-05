"""
Базовые признаки для стилометрии (65 штук)
"""

import re
import math
import numpy as np
from collections import Counter
from typing import List


class BaseFeatureExtractor:
    def extract(self, text: str) -> List[float]:
        if not text:
            return [0.0] * 65

        features = []

        words = text.split()
        chars = len(text)
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]

        n_words = len(words)
        n_chars = chars
        n_sentences = len(sentences) if sentences else 1

        # --- Статистика текста ---
        features.append(n_words)
        features.append(n_chars)
        features.append(n_sentences)
        features.append(n_chars / n_words if n_words > 0 else 0)
        features.append(n_words / n_sentences)
        features.append(n_chars / n_sentences)

        # --- Лексическое разнообразие ---
        unique_words = set(w.lower() for w in words)
        features.append(len(unique_words) / n_words if n_words > 0 else 0)

        word_counts = Counter(w.lower() for w in words)
        hapax = sum(1 for c in word_counts.values() if c == 1)
        features.append(hapax / n_words if n_words > 0 else 0)

        # --- Индекс Томашевского (степень лексического разнообразия) ---
        features.append(len(unique_words) / math.sqrt(2 * n_words) if n_words > 0 else 0)

        # --- Энтропия частот слов ---
        features.append(self._entropy(word_counts, n_words))

        # --- Энтропия частот букв ---
        char_counts = Counter(text.lower())
        features.append(self._entropy(char_counts, n_chars))

        # --- Энтропия биграмм ---
        bigrams = [text[i:i+2] for i in range(len(text) - 1)]
        bigram_counts = Counter(bigrams)
        features.append(self._entropy(bigram_counts, len(bigrams)))

        # --- Коэффициент Юла (K) ---
        features.append(self._yule_k(word_counts, n_words))

        # --- Длины слов ---
        word_lengths = [len(w) for w in words]
        features.append(np.mean(word_lengths) if word_lengths else 0)
        features.append(np.std(word_lengths) if word_lengths else 0)
        features.append(np.median(word_lengths) if word_lengths else 0)
        features.append(sum(1 for l in word_lengths if l <= 3) / n_words if n_words > 0 else 0)
        features.append(sum(1 for l in word_lengths if l >= 8) / n_words if n_words > 0 else 0)
        features.append(max(word_lengths) if word_lengths else 0)

        # --- Пунктуация ---
        punct_marks = ['.', ',', '!', '?', ':', ';', '-', '—', '…', '"', '(', ')', '«', '»', "'"]
        total_punct = 0
        for mark in punct_marks:
            count = text.count(mark)
            features.append(count)
            total_punct += count
        features.append(total_punct / n_chars if n_chars > 0 else 0)

        # --- Стоп-слова (расширенный список) ---
        stop_words = [
            'и', 'в', 'не', 'на', 'я', 'что', 'он', 'с', 'а', 'к',
            'но', 'как', 'это', 'все', 'его', 'так', 'же', 'за', 'по',
            'от', 'из', 'у', 'до', 'для', 'о', 'об', 'при', 'со',
            'они', 'мы', 'вы', 'она', 'оно', 'бы', 'же', 'ли', 'было',
            'быть', 'был', 'была', 'были', 'будет', 'будут',
        ]
        word_freq = Counter(w.lower() for w in words)
        for sw in stop_words:
            features.append(word_freq.get(sw, 0) / n_words if n_words > 0 else 0)

        # --- Частоты букв ---
        russian_letters = ['о', 'е', 'а', 'и', 'н', 'т', 'с', 'р', 'в', 'л']
        text_lower = text.lower()
        for letter in russian_letters:
            features.append(text_lower.count(letter) / n_chars if n_chars > 0 else 0)

        # --- Стилистические ---
        upper_count = sum(1 for c in text if c.isupper())
        features.append(upper_count / n_chars if n_chars > 0 else 0)

        digit_count = sum(1 for c in text if c.isdigit())
        features.append(digit_count / n_chars if n_chars > 0 else 0)

        dialogue_dashes = len(re.findall(r'^-|^—', text, re.MULTILINE))
        features.append(dialogue_dashes)

        # Доля прямой речи (строки, начинающиеся с тире)
        lines = text.split('\n')
        dialogue_lines = sum(1 for l in lines if re.match(r'^[\s]*[-—]', l))
        features.append(dialogue_lines / len(lines) if lines else 0)

        punct_set = set(c for c in text if c in punct_marks)
        features.append(len(punct_set) / len(punct_marks))

        return features

    @staticmethod
    def _entropy(counter: Counter, total: int) -> float:
        if total <= 0:
            return 0.0
        ent = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                ent -= p * math.log2(p)
        return ent

    @staticmethod
    def _yule_k(counter: Counter, total: int) -> float:
        if total <= 0:
            return 0.0
        freq_of_freq = Counter(counter.values())
        M2 = sum(i * i * v for i, v in freq_of_freq.items())
        M1 = total
        return 10000 * (M2 - M1) / (M1 * M1) if M1 > 0 else 0.0

    def get_feature_names(self) -> List[str]:
        punct_names = [
            'punct_dot', 'punct_comma', 'punct_exclam', 'punct_quest',
            'punct_colon', 'punct_semicolon', 'punct_hyphen', 'punct_dash',
            'punct_ellipsis', 'punct_quote', 'punct_lparen', 'punct_rparen',
            'punct_lfquote', 'punct_rfquote', 'punct_apostrophe',
        ]
        stop_names = [
            'sw_i', 'sw_v', 'sw_ne', 'sw_na', 'sw_ya', 'sw_chto', 'sw_on',
            'sw_s', 'sw_a', 'sw_k', 'sw_no', 'sw_kak', 'sw_eto', 'sw_vse',
            'sw_ego', 'sw_tak', 'sw_zhe', 'sw_za', 'sw_po', 'sw_ot', 'sw_iz',
            'sw_u', 'sw_do', 'sw_dlya', 'sw_o', 'sw_ob', 'sw_pri', 'sw_so',
            'sw_oni', 'sw_my', 'sw_vy', 'sw_ona', 'sw_ono', 'sw_by', 'sw_zhe2',
            'sw_li', 'sw_bylo', 'sw_byt', 'sw_byl', 'sw_byla', 'sw_byli',
            'sw_budet', 'sw_budut',
        ]
        freq_names = [f'freq_{l}' for l in ['o', 'e', 'a', 'i', 'n', 't', 's', 'r', 'v', 'l']]

        return [
            'word_count', 'char_count', 'sentence_count',
            'avg_word_length', 'avg_sentence_length', 'avg_sentence_chars',
            'type_token_ratio', 'hapax_ratio',
            'tomashewski_index', 'word_entropy', 'char_entropy',
            'bigram_entropy', 'yule_k',
            'word_len_mean', 'word_len_std', 'word_len_median',
            'short_word_ratio', 'long_word_ratio', 'max_word_len',
        ] + punct_names + ['punct_density'] + stop_names + freq_names + [
            'upper_ratio', 'digit_ratio', 'dialogue_dashes',
            'dialogue_ratio', 'punct_diversity',
        ]
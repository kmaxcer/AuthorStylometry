"""
Признаки на основе морфологии: части речи, синтаксические характеристики
"""

import re
import logging
from collections import Counter
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import pymorphy3
    HAS_PYMORPHY = True
except ImportError:
    HAS_PYMORPHY = False
    logger.warning("pymorphy2 недоступен. POS-признаки будут оценены по словарям.")

POS_CATEGORIES = {
    'NOUN': 'существительное',
    'VERB': 'глагол',
    'INFN': 'инфинитив',
    'ADJF': 'прилагательное',
    'ADVB': 'наречие',
    'PREP': 'предлог',
    'CONJ': 'союз',
    'PRCL': 'частица',
    'NPRO': 'местоимение',
    'GRND': 'деепричастие',
    'PRTF': 'причастие',
    'NUMR': 'числительное',
    'INTJ': 'междометие',
}

POS_GROUPS = {
    'noun': ['NOUN'],
    'verb': ['VERB', 'INFN'],
    'adj': ['ADJF', 'ADJS'],
    'adv': ['ADVB'],
    'prep': ['PREP'],
    'conj': ['CONJ'],
    'prcl': ['PRCL'],
    'pron': ['NPRO'],
    'verb_all': ['VERB', 'INFN', 'GRND', 'PRTF', 'PRTS'],
    'partcp': ['PRTF', 'PRTS'],
    'gerund': ['GRND'],
    'num': ['NUMR'],
}

# Словари для fallback-оценки
_PREPOSITIONS = frozenset([
    'без', 'безо', 'близ', 'в', 'во', 'вместо', 'вне', 'для', 'до', 'за',
    'из', 'изо', 'из-за', 'из-под', 'к', 'ко', 'кроме', 'между', 'меж',
    'на', 'над', 'надо', 'о', 'об', 'обо', 'от', 'ото', 'перед', 'передо',
    'по', 'под', 'подо', 'после', 'пред', 'при', 'про', 'ради', 'с', 'со',
    'сквозь', 'среди', 'у', 'через', 'чрез',
])

_CONJUNCTIONS = frozenset([
    'и', 'а', 'но', 'да', 'или', 'либо', 'ни', 'хотя', 'хоть', 'что',
    'чтобы', 'как', 'когда', 'если', 'пока', 'после', 'словно', 'будто',
    'хотя', 'однако', 'зато', 'тоже', 'также', 'то есть', 'причем',
    'причём', 'ибо', 'дабы', 'ежели', 'кабы', 'коли',
])

_PARTICLES = frozenset([
    'бы', 'б', 'вот', 'вон', 'да', 'даже', 'едва', 'еще', 'ещё', 'же',
    'ведь', 'именно', 'лишь', 'мол', 'не', 'ни', 'ну', 'оно', 'поди',
    'пожалуй', 'попросту', 'просто', 'прямо', 'разве', 'только', 'угодно',
    'уж', 'чуть', 'якобы', 'дескать', 'вряд', 'авось',
])

_PRONOUNS = frozenset([
    'я', 'мы', 'ты', 'вы', 'он', 'она', 'оно', 'они', 'себя',
    'мой', 'моя', 'моё', 'мои', 'твой', 'наш', 'ваш',
    'свой', 'этот', 'эта', 'это', 'эти', 'тот', 'та', 'те',
    'кто', 'что', 'какой', 'какая', 'какое', 'какие', 'который', 'чей',
    'сам', 'сама', 'само', 'сами', 'самый', 'весь', 'вся', 'всё', 'все',
    'каждый', 'другой', 'иной', 'некоторый', 'некий', 'каков',
])

_INFINITIVE_SUFFIXES = ('ть', 'ться', 'ти', 'тись')

_PREFIXES = frozenset([
    'без', 'бес', 'вз', 'вс', 'вне', 'до', 'за', 'из', 'ис',
    'на', 'над', 'не', 'низ', 'о', 'об', 'от', 'пере', 'по',
    'под', 'пред', 'при', 'про', 'раз', 'рас', 'с', 'со', 'у',
])


class PosFeatureExtractor:
    def __init__(self):
        self._morph = None
        self._cache = {}

    @property
    def morph(self):
        if self._morph is None and HAS_PYMORPHY:
            self._morph = pymorphy3.MorphAnalyzer()
        return self._morph

    def _parse_word(self, word: str) -> Optional[str]:
        if not HAS_PYMORPHY:
            return self._estimate_pos(word)

        if word in self._cache:
            return self._cache[word]
        parsed = self.morph.parse(word)
        tag = parsed[0].tag.POS if parsed else None
        self._cache[word] = tag
        return tag

    @staticmethod
    def _estimate_pos(word: str) -> Optional[str]:
        if word in _PREPOSITIONS:
            return 'PREP'
        if word in _CONJUNCTIONS:
            return 'CONJ'
        if word in _PARTICLES:
            return 'PRCL'
        if word in _PRONOUNS:
            return 'NPRO'
        if len(word) >= 3 and word.endswith(_INFINITIVE_SUFFIXES):
            return 'INFN'
        if len(word) >= 4 and (word.endswith('ся') or word.endswith('сь')):
            if word[:-2].endswith(_INFINITIVE_SUFFIXES[:-2]):
                return 'INFN'
            return 'VERB'
        if word.endswith('ый') or word.endswith('ая') or word.endswith('ое') or word.endswith('ие'):
            return 'ADJF'
        if word.endswith('о') and len(word) > 3:
            return 'ADVB'
        if word.endswith('ие') or word.endswith('ий') or word.endswith('ть'):
            return 'VERB'
        return 'NOUN'

    def _get_tokens(self, text: str) -> List[str]:
        return re.findall(r'[а-яёА-ЯЁ]+', text.lower())

    def extract(self, text: str) -> List[float]:
        tokens = self._get_tokens(text)
        n_tokens = len(tokens)
        n_features = len(POS_CATEGORIES) + len(POS_GROUPS) + 4
        if n_tokens == 0:
            return [0.0] * n_features

        pos_counts = Counter()
        for token in tokens:
            tag = self._parse_word(token)
            if tag:
                pos_counts[tag] += 1

        features = []

        for tag in POS_CATEGORIES:
            features.append(pos_counts.get(tag, 0) / n_tokens)

        for group_name, tags in POS_GROUPS.items():
            count = sum(pos_counts.get(t, 0) for t in tags)
            features.append(count / n_tokens)

        verb_count = sum(pos_counts.get(t, 0) for t in POS_GROUPS['verb_all'])
        noun_count = pos_counts.get('NOUN', 1)
        features.append(verb_count / noun_count)

        adj_count = sum(pos_counts.get(t, 0) for t in POS_GROUPS['adj'])
        features.append(adj_count / noun_count)

        content_words = [
            w for w, t in zip(tokens, [self._parse_word(w) for w in tokens])
            if t and t not in ('PREP', 'CONJ', 'PRCL', 'NPRO')
        ]
        features.append(sum(len(w) for w in content_words) / len(content_words) if content_words else 0)

        prefix_count = sum(1 for t in tokens if len(t) > 4 and any(t.startswith(p) for p in _PREFIXES))
        features.append(prefix_count / n_tokens)

        return features

    def get_feature_names(self) -> List[str]:
        names = [f'pos_{tag.lower()}' for tag in POS_CATEGORIES]
        names += [f'pos_group_{g}' for g in POS_GROUPS]
        names += ['pos_verb_noun_ratio', 'pos_adj_noun_ratio', 'pos_content_word_len', 'pos_prefix_ratio']
        return names

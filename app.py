import streamlit as st
import json
import re
import numpy as np
import torch
import joblib
import pandas as pd
from pathlib import Path
from collections import Counter

from src.data.preprocessor import clean_text
from src.features.base_features import BaseFeatureExtractor
from src.features.ngram_features import NgramExtractor
from src.features.pos_features import PosFeatureExtractor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / 'models_saved'
DATASETS_DIR = BASE_DIR / 'data' / 'datasets'

AUTHORS = ['dostoevsky', 'tolstoy', 'leskov']
AUTHOR_LABELS = {i: name for i, name in enumerate(AUTHORS)}
MIN_TEXT_LENGTH = 200
RECOMMENDED_TEXT_LENGTH = 1000


@st.cache_resource
def load_ml_models():
    models = {}
    ml_dir = MODELS_DIR / 'ml_subset'
    for name in ['xgboost', 'logreg', 'svm', 'lightgbm']:
        path = ml_dir / f'{name}.pkl'
        if path.exists():
            models[name] = joblib.load(path)
    return models


@st.cache_resource
def load_ensemble():
    ensemble_dir = MODELS_DIR / 'ensemble'
    ensemble_path = ensemble_dir / 'voting_ensemble.pkl'
    scaler_path = ensemble_dir / 'scaler.pkl'

    if ensemble_path.exists() and scaler_path.exists():
        ensemble = joblib.load(ensemble_path)
        scaler = joblib.load(scaler_path)
        return ensemble, scaler
    return None, None


@st.cache_resource
def load_rubert():
    rubert_path = MODELS_DIR / 'bert-tiny'
    if not rubert_path.exists():
        return None, None
    try:
        model = AutoModelForSequenceClassification.from_pretrained(str(rubert_path))
        tokenizer = AutoTokenizer.from_pretrained(str(rubert_path))
        model.eval()
        return model, tokenizer
    except Exception:
        return None, None


@st.cache_resource
def load_model_accuracies():
    accuracies = {}
    results_path = MODELS_DIR / 'ml_subset' / 'results.json'
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            for name, data in results.items():
                accuracies[name] = data['accuracy'] * 100

    ensemble_results_path = MODELS_DIR / 'ensemble' / 'results.json'
    if ensemble_results_path.exists():
        with open(ensemble_results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            for name, acc in results.items():
                accuracies[name] = acc * 100

    return accuracies


@st.cache_resource
def load_ngram_vocab():
    path = DATASETS_DIR / 'ml' / 'feature_names.json'
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as f:
        feature_names = json.load(f)
    return [n.replace('ngram_', '') for n in feature_names if n.startswith('ngram_')]


@st.cache_resource
def load_scaler():
    scaler_path = MODELS_DIR / 'ml_subset' / 'scaler.pkl'
    if scaler_path.exists():
        return joblib.load(scaler_path)
    return None


@st.cache_resource
def load_feature_extractors():
    return BaseFeatureExtractor(), PosFeatureExtractor(), NgramExtractor(n=3, top_k=500)


def predict_ml(text, model, model_name, ngram_vocab, scaler):
    base_extractor, pos_extractor, ngram_extractor = load_feature_extractors()
    ngram_extractor.vocab = ngram_vocab

    base = base_extractor.extract(text)
    ngrams = ngram_extractor.extract(text)
    pos = pos_extractor.extract(text)
    features = np.array(base + ngrams + pos, dtype=np.float32).reshape(1, -1)

    if model_name in ['logreg', 'svm', 'lightgbm'] and scaler is not None:
        features = scaler.transform(features)

    pred = model.predict(features)[0]

    # Получаем вероятности в зависимости от типа модели
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(features)[0]
    elif hasattr(model, 'decision_function'):
        # Для SVM без probability=True
        decision = model.decision_function(features)
        # Приводим к 1D массиву
        if decision.ndim == 2:
            decision = decision[0]
        # Конвертируем в вероятности через softmax
        exp_decision = np.exp(decision - np.max(decision))
        probs = exp_decision / exp_decision.sum()
    else:
        probs = np.zeros(len(AUTHORS))
        probs[pred] = 1.0

    return AUTHORS[pred], {a: float(p) for a, p in zip(AUTHORS, probs)}


def predict_rubert(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].numpy()

    pred_idx = np.argmax(probs)
    return AUTHORS[pred_idx], {a: float(p) for a, p in zip(AUTHORS, probs)}


def predict_ensemble(text, ensemble, scaler, ngram_vocab):
    base_extractor, pos_extractor, ngram_extractor = load_feature_extractors()
    ngram_extractor.vocab = ngram_vocab

    base = base_extractor.extract(text)
    ngrams = ngram_extractor.extract(text)
    pos = pos_extractor.extract(text)
    features = np.array(base + ngrams + pos, dtype=np.float32).reshape(1, -1)

    features_scaled = scaler.transform(features)

    # Получаем вероятности от XGBoost
    xgb_probs = ensemble.named_estimators_['xgb'].predict_proba(features)[0]

    # Получаем вероятности от LogReg
    lr_probs = ensemble.named_estimators_['lr'].predict_proba(features_scaled)[0]

    # Получаем вероятности от SVM через decision_function
    svm_decision = ensemble.named_estimators_['svm'].decision_function(features_scaled)
    svm_probs = softmax(svm_decision, axis=1)[0]

    # Усредняем (теперь все массивы одинаковой формы)
    ensemble_probs = (xgb_probs + lr_probs + svm_probs) / 3
    pred = np.argmax(ensemble_probs)

    return AUTHORS[pred], {a: float(p) for a, p in zip(AUTHORS, ensemble_probs)}


def render_prediction(author, probs):
    confidence = max(probs.values())
    st.markdown(f"### 🎯 {author.capitalize()}")
    st.caption(f"Уверенность: {confidence * 100:.1f}%")

    cols = st.columns(3)
    for i, (name, prob) in enumerate(sorted(probs.items(), key=lambda x: -x[1])):
        with cols[i]:
            st.metric(name.capitalize(), f"{prob * 100:.1f}%")


def render_probability_bars(probs):
    st.markdown("**Распределение вероятностей**")
    for name, prob in sorted(probs.items(), key=lambda x: -x[1]):
        st.progress(float(prob), text=f"{name.capitalize()}: {prob * 100:.1f}%")


def get_text_stats(text):
    words = text.split()
    sentences = [s for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    word_lengths = [len(w) for w in words]
    vocab = set(w.lower() for w in words)

    stats = {
        'Символов': len(text),
        'Слов': len(words),
        'Предложений': len(sentences),
        'Уникальных слов': len(vocab),
        'TTR': round(len(vocab) / max(len(words), 1), 3),
        'Средняя длина слова': round(np.mean(word_lengths), 1) if word_lengths else 0,
        'Слов в предложении': round(len(words) / max(len(sentences), 1), 1),
    }
    return stats


def get_pos_stats(text, pos_extractor):
    tokens = re.findall(r'[а-яёА-ЯЁ]+', text.lower())
    if not tokens:
        return {}

    morph = pos_extractor.morph
    counts = Counter()
    for token in tokens[:500]:
        parsed = morph.parse(token)
        if parsed:
            tag = parsed[0].tag.POS
            if tag:
                counts[tag] += 1

    total = sum(counts.values())
    if total == 0:
        return {}

    main_pos = {
        'Существительные': 'NOUN',
        'Глаголы': 'VERB',
        'Прилагательные': 'ADJF',
        'Наречия': 'ADVB',
        'Предлоги': 'PREP',
        'Союзы': 'CONJ',
        'Частицы': 'PRCL',
        'Местоимения': 'NPRO',
    }

    stats = {}
    for label, tag in main_pos.items():
        pct = counts.get(tag, 0) / total * 100
        stats[label] = f"{pct:.1f}%"

    verb_all = sum(counts.get(t, 0) for t in ['VERB', 'INFN', 'GRND', 'PRTF', 'PRTS'])
    noun_count = counts.get('NOUN', 1)
    stats['Глагол/Существительное'] = f"{verb_all / noun_count:.2f}"

    return stats


def main():
    st.set_page_config(page_title="Стилометрия", page_icon="📚", layout="wide")
    st.title("📚 Стилометрия")
    st.markdown("Определение автора текста на основе стилистических особенностей")

    ml_models = load_ml_models()
    ensemble, ensemble_scaler = load_ensemble()
    rubert_model, rubert_tokenizer = load_rubert()
    ngram_vocab = load_ngram_vocab()
    scaler = load_scaler()
    model_accuracies = load_model_accuracies()
    base_extractor, pos_extractor, _ = load_feature_extractors()

    available_models = list(ml_models.keys())
    if ensemble is not None:
        available_models.append('ensemble')
    if rubert_model is not None:
        available_models.append('rubert')

    if not available_models:
        st.error("Нет доступных моделей. Запустите обучение.")
        return

    with st.sidebar:
        st.header("⚙️ Настройки")

        mode = st.radio("Режим работы", ["Одна модель", "Сравнение моделей"], horizontal=True)

        if mode == "Одна модель":
            model_choice = st.selectbox(
                "Выберите модель",
                available_models,
                format_func=lambda x: {
                    'ensemble': f"Ансамбль (Voting) - 91.6%",
                    'xgboost': f"XGBoost - {model_accuracies.get('xgboost', 89.6):.1f}%",
                    'lightgbm': f"LightGBM - {model_accuracies.get('lightgbm', 89.4):.1f}%",
                    'svm': f"SVM - {model_accuracies.get('svm', 89.4):.1f}%",
                    'logreg': f"LogReg - {model_accuracies.get('logreg', 88.7):.1f}%",
                    'rubert': f"RuBERT - 89.0%"
                }.get(x, x)
            )

        st.divider()
        st.markdown("**О проекте**")
        st.caption(f"Авторы: {', '.join([a.capitalize() for a in AUTHORS])}")
        st.caption(f"Минимальная длина: {MIN_TEXT_LENGTH} символов")
        st.caption(f"Рекомендуемая длина: {RECOMMENDED_TEXT_LENGTH}+ символов")
        st.caption(f"Лучшая модель: Ансамбль (91.6%)")

    if 'sample_text' not in st.session_state:
        st.session_state.sample_text = ""

    col_input, col_stats = st.columns([2, 1])

    with col_input:
        text_input = st.text_area(
            "Введите текст для анализа",
            value=st.session_state.sample_text,
            height=250,
            placeholder="Вставьте текст на русском языке..."
        )

    text_clean = clean_text(text_input) if text_input.strip() else ""
    text_len = len(text_clean)

    with col_stats:
        st.markdown("**📊 Статистика текста**")
        if text_len > 0:
            stats = get_text_stats(text_clean)
            for key, val in stats.items():
                st.metric(key, val)
        else:
            st.caption("Введите текст для анализа")

        if text_len > 0:
            if text_len < MIN_TEXT_LENGTH:
                st.warning(f"⚠️ Текст короткий ({text_len} < {MIN_TEXT_LENGTH})")
            elif text_len >= RECOMMENDED_TEXT_LENGTH:
                st.success(f"✅ Достаточная длина ({text_len} символов)")

    if text_len >= MIN_TEXT_LENGTH:
        with st.expander("📖 Части речи (POS-анализ)"):
            with st.spinner("Анализ..."):
                pos_stats = get_pos_stats(text_clean, pos_extractor)
                if pos_stats:
                    cols = st.columns(3)
                    for i, (key, val) in enumerate(pos_stats.items()):
                        cols[i % 3].metric(key, val)

    if st.button("🔍 Определить автора", type="primary", disabled=text_len == 0):
        if mode == "Одна модель":
            st.divider()

            if model_choice == 'rubert':
                if rubert_model is None:
                    st.error("Модель RuBERT не загружена")
                else:
                    author, probs = predict_rubert(text_clean, rubert_model, rubert_tokenizer)
                    render_prediction(author, probs)
                    render_probability_bars(probs)
            elif model_choice == 'ensemble':
                if ensemble is None:
                    st.error("Ансамбль не загружен")
                else:
                    author, probs = predict_ensemble(text_clean, ensemble, ensemble_scaler, ngram_vocab)
                    render_prediction(author, probs)
                    render_probability_bars(probs)
            elif model_choice in ml_models:
                author, probs = predict_ml(text_clean, ml_models[model_choice], model_choice, ngram_vocab, scaler)
                render_prediction(author, probs)
                render_probability_bars(probs)
            else:
                st.error(f"Модель {model_choice} недоступна")

        else:
            st.divider()
            st.subheader("📊 Сравнение моделей")

            results = []

            # ML модели
            for name in ml_models:
                try:
                    author, probs = predict_ml(text_clean, ml_models[name], name, ngram_vocab, scaler)
                    results.append({
                        'Модель': name.replace('_', ' ').title(),
                        'Предсказание': author.capitalize(),
                        'Уверенность': f"{max(probs.values()) * 100:.1f}%",
                        'Достоевский': f"{probs.get('dostoevsky', 0) * 100:.1f}%",
                        'Толстой': f"{probs.get('tolstoy', 0) * 100:.1f}%",
                        'Лесков': f"{probs.get('leskov', 0) * 100:.1f}%",
                    })
                except Exception as e:
                    results.append({'Модель': name, 'Ошибка': str(e)})

            # Ансамбль
            if ensemble is not None:
                try:
                    author, probs = predict_ensemble(text_clean, ensemble, ensemble_scaler, ngram_vocab)
                    results.append({
                        'Модель': 'Ансамбль (Voting)',
                        'Предсказание': author.capitalize(),
                        'Уверенность': f"{max(probs.values()) * 100:.1f}%",
                        'Достоевский': f"{probs.get('dostoevsky', 0) * 100:.1f}%",
                        'Толстой': f"{probs.get('tolstoy', 0) * 100:.1f}%",
                        'Лесков': f"{probs.get('leskov', 0) * 100:.1f}%",
                    })
                except Exception as e:
                    results.append({'Модель': 'Ансамбль', 'Ошибка': str(e)})

            # RuBERT
            if rubert_model is not None:
                try:
                    author, probs = predict_rubert(text_clean, rubert_model, rubert_tokenizer)
                    results.append({
                        'Модель': 'RuBERT',
                        'Предсказание': author.capitalize(),
                        'Уверенность': f"{max(probs.values()) * 100:.1f}%",
                        'Достоевский': f"{probs.get('dostoevsky', 0) * 100:.1f}%",
                        'Толстой': f"{probs.get('tolstoy', 0) * 100:.1f}%",
                        'Лесков': f"{probs.get('leskov', 0) * 100:.1f}%",
                    })
                except Exception as e:
                    results.append({'Модель': 'RuBERT', 'Ошибка': str(e)})

            if results:
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True, hide_index=True)

                preds = [r['Предсказание'] for r in results if 'Предсказание' in r]
                if len(set(preds)) == 1 and preds:
                    st.success(f"✅ Все модели согласны: **{preds[0]}**")
                elif preds:
                    counts = Counter(preds)
                    winner, count = counts.most_common(1)[0]
                    st.info(f"📊 Большинство моделей ({count}/{len(preds)}): **{winner}**")

    with st.expander("ℹ️ О проекте"):
        st.markdown("""
        **Как это работает:**
        - Анализируются стилистические признаки текста (длина слов, пунктуация, n-граммы, части речи)
        - Обученные модели сравнивают текст с эталонами Достоевского, Толстого и Лескова
        - Выдаётся вероятностное распределение по трём авторам

        **Модели:**
        - **Ансамбль (Voting)** — комбинация XGBoost, SVM, LogReg (91.6%)
        - **XGBoost** — градиентный бустинг (89.6%)
        - **LightGBM** — быстрый градиентный бустинг (89.4%)
        - **SVM** — метод опорных векторов (89.4%)
        - **LogReg** — логистическая регрессия (88.7%)

        **Лучший результат:** 91.55% на тестовой выборке
        """)

    st.divider()
    st.caption("Стилометрия | ML + Deep Learning для определения автора текста")


if __name__ == "__main__":
    main()
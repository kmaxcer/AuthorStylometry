import streamlit as st
import json
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.data.preprocessor import clean_text
from src.features.base_features import BaseFeatureExtractor
from src.features.ngram_features import NgramExtractor
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Стилометрия", page_icon="📚", layout="wide")

st.title("Стилометрия")
st.markdown("Определение автора текста на основе стилистических особенностей")


@st.cache_resource
def load_models():
    base_dir = Path(__file__).parent
    models_dir = base_dir / 'models_saved'

    import joblib

    # Модели на 3 автора
    ml_models = {}
    ml_models['xgboost'] = joblib.load(models_dir / 'ml_subset' / 'xgboost.pkl')
    ml_models['random_forest'] = joblib.load(models_dir / 'ml_subset' / 'random_forest.pkl')

    # RuBERT
    rubert_path = models_dir / 'bert-tiny'
    rubert_model = None
    rubert_tokenizer = None

    if rubert_path.exists():
        try:
            rubert_model = AutoModelForSequenceClassification.from_pretrained(str(rubert_path))
            rubert_tokenizer = AutoTokenizer.from_pretrained(str(rubert_path))
            rubert_model.eval()
        except Exception as e:
            rubert_model = None

    # Метаданные для 3 авторов
    authors_3 = ['dostoevsky', 'tolstoy', 'leskov']
    label_to_author = {i: name for i, name in enumerate(authors_3)}

    # N-gram словарь (берем из ML датасета, он одинаковый)
    datasets_dir = base_dir / 'data' / 'datasets'
    with open(datasets_dir / 'ml' / 'feature_names.json', 'r', encoding='utf-8') as f:
        feature_names = json.load(f)

    ngram_vocab = [n.replace('ngram_', '') for n in feature_names if n.startswith('ngram_')]

    return ml_models, rubert_model, rubert_tokenizer, label_to_author, ngram_vocab


with st.sidebar:
    st.header("Настройки")
    model_choice = st.selectbox(
        "Модель",
        ["xgboost", "random_forest", "rubert"],
        format_func=lambda x: {
            "xgboost": "XGBoost (88.5%)",
            "random_forest": "Random Forest (81.8%)",
            "rubert": "RuBERT-tiny (89.0%)"
        }[x]
    )
    st.info("3 автора: Достоевский, Толстой, Лесков")

text_input = st.text_area("Введите текст для анализа", height=200)

if st.button("Определить автора"):
    if not text_input.strip():
        st.warning("Введите текст")
    else:
        with st.spinner("Анализ текста..."):
            cleaned = clean_text(text_input)
            st.info(f"Длина текста: {len(cleaned)} символов")

            ml_models, rubert_model, rubert_tokenizer, label_to_author, ngram_vocab = load_models()

            if model_choice == "rubert":
                if rubert_model is None:
                    st.error("Модель RuBERT не найдена")
                else:
                    inputs = rubert_tokenizer(cleaned, return_tensors='pt', truncation=True, max_length=512)
                    with torch.no_grad():
                        outputs = rubert_model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=1)[0].numpy()

                    authors_3 = ['dostoevsky', 'tolstoy', 'leskov']
                    pred_idx = np.argmax(probs)
                    author = authors_3[pred_idx]
                    confidence = float(probs[pred_idx])

                    st.success(f"Автор: {author.capitalize()}")
                    st.info(f"Уверенность: {confidence * 100:.1f}%")

                    for a, p in zip(authors_3, probs):
                        st.progress(float(p), text=f"{a.capitalize()}: {p * 100:.1f}%")

            elif model_choice in ml_models:
                model = ml_models[model_choice]

                base_extractor = BaseFeatureExtractor()
                ngram_extractor = NgramExtractor(n=3, top_k=500)
                ngram_extractor.vocab = ngram_vocab

                features = base_extractor.extract(cleaned) + ngram_extractor.extract(cleaned)
                features = np.array(features).reshape(1, -1)

                pred = model.predict(features)[0]
                probs = model.predict_proba(features)[0]
                author = label_to_author[pred]
                confidence = float(max(probs))

                st.success(f"Автор: {author.capitalize()}")
                st.info(f"Уверенность: {confidence * 100:.1f}%")

                for i, prob in enumerate(probs):
                    if prob > 0.01:
                        a = label_to_author[i]
                        st.progress(float(prob), text=f"{a.capitalize()}: {prob * 100:.1f}%")
            else:
                st.error(f"Модель {model_choice} не найдена")
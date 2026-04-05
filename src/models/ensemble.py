"""
Создание и сохранение ансамбля (запустить один раз)
"""

import numpy as np
import joblib
from pathlib import Path
import json
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent.parent.parent
DATASETS_DIR = BASE_DIR / 'data' / 'datasets'
MODELS_DIR = BASE_DIR / 'models_saved'

print("Загрузка моделей...")

# Загружаем модели
xgb = joblib.load(MODELS_DIR / 'ml_subset' / 'xgboost.pkl')
svm = joblib.load(MODELS_DIR / 'ml_subset' / 'svm.pkl')
lr = joblib.load(MODELS_DIR / 'ml_subset' / 'logreg.pkl')
scaler = joblib.load(MODELS_DIR / 'ml_subset' / 'scaler.pkl')

print("Создание ансамбля...")

# Создаём Voting Classifier
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb),
        ('svm', svm),
        ('lr', lr)
    ],
    voting='soft',
    n_jobs=-1
)

# Загружаем данные для "обучения" ансамбля (нужно для совместимости)
ml_dir = DATASETS_DIR / 'ml'
X_train = np.load(ml_dir / 'X.npy')
y_train = np.load(ml_dir / 'y.npy')

# Фильтруем для 3 авторов
with open(DATASETS_DIR / 'author_labels.json', 'r') as f:
    labels = json.load(f)

authors_3 = ['dostoevsky', 'tolstoy', 'leskov']
author_to_label = labels['author_to_label']
subset_labels = [author_to_label[a] for a in authors_3]

train_mask = np.isin(y_train, subset_labels)
X_train = X_train[train_mask]
y_train = y_train[train_mask]

old_to_new = {old: new for new, old in enumerate(subset_labels)}
y_train = np.array([old_to_new[y] for y in y_train])

# Масштабируем
X_train_scaled = scaler.transform(X_train)

# "Обучаем" ансамбль (на самом деле он просто сохраняет ссылки на модели)
ensemble.fit(X_train_scaled, y_train)

# Сохраняем
ensemble_dir = MODELS_DIR / 'ensemble'
ensemble_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(ensemble, ensemble_dir / 'voting_ensemble.pkl')
joblib.dump(scaler, ensemble_dir / 'scaler.pkl')

print(f"✅ Ансамбль сохранён в {ensemble_dir / 'voting_ensemble.pkl'}")
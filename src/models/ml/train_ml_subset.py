"""
Обучение ML моделей на подмножестве авторов
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

from random_forest import create_random_forest
from svm_classifier import create_svm
from xgboost_model import create_xgboost
from logreg_model import create_logreg

try:
    import lightgbm as lgb

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM не установлен. Установите: pip install lightgbm")


class MLTrainerSubset:
    def __init__(self, authors_subset=None):
        self.base_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.datasets_dir = self.base_dir / 'data' / 'datasets'
        self.models_dir = self.base_dir / 'models_saved' / 'ml_subset'
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.models = {}

        # Добавляем модели
        self.models['logreg'] = create_logreg(C=1.0, class_weight='balanced')
        self.models['random_forest'] = create_random_forest()
        self.models['svm'] = create_svm()
        self.models['xgboost'] = create_xgboost()

        if HAS_LGBM:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )

        self.results = {}
        self.authors_subset = authors_subset
        self.class_names = None
        self.scaler = StandardScaler()

    def load_data(self):
        ml_dir = self.datasets_dir / 'ml'
        X_train = np.load(ml_dir / 'X.npy')
        y_train = np.load(ml_dir / 'y.npy')
        X_test = np.load(ml_dir / 'X_test.npy')
        y_test = np.load(ml_dir / 'y_test.npy')

        with open(self.datasets_dir / 'author_labels.json', 'r', encoding='utf-8') as f:
            labels = json.load(f)

        author_to_label = labels['author_to_label']

        if self.authors_subset:
            subset_labels = [author_to_label[a] for a in self.authors_subset]

            train_mask = np.isin(y_train, subset_labels)
            test_mask = np.isin(y_test, subset_labels)

            X_train = X_train[train_mask]
            y_train = y_train[train_mask]
            X_test = X_test[test_mask]
            y_test = y_test[test_mask]

            old_to_new = {old: new for new, old in enumerate(subset_labels)}
            y_train = np.array([old_to_new[y] for y in y_train])
            y_test = np.array([old_to_new[y] for y in y_test])

            self.class_names = self.authors_subset

        # Масштабируем для моделей, которым это нужно
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train, X_train_scaled, y_train, X_test, X_test_scaled, y_test

    def train_and_evaluate(self):
        X_train, X_train_scaled, y_train, X_test, X_test_scaled, y_test = self.load_data()

        for name, model in self.models.items():
            # Для LogReg, SVM, LightGBM используем масштабированные данные
            if name in ['logreg', 'svm', 'lightgbm']:
                X_train_use = X_train_scaled
                X_test_use = X_test_scaled
            else:
                X_train_use = X_train
                X_test_use = X_test

            model.fit(X_train_use, y_train)
            y_pred = model.predict(X_test_use)
            accuracy = accuracy_score(y_test, y_pred)

            self.results[name] = {
                'model': model,
                'accuracy': accuracy
            }

    def save_models(self):
        for name, result in self.results.items():
            joblib.dump(result['model'], self.models_dir / f'{name}.pkl')

        joblib.dump(self.scaler, self.models_dir / 'scaler.pkl')

        with open(self.models_dir / 'results.json', 'w', encoding='utf-8') as f:
            json.dump(
                {name: {'accuracy': float(result['accuracy'])} for name, result in self.results.items()},
                f, ensure_ascii=False, indent=2
            )

        if self.class_names:
            with open(self.models_dir / 'authors.json', 'w', encoding='utf-8') as f:
                json.dump({'authors': self.class_names}, f, ensure_ascii=False, indent=2)

    def run(self):
        self.train_and_evaluate()
        self.save_models()


def main():
    authors = ['dostoevsky', 'tolstoy', 'leskov']
    trainer = MLTrainerSubset(authors_subset=authors)
    trainer.run()


if __name__ == "__main__":
    main()
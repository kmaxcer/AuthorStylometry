# Стилометрия: Определение автора текста на материале русской литературы XIX века

## Описание проекта

Проект представляет собой полный пайплайн машинного обучения для атрибуции авторства русских литературных текстов XIX века. Система определяет автора текстового фрагмента на основе стилистических особенностей.

**Лучший результат:** 89% точности на 3 авторах (Достоевский, Толстой, Лесков) с использованием модели RuBERT-tiny.

## Возможности

- **Полный ML-пайплайн**: от очистки сырых текстов до обучения моделей
- **Два подхода**: классическое ML (Random Forest, SVM, XGBoost) и нейросети (LSTM, Transformer, RuBERT)
- **Комплексное извлечение признаков**: 548 признаков, включая лексические, пунктуационные и символьные n-граммы
- **Поддержка русского языка**: обработка специфических явлений (заикания, устаревшие частицы, OCR-артефакты)

## Структура проекта

```
AuthorStylometry/
├── data/
│   ├── raw/                 # Исходные тексты (не в репозитории)
│   ├── processed/           # Очищенные тексты
│   ├── windows/             # Текстовые окна (1000 символов, шаг 500)
│   └── datasets/            # Датасеты для ML и нейросетей
├── src/
│   ├── data/                # Обработка данных
│   │   ├── preprocessor.py
│   │   ├── window_generator.py
│   │   ├── loader.py
│   │   ├── splitter.py
│   │   └── dataset_builder.py
│   ├── features/            # Извлечение признаков
│   │   ├── base_features.py
│   │   └── ngram_features.py
│   ├── models/
│   │   ├── ml/              # Классические ML модели
│   │   │   ├── random_forest.py
│   │   │   ├── svm_classifier.py
│   │   │   ├── xgboost_model.py
│   │   │   ├── train_ml.py
│   │   │   └── train_ml_subset.py
│   │   └── nn/              # Нейросети
│   │       ├── lstm_model.py
│   │       ├── transformer_model.py
│   │       ├── rubert_model.py
│   │       └── train_nn.py
│   └── utils/               # Вспомогательные модули
├── models_saved/            # Обученные модели
└── requirements.txt         # Зависимости
```

## Источник данных

Тексты взяты из репозитория [RSD](https://github.com/nevmenandr/RSD):
- `author/fiction/period/19-1/corpus` → `data/raw/corpus_19_1/`
- `author/fiction/period/19-2/brevia/corpus` → `data/raw/corpus_19_2/`

## Установка

```bash
git clone https://github.com/your-username/AuthorStylometry.git
cd AuthorStylometry
pip install -r requirements.txt
```

## Использование

### 1. Очистка текстов
```bash
python src/data/preprocessor.py
```

### 2. Генерация текстовых окон
```bash
python src/data/window_generator.py
```

### 3. Сборка датасетов (признаки для ML + последовательности для нейросетей)
```bash
python src/data/dataset_builder.py
```

### 4. Обучение моделей

**Классическое ML:**
```bash
python src/models/ml/train_ml.py               # все авторы
python src/models/ml/train_ml_subset.py        # 3 автора (Достоевский, Толстой, Лесков)
```

**Нейросети:**
```bash
python src/models/nn/train_nn.py               # LSTM по умолчанию
# Для смены модели измените model_type в train_nn.py: 'lstm', 'transformer' или 'rubert'
```

## Результаты

| Модель | 3 автора | 15 авторов |
|--------|----------|------------|
| Random Forest | 81.8% | 50.0% |
| XGBoost | **88.5%** | — |
| LSTM (символьный) | 67.6% | 52.9% |
| **RuBERT-tiny** | **89.0%** | — |

## Лицензия

MIT


## Благодарности

- Тексты из репозитория [RSD](https://github.com/nevmenandr/RSD) (автор: Борис Орехов)
- Модель RuBERT от DeepPavlov / cointegrated
```

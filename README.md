# AuthorStylometry

Определение автора русского литературного текста XIX века по стилистическим особенностям.

## Результаты

| Модель | 3 автора | 15 авторов |
|--------|----------|------------|
| Random Forest | 81.8% | 50.0% |
| XGBoost | **88.5%** | — |
| LSTM (символьный) | 67.6% | 52.9% |
| **RuBERT-tiny** | **89.0%** | — |

3 автора: Достоевский, Толстой, Лесков.

## Возможности

- 548 признаков: лексические, пунктуационные, символьные n-граммы
- 6 моделей: Random Forest, SVM, XGBoost, LSTM, Transformer, RuBERT
- Предобработка русских текстов: заикания, устаревшие частицы -с, OCR-артефакты
- Веб-интерфейс (Streamlit) с режимом сравнения всех моделей

## Установка

```bash
git clone https://github.com/your-username/AuthorStylometry.git
cd AuthorStylometry
pip install -r requirements.txt
```

## Пайплайн

```
Сырые тексты → Очистка → Текстовые окна → Признаки → Обучение → Предсказание
```

### Полный цикл

```bash
python run.py
```

### Пошаговый запуск

```bash
# 1. Очистка текстов
python src/data/preprocessor.py

# 2. Генерация окон (1000 символов, шаг 500)
python src/data/window_generator.py

# 3. Сборка датасетов (признаки + последовательности)
python src/data/dataset_builder.py

# 4. Обучение
python src/models/ml/train_ml.py          # все авторы
python src/models/ml/train_ml_subset.py   # 3 автора
python src/models/nn/train_nn.py          # нейросети
```

### Веб-приложение

```bash
streamlit run app.py
```

Открывается на `http://localhost:8501`. Два режима: предсказание одной моделью и сравнение всех.

## Структура проекта

```
AuthorStylometry/
├── app.py                        # Веб-интерфейс (Streamlit)
├── run.py                        # Точка входа пайплайна
├── requirements.txt              # Зависимости
│
├── src/
│   ├── data/
│   │   ├── preprocessor.py       # Очистка текстов
│   │   ├── window_generator.py   # Разбиение на окна
│   │   ├── dataset_builder.py    # Сборка ML и NN датасетов
│   │   ├── loader.py             # Загрузка окон
│   │   └── splitter.py           # Train/test split по произведениям
│   │
│   ├── features/
│   │   ├── base_features.py      # 48 базовых признаков
│   │   └── ngram_features.py     # Символьные n-граммы
│   │
│   ├── models/
│   │   ├── ml/
│   │   │   ├── random_forest.py
│   │   │   ├── svm_classifier.py
│   │   │   ├── xgboost_model.py
│   │   │   ├── train_ml.py       # Обучение на всех авторах
│   │   │   └── train_ml_subset.py
│   │   └── nn/
│   │       ├── lstm_model.py
│   │       ├── transformer_model.py
│   │       ├── rubert_model.py
│   │       ├── attention.py
│   │       └── train_nn.py
│   │
│   ├── visualization/            # Графики и сравнения
│   └── utils/                    # Метрики, логирование
│
├── data/                         # Не в репозитории
│   ├── raw/                      # Исходные тексты
│   ├── processed/                # Очищенные тексты
│   ├── windows/                  # Текстовые окна
│   └── datasets/                 # Готовые датасеты
│
├── models_saved/                 # Обученные модели
├── experiments/                  # Результаты экспериментов
└── reports/                      # Отчёты
```

## Признаки

48 базовых признаков извлекаются из каждого текстового окна:

| Группа | Кол-во | Примеры |
|--------|--------|---------|
| Статистика текста | 7 | кол-во слов/предложений, TTR, hapax ratio |
| Пунктуация | 16 | частоты точки, запятой, тире и т.д. |
| Стоп-слова | 10 | относительные частоты «и», «в», «не», ... |
| Частоты букв | 10 | относительные частоты «о», «е», «а», ... |
| Стилистические | 5 | upper ratio, digit ratio, dialogue dashes, ... |
| n-граммы | 500 | символьные 3-граммы |

## Источник данных

Тексты из репозитория [RSD](https://github.com/nevmenandr/RSD):
- `author/fiction/period/19-1/corpus` → `data/raw/corpus_19_1/`
- `author/fiction/period/19-2/brevia/corpus` → `data/raw/corpus_19_2/`

## Лицензия

MIT

## Благодарности

- Тексты: [RSD](https://github.com/nevmenandr/RSD) (Борис Орехов)
- Модель: [RuBERT](https://huggingface.co/cointegrated/rubert-tiny2) (DeepPavlov / cointegrated)

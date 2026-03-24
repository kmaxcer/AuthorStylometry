#!/usr/bin/env python
"""
Главный скрипт для запуска всего пайплайна стилометрии
"""

import argparse
import sys
from pathlib import Path

# Добавляем src в путь для импортов
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def main():
    parser = argparse.ArgumentParser(description='Stylometry pipeline for author attribution')
    parser.add_argument('--step', type=str, default='all',
                        choices=['preprocess', 'windows', 'datasets', 'train_ml', 'train_nn', 'all'],
                        help='Какой этап выполнить')
    parser.add_argument('--model', type=str, default='xgboost',
                        choices=['random_forest', 'svm', 'xgboost', 'lstm', 'transformer', 'rubert'],
                        help='Модель для обучения (для train_ml/train_nn)')
    args = parser.parse_args()

    if args.step in ['preprocess', 'all']:
        from data.preprocessor import main as preprocess_main
        preprocess_main()

    if args.step in ['windows', 'all']:
        from data.window_generator import main as window_main
        window_main()

    if args.step in ['datasets', 'all']:
        from data.dataset_builder import main as dataset_main
        dataset_main()

    if args.step in ['train_ml', 'all']:
        if args.model in ['random_forest', 'svm', 'xgboost']:
            from models.ml.train_ml import MLTrainer
            trainer = MLTrainer()
            trainer.run()
        else:
            from models.ml.train_ml import MLTrainer
            trainer = MLTrainer()
            trainer.run()

    if args.step in ['train_nn', 'all']:
        from models.nn.train_nn import NNTrainer
        model_type = args.model if args.model in ['lstm', 'transformer', 'rubert'] else 'rubert'
        trainer = NNTrainer(model_type=model_type, batch_size=32, epochs=10)
        trainer.run()


if __name__ == "__main__":
    main()
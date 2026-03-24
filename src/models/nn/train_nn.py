"""
Обучение нейросетей для стилометрии
"""

import json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from lstm_model import create_lstm_model
from transformer_model import create_transformer_model
from rubert_model import create_rubert_model


class CharDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class NNTrainer:
    def __init__(self, model_type='lstm', batch_size=64, epochs=20, lr=2e-5, early_stopping_patience=3):
        self.base_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.datasets_dir = self.base_dir / 'data' / 'datasets'
        self.models_dir = self.base_dir / 'models_saved' / 'nn'
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.model_type = model_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.early_stopping_patience = early_stopping_patience

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_data(self):
        nn_dir = self.datasets_dir / 'nn'
        X_train = np.load(nn_dir / 'X_sequences.npy')
        y_train = np.load(nn_dir / 'y_labels.npy')
        X_test = np.load(nn_dir / 'X_test_sequences.npy')
        y_test = np.load(nn_dir / 'y_test_labels.npy')

        with open(self.datasets_dir / 'author_labels.json', 'r', encoding='utf-8') as f:
            self.labels = json.load(f)

        with open(nn_dir / 'char_to_idx.json', 'r', encoding='utf-8') as f:
            self.char_to_idx = json.load(f)

        return X_train, y_train, X_test, y_test

    def decode_sequences(self, sequences):
        idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        texts = []
        for seq in sequences:
            chars = [idx_to_char.get(idx, '') for idx in seq if idx not in [0, 1]]
            texts.append(''.join(chars))
        return texts

    def filter_authors(self, X_train, y_train, X_test, y_test, authors):
        author_to_label = self.labels['author_to_label']
        subset_labels = [author_to_label[a] for a in authors]

        train_mask = np.isin(y_train, subset_labels)
        test_mask = np.isin(y_test, subset_labels)

        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]

        old_to_new = {old: new for new, old in enumerate(subset_labels)}
        y_train = np.array([old_to_new[y] for y in y_train])
        y_test = np.array([old_to_new[y] for y in y_test])

        return X_train, y_train, X_test, y_test, len(authors)

    def create_model(self, vocab_size, num_classes):
        if self.model_type == 'lstm':
            return create_lstm_model(vocab_size, num_classes)
        elif self.model_type == 'transformer':
            return create_transformer_model(vocab_size, num_classes)
        elif self.model_type == 'rubert':
            return create_rubert_model(num_classes)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train_lstm_transformer(self, X_train, y_train, X_test, y_test, num_classes, name):
        vocab_size = len(self.char_to_idx)

        train_dataset = CharDataset(X_train, y_train)
        test_dataset = CharDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        model = self.create_model(vocab_size, num_classes)
        model = model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=self.lr)

        best_acc = 0
        patience_counter = 0

        for epoch in range(self.epochs):
            model.train()
            correct = 0
            total = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

            train_acc = correct / total

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    outputs = model(X_batch)
                    _, predicted = torch.max(outputs, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()

            test_acc = correct / total

            if test_acc > best_acc:
                best_acc = test_acc
                patience_counter = 0
                torch.save(model.state_dict(), self.models_dir / f'{self.model_type}_{name}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break

        return best_acc

    def train_rubert(self, X_train, y_train, X_test, y_test, num_classes, name):
        train_texts = self.decode_sequences(X_train)
        test_texts = self.decode_sequences(X_test)

        model = self.create_model(None, num_classes)
        model = model.to(self.device)

        train_dataset = TextDataset(train_texts, y_train, model.tokenizer)
        test_dataset = TextDataset(test_texts, y_test, model.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = AdamW(model.model.parameters(), lr=self.lr)

        best_acc = 0
        patience_counter = 0

        for epoch in range(self.epochs):
            model.train()
            correct = 0
            total = 0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                preds = torch.argmax(outputs.logits, dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

            train_acc = correct / total

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = model(input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=1)
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()

            test_acc = correct / total

            if test_acc > best_acc:
                best_acc = test_acc
                patience_counter = 0
                torch.save(model.model.state_dict(), self.models_dir / f'{self.model_type}_{name}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break

        return best_acc

    def train(self, X_train, y_train, X_test, y_test, num_classes, name):
        if self.model_type == 'rubert':
            return self.train_rubert(X_train, y_train, X_test, y_test, num_classes, name)
        else:
            return self.train_lstm_transformer(X_train, y_train, X_test, y_test, num_classes, name)

    def run(self):
        X_train, y_train, X_test, y_test = self.load_data()

        results = {}

        results['15_authors'] = self.train(X_train, y_train, X_test, y_test, self.labels['n_classes'], '15_authors')

        authors_3 = ['dostoevsky', 'tolstoy', 'leskov']
        X_train_3, y_train_3, X_test_3, y_test_3, num_3 = self.filter_authors(
            X_train, y_train, X_test, y_test, authors_3
        )
        results['3_authors'] = self.train(X_train_3, y_train_3, X_test_3, y_test_3, num_3, '3_authors')

        with open(self.models_dir / 'results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    trainer = NNTrainer(model_type='rubert', batch_size=32, epochs=10, lr=2e-5)
    trainer.run()


if __name__ == "__main__":
    main()
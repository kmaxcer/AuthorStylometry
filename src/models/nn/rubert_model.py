"""
Модуль для работы с RuBERT-tiny
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class RuBERTTinyClassifier(nn.Module):
    def __init__(self, num_classes, pretrained='cointegrated/rubert-tiny'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained,
            num_labels=num_classes
        )

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)


def create_rubert_model(num_classes, pretrained='cointegrated/rubert-tiny'):
    return RuBERTTinyClassifier(num_classes, pretrained)
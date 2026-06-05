import torch
import torch.nn as nn
from transformers import BertModel

class berttestClassifier(nn.Module):
    def __init__(self, num_classes=3, dropout_prob=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(768, num_classes)

    def forward(self, x):
        outputs = self.bert(x)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits
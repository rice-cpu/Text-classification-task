import torch
import torch.nn as nn
from transformers import BertModel

class berttextClassifier(nn.Module):
    def __init__(self, num_classes=14, dropout_prob=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)

        return logits
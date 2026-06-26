import torch
import torch.nn as nn
from transformers import BertModel

class berttextClassifier(nn.Module):
    def __init__(self, model_name="bert-base-chinese", num_classes=15):
        super (berttextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask,token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        dropped_output = self.drop(pooled_output)
        logits = self.out(dropped_output)

        return logits
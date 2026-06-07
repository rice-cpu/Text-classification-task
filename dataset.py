import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class testDataset(Dataset):
    def __init__(self, file_path, max_len):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.max_len = max_len
        self.texts = []
        self.labels = []
        labels_raw = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("_!_")
                if len(parts) < 4:
                    continue
                self.texts.append(parts[3])
                labels_raw.append(parts[1])

        unique_labels = sorted(list(set(labels_raw)))
        label_map = {raw_id: idx for idx, raw_id in enumerate(unique_labels)}

        self.labels = [label_map[raw_id] for raw_id in labels_raw]

        self.num_classes = len(unique_labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
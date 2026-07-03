import os
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data_dir, filename, tokenizer, config=None):
        self.tokenizer = tokenizer
        self.config = config

        if config:
            self.is_keyword = config.get("is_keyword", False)
            self.split_char = config.get("split_char", " ， ")
        else:
            self.is_keyword = False
            self.split_char = " ， "

        self.file_path = os.path.join(data_dir, filename)
        self.texts, self.labels = self._load_data(self.file_path)

    def _load_data(self, file_path):
        texts, raw_labels = [], []
        if self.config and "class_list" in self.config:
            fixed_class_list = self.config["class_list"]
        else:
            fixed_class_list = ["100", "101", "102", "103",
                                "104", "106", "107", "108",
                                "109", "110", "112", "113",
                                "114", "115", "116"]
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split("_!_")
                try:
                    title = parts[3]
                    label = parts[1]

                    if self.is_keyword and len(parts) >= 5 and parts[4].strip():
                        keywords = parts[4].strip()
                        full_text = title + self.split_char + keywords
                    else:
                        full_text = title

                    texts.append(full_text)
                    raw_labels.append(label)
                except IndexError:
                    continue

        labels = [fixed_class_list.index(l) for l in raw_labels if l in fixed_class_list]
        return texts, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        max_len = self.config.get("max_length", 64) if self.config else 64
        inputs = self.tokenizer(
            str(self.texts[index]),
            truncation=True,
            max_length=max_len,
            padding=False
        )
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'token_type_ids': inputs.get('token_type_ids', [0] * len(inputs['input_ids'])),
            'labels': self.labels[index]
        }
    @staticmethod
    def collate_fn(batch,tokenizer):
        labels = [item['labels'] for item in batch]
        features = [{k: v for k, v in item.items() if k != 'labels'} for item in batch]
        padded_features = tokenizer.pad(features,padding=True,return_tensors="pt")
        padded_features['labels'] = torch.tensor(labels, dtype=torch.long)
        return padded_features

import os
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data_dir, filename, tokenizer, max_len=64):
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.file_path = os.path.join(data_dir, filename)
        self.texts, self.labels = self._load_data(self.file_path)

    def _load_data(self, file_path):
        texts, raw_labels = [], []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split("_!_")
                if len(parts) < 4: continue

                texts.append(parts[3])
                raw_labels.append(parts[1])

        unique_labels = sorted(list(set(raw_labels)))
        label_map = {label_name: idx for idx, label_name in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)

        labels = [label_map[l] for l in raw_labels]
        return texts, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        inputs = self.tokenizer(
            str(self.texts[index]),
            truncation=True,
            max_length=self.max_len
        )
        return {
            'input_ids':inputs['input_ids'],
            'attention_mask':inputs['attention_mask'],
            'labels':self.labels[index]
        }
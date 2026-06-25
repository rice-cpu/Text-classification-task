import os
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data_dir, filename, tokenizer, max_len=64, config=None):
        self.tokenizer = tokenizer
        self.max_len = max_len

        if config:
            self.is_keyword = config.get("is_keyword", False)
            self.split_char = config.get("split_char", " ， ")
        else:
            self.is_keyword = False
            self.split_char = " ，"

        self.file_path = os.path.join(data_dir, filename)
        self.texts, self.labels = self._load_data(self.file_path)

    def _load_data(self, file_path):
        texts, raw_labels = [], []

        fixed_class_list = [
            "100", "101", "102", "103", "104",
            "106", "107", "108", "109", "110",
            "112", "113", "114", "115", "116"
        ]

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
        inputs = self.tokenizer(
            str(self.texts[index]),
            truncation=True,
            max_length=self.max_len
        )
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': self.labels[index]
        }
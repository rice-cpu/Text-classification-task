import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class testDataset(Dataset):
    def __init__(self, texts, labels, max_len=64):
        self.labels = labels
        print(f"texts的类型是: {type(texts)}，总长度是: {len(texts)}")
        print(f"labels的总长度是: {len(labels)}")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        clean_texts = [str(t) for t in texts]
        self.encodings = self.tokenizer(
            list(texts),
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        print(f"input_ids矩阵形状为: {self.encodings['input_ids'].shape}")

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class testDataset(Dataset):
    def __init__(self, texts, labels, max_len=64):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(text,max_length=self.max_len,padding='max_length',truncation=True,return_tensors="pt")
        return encoding['input_ids'].squeeze(0), torch.tensor(label, dtype=torch.long)
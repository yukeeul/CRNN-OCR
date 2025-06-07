import pickle
import string

import torch
import torchvision.transforms as T
from datasets import load_dataset as load_dataset_hf
from dvc.api import open as dvc_open


def load_dataset(from_dvc=True):
    if from_dvc:
        with dvc_open("data/dataset.pickle", remote="data", mode="rb") as f:
            ds = pickle.load(f)
    else:
        ds = load_dataset_hf("Teklia/IAM-line")
    return ds


class Dataset:
    def __init__(self, hf_data, img_size=(2000, 128)):
        self.data = hf_data
        self.transform = T.Compose(
            [
                T.Resize(img_size, T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(0.5, 0.5),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]["image"]
        img = self.transform(img)
        label = self.data[idx]["text"]
        return img, label


class Tokenizer:
    def __init__(self, max_label_length=100):
        self.d = {"|blank|": 0}
        banned_symbols = ["\t", "\n", "\x0b", "\x0c", "\r"]
        symbols = string.printable
        for c in symbols:
            if c not in banned_symbols:
                self.d[c] = len(self.d)

        # self.d["<eos>"] = len(self.d)

        self.inv_d = {v: k for (k, v) in self.d.items()}

        self.max_label_length = max_label_length

    def __len__(self):
        return len(self.d)

    def tokenize(self, labels):
        tensor_label = [self.d[c] for c in labels]
        # tensor_label.append(self.d["<eos>"])
        if self.max_label_length is not None:
            while len(tensor_label) < self.max_label_length:
                tensor_label.append(0)
            tensor_label = tensor_label[: self.max_label_length]
        return torch.LongTensor(tensor_label)

    def decode(self, tokens):
        return "".join([self.inv_d[token.item()] for token in tokens])


def ctc_greedy_decode(indices, blank_index=0):
    res = []
    for i in range(len(indices)):
        collapsed = []
        prev = None
        for char in indices[i]:
            if char != prev:
                collapsed.append(char)
                prev = char
        collapsed = [c for c in collapsed if c != blank_index]
        res.append(collapsed)
    return res

from collections import Counter
from typing import List, Set

import numpy as np
import torch
from torch.utils import data


class Corpus:
    def __init__(self, text_data: List[str], nickname_set: Set[str]):
        self.text_data = text_data
        self.nickname_set = nickname_set
        word_counter = Counter(self.text_data)
        sorted_vocab = sorted(word_counter, key=word_counter.get, reverse=True)
        self.table_int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
        self.table_vocab_to_int = {k: w for w, k in self.table_int_to_vocab.items()}

    def word_to_int(self, arr: List[str]) -> List[int]:
        return list(map(lambda x: self.table_vocab_to_int[x], arr))

    def int_to_word(self, arr: List[int]) -> List[str]:
        return list(map(lambda x: self.table_int_to_vocab[x], arr))

    def dumps(self) -> dict:
        return {"text_data": self.text_data, "nickname_set": list(self.nickname_set)}

    @classmethod
    def loads(cls, json_dict: dict):
        return cls(json_dict["text_data"], set(json_dict["nickname_set"]))


class Dataset(data.Dataset):
    def __init__(self, corpus: Corpus, seq_size: int):
        self.seq_count = int(len(corpus.text_data) / seq_size)
        self.seq_size = seq_size
        text_data = corpus.text_data[:self.seq_size * self.seq_count]
        text_data = corpus.word_to_int(text_data)
        shifted_data = []
        shifted_data[:-1] = text_data[1:]
        shifted_data.append(text_data[0])
        self.input_data = torch.from_numpy(np.array(text_data).reshape(self.seq_count, -1))
        self.output_data = torch.from_numpy(np.array(shifted_data).reshape(self.seq_count, -1))

    def __getitem__(self, item):
        return self.input_data[item], self.output_data[item]

    def __len__(self):
        return self.seq_count

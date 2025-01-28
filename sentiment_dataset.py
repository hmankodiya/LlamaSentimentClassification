from typing import Union, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


def tokenize_text(
    text_samples: Union[str, List[str]],
    tokenizer,
    max_length=None,
    truncation=True,
    use_encode=True,
    padding=True,
    return_tensors=None,
):
    if use_encode:
        return tokenizer.encode(
            text_samples,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_tensors=return_tensors,
        )

    return tokenizer(
        text_samples,
        truncation=truncation,
        max_length=max_length,
        padding=padding,
        return_tensors=return_tensors,
    )


def load_sentiment_dataset(
    dataset_path: str,
    dataset_language: str = "all",
    split_type: str = "train",
    sub_split_size: float = None,
):
    dataset = load_dataset(dataset_path, dataset_language, split=split_type)
    if isinstance(sub_split_size, float) and 0.0 < sub_split_size <= 1.0:
        dataset = dataset.train_test_split(
            test_size=sub_split_size, stratify_by_column="label"
        )["test"]

    return dataset


_INDEX2LABEL = {0: "positive", 1: "neutral", 2: "negative"}
_LABEL2INDEX = {value: key for key, value in _INDEX2LABEL.items()}


class SentimentDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        dataset_language,
        tokenizer,
        split_type,
        return_tensors=None,
        return_dict=False,
        **kwargs
    ):
        self.kwargs = kwargs

        self.dataset_path = dataset_path
        self.dataset_language = dataset_language
        self.split_type = split_type
        self.dataset = load_sentiment_dataset(
            dataset_path,
            dataset_language,
            split_type,
            sub_split_size=self.kwargs.pop("sub_split_size", None),
        )
        self.tokenizer = tokenizer
        self.return_tensors = return_tensors
        self.return_dict = return_dict
        self.index2label = _INDEX2LABEL
        self.label2index = _LABEL2INDEX

    def __len__(self):
        # return 10
        return len(self.dataset)

    def __getitem__(self, index):
        items = self.dataset[index]
        text, labels = items["text"], items["label"]

        input_ids = tokenize_text(
            text,
            self.tokenizer,
            truncation=self.kwargs.get("truncation", True),
            max_length=self.kwargs.get("max_length", 1024),
            padding=self.kwargs.get("padding", True),
            use_encode=True,
            return_tensors=self.return_tensors,
        )
        if self.return_tensors == "pt":
            labels = torch.tensor(labels)

        elif self.return_tensors == "np":
            labels = np.array(labels)

        if self.return_dict:
            return dict(input_ids=input_ids, labels=labels)

        return (input_ids, labels)

import os
from typing import Callable

import pandas as pd
from transformers.tokenization_auto import AutoTokenizer


def get_sentences(data: pd.DataFrame):
    labeled_sentences = (
        data.groupby("sentence_id")
        .apply(
            lambda s: [
                (w, t)
                for w, t in zip(
                    s["word"].values.tolist(),
                    s["label"].values.tolist(),
                )
            ]
        )
        .values.tolist()
    )

    sentences = [[elt[0] for elt in sentence] for sentence in labeled_sentences]
    labels = [[elt[1] for elt in sentence] for sentence in labeled_sentences]

    return sentences, labels


def remove_bio(label: str):
    if label.startswith("B-") or label.startswith("I-"):
        return label[2:]
    return label


def get_explainer_report_template() -> str:
    current_dir = os.path.dirname(os.path.realpath(__file__))

    with open(
        os.path.join(current_dir, "assets", "ner_explainer_report_template.html"), "r"
    ) as f:
        report_template = f.read()

    return report_template


def tokenize_and_keep_words(sentence: str, tokenizer_func: Callable):
    words = sentence.split(" ")
    tokens_and_words = [
        (token, word) for word in words for token in tokenizer_func(word)
    ]

    return tokens_and_words

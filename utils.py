from itertools import groupby

import pandas as pd


def split_str_into_words(sentence: str):
    for k, g in groupby(enumerate(sentence), lambda x: x[1].isalnum()):
        word_split = list(g)
        if k:
            word_str = "".join([elt[1] for elt in word_split])
            yield word_str
        else:
            for char in word_split:
                if not char[1].isspace():
                    yield char[1]


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

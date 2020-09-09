import pandas as pd


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

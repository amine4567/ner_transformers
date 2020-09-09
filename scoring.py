import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from seqeval.metrics.sequence_labeling import get_entities
import portion

from utils import remove_bio

logger = logging.getLogger(__name__)


def entity_interval_size(ent: portion.interval.Interval):
    return ent.upper - ent.lower + 1


def compute_partial_scores_by_tag(
    real_tags_by_sentence: List[List[str]],
    pred_tags_by_sentence: List[List[str]],
    desired_tag: str,
) -> Tuple[float, float, float, float, float, float]:
    all_tags = set([elt for ll in real_tags_by_sentence for elt in ll])

    try:
        assert "B-" + desired_tag in all_tags or "I" + desired_tag in all_tags
    except AssertionError:
        raise ValueError(f"'{desired_tag}' is not a viable tag.")

    real_tags_by_sentence = list(map(np.array, real_tags_by_sentence))
    pred_tags_by_sentence = list(map(np.array, pred_tags_by_sentence))

    for i, sentence_labels in enumerate(real_tags_by_sentence):
        real_tags_by_sentence[i][~np.char.endswith(sentence_labels, desired_tag)] = "O"

    for i, sentence_labels in enumerate(pred_tags_by_sentence):
        pred_tags_by_sentence[i][~np.char.endswith(sentence_labels, desired_tag)] = "O"

    real_tags_by_sentence = list(map(list, real_tags_by_sentence))
    pred_tags_by_sentence = list(map(list, pred_tags_by_sentence))

    real_entities = set(get_entities(real_tags_by_sentence))
    pred_entities = set(get_entities(pred_tags_by_sentence))

    all_tp = 0.0
    all_fp = 0.0
    all_fn = 0.0

    # We get entities that match exactly between reality and prediction
    full_tp_ents = real_entities & pred_entities

    all_tp += len(full_tp_ents)

    real_entities_rest = real_entities - full_tp_ents
    pred_entities_rest = pred_entities - full_tp_ents

    # We process the rest to get potentiel partial matches
    real_entities_rest = [portion.closed(elt[1], elt[2]) for elt in real_entities_rest]
    pred_entities_rest = [portion.closed(elt[1], elt[2]) for elt in pred_entities_rest]

    processed_pred_ents = []
    for real_ent in real_entities_rest:
        intersected_pred_ents = [
            {
                "pred_ent": pred_entities_rest[i],
                "intersection": elt,
                "pred_ent_size": entity_interval_size(pred_entities_rest[i]),
                "intersection_size": entity_interval_size(elt),
            }
            for i, elt in enumerate(
                list(map(lambda x: real_ent & x, pred_entities_rest))
            )
            if not elt.empty
        ]
        if len(intersected_pred_ents) != 0:
            real_ent_size = entity_interval_size(real_ent)
            ent_tp = (
                sum([elt["intersection_size"] for elt in intersected_pred_ents])
                / real_ent_size
            )
            ent_fn = 1.0 - ent_tp
            ent_fp = sum(
                [
                    (elt["pred_ent_size"] - elt["intersection_size"])
                    / elt["pred_ent_size"]
                    for elt in intersected_pred_ents
                    if elt["pred_ent"] not in processed_pred_ents
                ]
            )

            processed_pred_ents.extend(
                [
                    elt["pred_ent"]
                    for elt in intersected_pred_ents
                    if elt["pred_ent"] not in processed_pred_ents
                ]
            )
        else:
            ent_tp = 0.0
            ent_fn = 1.0
            ent_fp = 0.0

        all_tp += ent_tp
        all_fn += ent_fn
        all_fp += ent_fp

    all_fp += len(set(pred_entities_rest) - set(processed_pred_ents))

    partial_recall = all_tp / (all_tp + all_fn)
    partial_precision = all_tp / (all_tp + all_fp)
    partial_f1_score = (
        2 * partial_precision * partial_recall / (partial_precision + partial_recall)
    )

    return partial_precision, partial_recall, partial_f1_score, all_tp, all_fn, all_fp


def make_partial_match_report(
    real_tags_by_sentence: List[List[str]],
    pred_tags_by_sentence: List[List[str]],
) -> pd.DataFrame:
    all_raw_tags = list(
        set(map(remove_bio, set([elt for ll in pred_tags_by_sentence for elt in ll])))
        - set("O")
    )

    report = pd.DataFrame(
        data=[
            compute_partial_scores_by_tag(
                real_tags_by_sentence, pred_tags_by_sentence, tag
            )
            for tag in all_raw_tags
        ],
        columns=["precision", "recall", "f1_score", "TP", "FN", "FP"],
        index=all_raw_tags,
    )

    performance_values = report[["TP", "FN", "FP"]].sum()
    micro_precision = performance_values.TP / (
        performance_values.TP + performance_values.FP
    )
    micro_recall = performance_values.TP / (
        performance_values.TP + performance_values.FN
    )
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

    report = report[["precision", "recall", "f1_score"]]
    report.loc["micro_avg"] = (micro_precision, micro_recall, micro_f1)
    report.loc["macro_avg"] = report.mean()

    return report

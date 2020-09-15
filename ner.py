from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
import torch
from tqdm import trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from keras.preprocessing.sequence import pad_sequences
from seqeval.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax

from utils import get_sentences
from scoring import (
    make_partial_match_report,
    make_tokenwise_scores_report,
    compute_semeval_scores,
)


@dataclass
class NERArgs:
    do_lower_case: bool = False
    max_sequence_len = None
    batch_size: int = 32
    pad_token_value: str = "PAD"
    epochs: int = 5
    max_grad_norm: float = 1.0
    no_decay_params: Tuple = ("bias", "gamma", "beta")
    weight_decay_rate: float = 0.01
    learning_rate: float = 3e-5
    adam_epsilon: float = 1e-8


class NERModel:
    def __init__(
        self, model_name: str, labels: List[str], ner_args: Optional[NERArgs] = None
    ):
        self.model_name = model_name

        self.ner_args = ner_args if ner_args is not None else NERArgs()

        self.label_values = labels + [self.ner_args.pad_token_value]
        self.labels = {t: i for i, t in enumerate(self.label_values)}

        self.set_raw_labels_data()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, do_lower_case=self.ner_args.do_lower_case
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.labels),
            output_attentions=False,
            output_hidden_states=False,
        )
        self.model.cuda()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_raw_labels_data(self):
        labels_ids_df = (
            pd.DataFrame(
                index=self.labels.keys(),
                columns=["id"],
                data=self.labels.values(),
            )
            .reset_index()
            .rename(columns={"index": "bio_tag"})
        )
        labels_ids_df["tag_raw"] = (
            labels_ids_df.bio_tag.str.split("-")
            .str.get(1)
            .fillna(labels_ids_df.bio_tag)
        )

        self.raw_labels_ids = {
            raw_tag: tmp_df.id.values
            for raw_tag, tmp_df in labels_ids_df.groupby("tag_raw")
            if raw_tag != self.ner_args.pad_token_value
        }
        self.raw_labels_values = list(self.raw_labels_ids.keys())

    def tokenize_and_preserve_labels(self, sentence, text_labels):
        tokenized_sentence = []
        labels = []

        for word, label in zip(sentence, text_labels):
            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)

        return tokenized_sentence, labels

    def pad_tokens(
        self, tokens: List[str], tokens_labels: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        input_ids = torch.tensor(
            pad_sequences(
                [self.tokenizer.convert_tokens_to_ids(txt) for txt in tokens],
                maxlen=self.ner_args.max_sequence_len,
                dtype="long",
                value=0.0,
                truncating="post",
                padding="post",
            )
        )

        attention_masks = torch.tensor(
            [[float(i != 0.0) for i in ii] for ii in input_ids]
        )

        if tokens_labels:
            padded_labels = torch.tensor(
                pad_sequences(
                    [[self.labels.get(ll) for ll in lab] for lab in tokens_labels],
                    maxlen=self.ner_args.max_sequence_len,
                    value=self.labels[self.ner_args.pad_token_value],
                    padding="post",
                    dtype="long",
                    truncating="post",
                )
            )

            return input_ids, attention_masks, padded_labels

        return input_ids, attention_masks, None

    def prepare_dataloader(self, text_data: pd.DataFrame):
        sentences, labels = get_sentences(text_data)
        tokenized_texts_and_labels = [
            self.tokenize_and_preserve_labels(sent, labs)
            for sent, labs in zip(sentences, labels)
        ]

        tokenized_texts = [
            token_label_pair[0] for token_label_pair in tokenized_texts_and_labels
        ]
        tokenized_labels = [
            token_label_pair[1] for token_label_pair in tokenized_texts_and_labels
        ]

        input_ids, attention_masks, padded_labels = self.pad_tokens(
            tokenized_texts, tokenized_labels
        )

        tensor = TensorDataset(input_ids, attention_masks, padded_labels)
        sampler = RandomSampler(tensor)
        dataloader = DataLoader(
            tensor, sampler=sampler, batch_size=self.ner_args.batch_size
        )

        return dataloader

    def fit(
        self,
        train_data: pd.DataFrame,
        eval_data: pd.DataFrame = None,
        eval_exact_match: bool = True,
        eval_partial_match: bool = False,
        eval_tokenwise_scores: bool = False,
        eval_semeval_scores: bool = False,
    ):
        """[summary]

        :Parameters:
            - train_data (pd.DataFrame): expected to have three columns (sentence_id, word, label)
            - eval_data (pd.DataFrame, optional): same as train_data. Defaults to None.
        """
        tr_dataloader = self.prepare_dataloader(train_data)

        param_optimizer = list(self.model.named_parameters())
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if not any(nd in n for nd in self.ner_args.no_decay_params)
                ],
                "weight_decay_rate": self.ner_args.weight_decay_rate,
            },
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if any(nd in n for nd in self.ner_args.no_decay_params)
                ],
                "weight_decay_rate": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.ner_args.learning_rate,
            eps=self.ner_args.adam_epsilon,
        )

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(tr_dataloader) * self.ner_args.epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        ## Store the average loss after each epoch so we can plot them.
        self.train_loss_values: List = []
        self.validation_loss_values: List = []

        for _ in trange(self.ner_args.epochs, desc="Epoch"):
            # ========================================
            #               Training
            # ========================================
            # Perform one full pass over the training set.

            # Put the model into training mode.
            self.model.train()
            # Reset the total loss for this epoch.
            total_loss = 0

            # Training loop
            for batch in tr_dataloader:
                # add batch to gpu
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                # Always clear any previously calculated gradients before performing a backward pass.
                self.model.zero_grad()
                # forward pass
                # This will return the loss (rather than the model output)
                # because we have provided the `labels`.
                outputs = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )
                # get the loss
                loss = outputs[0]
                # Perform a backward pass to calculate the gradients.
                loss.backward()
                # track train loss
                total_loss += loss.item()
                # Clip the norm of the gradient
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(),
                    max_norm=self.ner_args.max_grad_norm,
                )
                # update parameters
                optimizer.step()
                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(tr_dataloader)
            print("Average train loss: {}".format(avg_train_loss))

            # Store the loss value for plotting the learning curve.
            self.train_loss_values.append(avg_train_loss)

            if eval_data is not None:
                eval_loss = self.evaluate(
                    eval_data,
                    exact_match=eval_exact_match,
                    partial_match=eval_partial_match,
                    tokenwise_scores=eval_tokenwise_scores,
                    semeval_scores=eval_semeval_scores,
                )
                self.validation_loss_values.append(eval_loss)

    def evaluate(
        self,
        eval_data: pd.DataFrame,
        exact_match: bool = True,
        partial_match: bool = False,
        tokenwise_scores: bool = False,
        semeval_scores: bool = False,
    ):
        eval_dataloader = self.prepare_dataloader(eval_data)

        # Put the model into evaluation mode
        self.model.eval()
        # Reset the validation loss for this epoch.
        eval_loss = 0.0
        predictions, true_labels = [], []
        for batch in eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                outputs = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )
            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            # Calculate the accuracy for this batch of test sentences.
            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss / len(eval_dataloader)
        print("Validation loss: {}".format(eval_loss))

        pred_tags_by_sentence = [
            [
                self.label_values[p_i]
                for p_i, l_i in zip(p, l)
                if self.label_values[l_i] != self.ner_args.pad_token_value
            ]
            for p, l in zip(predictions, true_labels)
        ]
        valid_tags_by_sentence = [
            [
                self.label_values[l_i]
                for l_i in ll
                if self.label_values[l_i] != self.ner_args.pad_token_value
            ]
            for ll in true_labels
        ]

        print(
            "Validation Accuracy: {}".format(
                accuracy_score(valid_tags_by_sentence, pred_tags_by_sentence)
            )
        )
        print(
            "Validation F1-Score: {}".format(
                f1_score(valid_tags_by_sentence, pred_tags_by_sentence)
            )
        )
        print()

        if exact_match:
            print("=== Exact match scores ===")
            print(classification_report(valid_tags_by_sentence, pred_tags_by_sentence))
            print()

        if partial_match:
            print("=== Partial match scores ===")

            partial_match_report = make_partial_match_report(
                valid_tags_by_sentence, pred_tags_by_sentence
            )
            print(partial_match_report)
            print()

        if tokenwise_scores:
            print("=== tokenwise scores ===")
            print(
                make_tokenwise_scores_report(
                    valid_tags_by_sentence, pred_tags_by_sentence
                )
            )
            print()

        if semeval_scores:
            print("=== SemEval scores ===")
            semeval_scores = compute_semeval_scores(
                valid_tags_by_sentence, pred_tags_by_sentence, self.raw_labels_values
            )
            for score_type, scores in semeval_scores.items():
                print(f"---- {score_type} ----")
                print()
                print(scores)
            print()

        print()

        return eval_loss

    def predict(
        self,
        sentences: Union[List[str], List[List[str]]],
        pre_tokenized: bool = False,
        remove_pad_token_logit: bool = True,
        by_raw_labels: bool = True,
    ):
        self.model.eval()
        sentences_tokens = []
        sentences_logits = []
        for text in sentences:
            if pre_tokenized is True:
                tokens = text
            else:
                tokens = self.tokenizer.tokenize(text)

            tokens_ids, attention_mask, _ = self.pad_tokens([tokens])

            tokens_ids = tokens_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            with torch.no_grad():
                tokens_logits = (
                    self.model(
                        tokens_ids, token_type_ids=None, attention_mask=attention_mask
                    )[0][0]
                    .cpu()
                    .numpy()
                )

            if remove_pad_token_logit:
                tokens_logits = np.delete(
                    tokens_logits,
                    obj=self.labels[self.ner_args.pad_token_value],
                    axis=1,
                )

            if by_raw_labels:
                tokens_logits = np.vstack(
                    [
                        tokens_logits[:, self.raw_labels_ids[raw_tag]].mean(axis=1)
                        for raw_tag in self.raw_labels_values
                    ]
                ).transpose()

            sentences_tokens.append(tokens)
            sentences_logits.append(tokens_logits)

        return sentences_tokens, sentences_logits

    def predict_proba(
        self,
        sentences: List[str],
        pre_tokenized: bool = False,
        remove_pad_token_logit: bool = True,
        by_raw_labels: bool = True,
    ):
        sentences_tokens, sentences_logits = self.predict(
            sentences,
            pre_tokenized=pre_tokenized,
            by_raw_labels=by_raw_labels,
            remove_pad_token_logit=remove_pad_token_logit,
        )

        sentences_probas = [
            softmax(
                logits,
                axis=1,
            )
            for logits in sentences_logits
        ]

        return sentences_tokens, sentences_probas

    def predict_proba_and_preserve_labels(self, text_data: pd.DataFrame):
        sentences, labels = get_sentences(text_data)
        tokenized_texts_and_labels = [
            self.tokenize_and_preserve_labels(sent, labs)
            for sent, labs in zip(sentences, labels)
        ]
        sentences_tokens = [elt[0] for elt in tokenized_texts_and_labels]
        sentences_true_labels = [elt[1] for elt in tokenized_texts_and_labels]

        _, sentences_probas = self.predict_proba(sentences_tokens, pre_tokenized=True)

        return sentences_tokens, sentences_probas, sentences_true_labels

    def plot_learning_curve(self):
        # Use plot styling from seaborn.
        sns.set(style="darkgrid")

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12, 6)

        # Plot the learning curve.
        plt.plot(self.train_loss_values, "b-o", label="training loss")

        if len(self.validation_loss_values) > 0:
            plt.plot(self.validation_loss_values, "r-o", label="validation loss")

        plt.title("Learning curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.show()

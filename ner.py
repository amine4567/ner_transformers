from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from transformers import (
    BertTokenizer,
    BertForTokenClassification,
    RobertaTokenizer,
    RobertaForTokenClassification,
    CamembertTokenizer,
    CamembertForTokenClassification,
    FlaubertTokenizer,
    FlaubertForTokenClassification,
    XLMRobertaTokenizer,
    XLMRobertaForTokenClassification,
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

from utils import split_str_into_words, get_sentences

model_types = {
    "bert": {
        "tokenizer": BertTokenizer,
        "model": BertForTokenClassification,
    },
    "roberta": {"tokenizer": RobertaTokenizer, "model": RobertaForTokenClassification},
    "camembert": {
        "tokenizer": CamembertTokenizer,
        "model": CamembertForTokenClassification,
    },
    "flaubert": {
        "tokenizer": FlaubertTokenizer,
        "model": FlaubertForTokenClassification,
    },
    "xlm-roberta": {
        "tokenizer": XLMRobertaTokenizer,
        "model": XLMRobertaForTokenClassification,
    },
}

models_map = {
    "bert-base-multilingual-uncased": "bert",
    "roberta-base": "roberta",
    "camembert-base": "camembert",
    "flaubert/flaubert_base_cased": "flaubert",
    "xlm-roberta-base": "xlm-roberta",
}


@dataclass
class NERArgs:
    do_lower_case: bool = False
    max_sequence_len = None
    batch_size: int = 32
    pad_token: str = "PAD"
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
        self.model_type = models_map[self.model_name]

        self.ner_args = ner_args if ner_args is not None else NERArgs()

        self.label_values = labels + [self.ner_args.pad_token]
        self.labels = {t: i for i, t in enumerate(self.label_values)}

        model_tools = model_types[self.model_type]

        self.tokenizer_class = model_tools["tokenizer"]
        self.tokenizer = self.tokenizer_class.from_pretrained(
            self.model_name, do_lower_case=self.ner_args.do_lower_case
        )

        self.model_class = model_tools["model"]

        self.model = self.model_class.from_pretrained(
            self.model_name,
            num_labels=len(self.labels),
            output_attentions=False,
            output_hidden_states=False,
        )
        self.model.cuda()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                    value=self.labels["PAD"],
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

    def fit(self, train_data: pd.DataFrame, eval_data: pd.DataFrame = None):
        """[summary]

        :Parameters:
            - train_data (pd.DataFrame): expected to have three columns (sentence_id, word, label)
            - eval_data (pd.DataFrame, optional): same as train_data. Defaults to None.
        """
        tr_dataloader = self.prepare_dataloader(train_data)
        eval_dataloader = self.prepare_dataloader(eval_data)

        ############## TODO: dirty
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
        self.loss_values: List = []
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
            self.loss_values.append(avg_train_loss)

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

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
            self.validation_loss_values.append(eval_loss)
            print("Validation loss: {}".format(eval_loss))
            pred_tags = [
                self.label_values[p_i]
                for p, l in zip(predictions, true_labels)
                for p_i, l_i in zip(p, l)
                if self.label_values[l_i] != "PAD"
            ]
            valid_tags = [
                self.label_values[l_i]
                for ll in true_labels
                for l_i in ll
                if self.label_values[l_i] != "PAD"
            ]
            print(
                "Validation Accuracy: {}".format(accuracy_score(valid_tags, pred_tags))
            )
            print("Validation F1-Score: {}".format(f1_score(valid_tags, pred_tags)))
            print("Validation classification report:")
            print(classification_report(valid_tags, pred_tags))
            print()
        ############ TODO: dirty

    def predict(self, text: str):
        self.model.eval()

        tokens = self.tokenizer.tokenize(text)
        tokens_ids, attention_mask, _ = self.pad_tokens([tokens])

        tokens_ids = tokens_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            prediction_outputs = self.model(
                tokens_ids, token_type_ids=None, attention_mask=attention_mask
            )

        return prediction_outputs

    def plot_learning_curve(self):
        # Use plot styling from seaborn.
        sns.set(style="darkgrid")

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12, 6)

        # Plot the learning curve.
        plt.plot(self.loss_values, "b-o", label="training loss")
        plt.plot(self.validation_loss_values, "r-o", label="validation loss")

        plt.title("Learning curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.show()

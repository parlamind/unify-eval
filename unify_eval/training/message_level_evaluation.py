import json
from typing import Dict, Set

import numpy as np
from tensorboardX import SummaryWriter

from unify_eval.model.mixins.classification import Classifier, MessageLevelClassifier
from unify_eval.model.types import Label
from unify_eval.training.callback import PlottingCallback
from unify_eval.utils.corpus import normalize, group
from unify_eval.utils.load_data import KeyedBatchLoader


def load_messages(path: str,
                  text_kw: str = "clauses",
                  normalize_text: bool = True,
                  include_subject:bool=True) -> Dict[str, np.ndarray]:
    with open(path, "r") as f:
        corpus = json.load(f)
    f_normalize = lambda x: normalize(x) if normalize_text else x
    maybe_subject = lambda message : [message["subject"]] if include_subject else []
    return {
        text_kw: np.array(
            [[f_normalize(clause) for clause in maybe_subject(message) + message["clauses"]] for message in corpus]),
        "labels": np.array([message["goldIndexIds"] for message in corpus])
    }


class MessageLevelEvaluation(PlottingCallback):
    def __init__(self,
                 folder_path: str,
                 relative_path: str,
                 data_loader: KeyedBatchLoader,
                 junk_label: Label = None,
                 junk_threshold: float = None,
                 labels_to_evaluate: Set[int] = None,
                 text_kw: str = "clauses",
                 minibatch_size: int = 16):
        super().__init__(folder_path,relative_path)
        self.junk_label = junk_label
        self.junk_threshold = junk_threshold
        self.data_loader = data_loader
        self.labels_to_evaluate = labels_to_evaluate
        self.minibatch_size = minibatch_size
        self.text_kw = text_kw

    def __call__(self, model: Classifier, i_minibatch: int, iteration: int, *args, **kwargs):

        print("isolated evaluation ...")
        print(f"labels to evaluate: {self.labels_to_evaluate}")
        y_true_per_message = []
        y_pred_per_message = []

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        messages = []

        for i_minibatch, minibatch in enumerate(self.data_loader.yield_minibatches(minibatch_size=self.minibatch_size)):
            message_lengths = [len(message_clauses) for message_clauses in minibatch[self.text_kw]]
            messages.extend(minibatch[self.text_kw])

            # flatten clauses
            clauses = [clause for message_clauses in minibatch.pop(self.text_kw) for clause in message_clauses]

            # add true labels (filter against excluded labels before)
            y_true_per_message.extend([list({label for label in message_labels if
                                             self.labels_to_evaluate and label in self.labels_to_evaluate})
                                       for message_labels in minibatch.pop("labels")])

            flat_predictions = model.predict_label(**{self.text_kw: clauses}, **minibatch, **kwargs,
                                                   junk_label_index=model.label_mapper.actuallabel2index[
                                                       self.junk_label] if self.junk_label else None,
                                                   junk_threshold=self.junk_threshold if self.junk_threshold else None)

            # group predicted labels by message again
            counter = 0
            for message_length in message_lengths:
                predictions_per_message = flat_predictions[counter:+ counter + message_length]
                y_pred_per_message.append(
                    list(set(label for label in predictions_per_message if label != self.junk_label)))

                counter += message_length

        self.data_loader.reset()

        assert len(messages) == len(y_true_per_message)
        for message, y_true, y_pred in zip(messages, y_true_per_message, y_pred_per_message):
            print(f"true {y_true} pred {[y for y in y_pred if y != self.junk_label]} {message}")
            true_positives += len([y for y in y_true if y in y_pred])
            false_positives += len([y for y in y_pred if y not in y_true])
            false_negatives += len([y for y in y_true if y not in y_pred])

        precision = (true_positives / (true_positives + false_positives)) if (
                                                                                     true_positives + false_positives) != 0 else 0
        recall = (true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) != 0 else 0
        f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) != 0 else 0
        print(f"precision {precision}")
        print(f"recall {recall}")
        print(f"f1 {f1}")
        self.writer.add_scalar(tag="isolated_precision", scalar_value=precision, global_step=self.global_step)
        self.writer.add_scalar(tag="isolated_recall", scalar_value=recall, global_step=self.global_step)
        self.writer.add_scalar(tag="isolated_f1", scalar_value=f1, global_step=self.global_step)
        return model


class MessageLevelIsolatedEvaluation(PlottingCallback):
    def __init__(self,
                 folder_path: str,
                 data_loader: KeyedBatchLoader,
                 junk_label: Label = None,
                 junk_threshold: float = None,
                 labels_to_evaluate: Set[int] = None):
        super().__init__(folder_path)
        self._writer = SummaryWriter(logdir=folder_path)
        self.junk_label = junk_label
        self.junk_threshold = junk_threshold
        self.data_loader = data_loader
        self.labels_to_evaluate = labels_to_evaluate

    def __call__(self, model: MessageLevelClassifier, i_minibatch: int, iteration: int, *args, **kwargs):

        print("isolated evaluation ...")
        y_true_per_message = []
        y_pred_per_message = []

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i_message, message in enumerate(self.data_loader.yield_minibatches(minibatch_size=1)):
            clauses = message.pop("clauses")[0]
            labels = message.pop("labels")[0]
            y_true_per_message.append(list(set(int(label) for label in labels if
                                               self.labels_to_evaluate and label in self.labels_to_evaluate)))
            y_pred_per_message.append(list(set(int(label) for labels_per_message in model.predict_labels(
                clauses=[clauses],
                **message,
                **kwargs,
                junk_label_index=model.label_mapper.actuallabel2index[self.junk_label] if self.junk_label else None,
                junk_threshold=self.junk_threshold if self.junk_threshold else None) for label in labels_per_message if
                                               int(label) != self.junk_label)))

        for y_true, y_pred in zip(y_true_per_message, y_pred_per_message):
            true_positives += len([y for y in y_true if y in y_pred])
            false_positives += len([y for y in y_pred if y not in y_true])
            false_negatives += len([y for y in y_true if y not in y_pred])

        self.data_loader.reset()

        precision = (true_positives / (true_positives + false_positives)) if (
                                                                                     true_positives + false_positives) != 0 else 0
        recall = (true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) != 0 else 0
        f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) != 0 else 0
        self._writer.add_scalar(tag="isolated_precision", scalar_value=precision, global_step=iteration)
        self._writer.add_scalar(tag="isolated_recall", scalar_value=recall, global_step=iteration)
        self._writer.add_scalar(tag="isolated_f1", scalar_value=f1, global_step=iteration)
        return model


class SaveWrongMessagePredictions(PlottingCallback):
    def __init__(self,
                 folder_path: str,
                 data_loader: KeyedBatchLoader,
                 junk_label: Label = None,
                 junk_threshold: float = None,
                 labels_to_evaluate: Set[int] = None,
                 text_kw: str = "clauses"):
        super().__init__(folder_path)
        self._writer = SummaryWriter(logdir=folder_path)
        self.junk_label = junk_label
        self.junk_threshold = junk_threshold
        self.data_loader = data_loader
        self.labels_to_evaluate = labels_to_evaluate
        self.text_kw = text_kw

    def __call__(self, model: Classifier, i_minibatch: int, iteration: int, *args, **kwargs):
        print("saving wrong predictions for message level prediction...")
        y_true_per_message = []
        y_pred_per_message = []
        messages = []

        for i_message, message in enumerate(self.data_loader.yield_minibatches(minibatch_size=1)):
            texts = message.pop(self.text_kw)[0]
            labels = message.pop("labels")[0]

            y_true = list(set(int(label) for label in labels if
                              self.labels_to_evaluate and label in self.labels_to_evaluate))

            y_pred = list(set(int(label) for label in model.predict_label(
                **{self.text_kw: texts},
                **message,
                **kwargs,
                junk_label_index=model.label_mapper.actuallabel2index[self.junk_label] if self.junk_label else None,
                junk_threshold=self.junk_threshold if self.junk_threshold else None) if int(label) != self.junk_label))

            if y_true != y_pred:
                y_true_per_message.append(y_true)
                y_pred_per_message.append(y_pred)
                messages.append("  \n".join(texts))

        self.data_loader.reset()

        predictions = [(tuple(ys_true), tuple(ys_pred)) for ys_true, ys_pred in
                       zip(y_true_per_message, y_pred_per_message)]
        grouped = group(messages, predictions)
        for prediction, messages in grouped.items():
            merged_messages = "\n___\n".join(messages)
            self._writer.add_text(text_string=merged_messages,
                                  tag=f"message_level true {prediction[0]} prediction {prediction[1]}",
                                  global_step=iteration)
        return model

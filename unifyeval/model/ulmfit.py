import json
import os
from dataclasses import dataclass
from typing import Dict, List, Set
from typing import Iterable

import numpy as np
import pandas as pd
from bpemb import BPEmb
from cleantext import clean
from fastai.text import TextClasDataBunch, AWD_LSTM, \
    text_classifier_learner, TextLMDataBunch, language_model_learner, RNNLearner, \
    DatasetType, Vocab, load_learner, awd_lstm_lm_config, transform, awd_lstm_clas_config
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from unifyeval.model.mixins.classification import Classifier
from unifyeval.model.types import Tensor, Label
from unifyeval.training.message_level_evaluation import load_messages
from unifyeval.utils.corpus import Corpus
from unifyeval.utils.label_mapper import LabelMapper
from unifyeval.utils.load_data import KeyedBatchLoader
from unifyeval.utils.load_message_data import yield_normalized_batched_texts

"""
NOT TO BE INCLUDED IN FINAL PUBLIC API
"""

class ULMFiTModel(Classifier):
    """
    LABEL INFERENCE ONLY BECAUSE FASTAI HAS A TERRIBLE API
    """

    def __init__(self, label_mapper: LabelMapper, rnn_learner: RNNLearner, lang: str):
        super().__init__(label_mapper)

        self.rnn_learner: RNNLearner = rnn_learner
        self.lang = lang
        if self.lang == "german":
            self.bpemb_de = BPEmb(lang="de", vs=25000, dim=300)

    def predict_label_probabilities(self, **kwargs) -> Tensor:
        texts = kwargs["clauses"]
        if self.lang == "german":
            texts = [self.bpemb_de.encode_ids_with_bos_eos(clean(t, stp_lang='german')) for t in texts]
        self.rnn_learner.data.add_test(items=texts)
        preds, _ = self.rnn_learner.get_preds(ds_type=DatasetType.Test)
        raw_probs = np.vstack(preds)

        return raw_probs

    def get_logits(self, **kwargs) -> Tensor:
        raise NotImplementedError()

    def train(self, **kwargs) -> "ULMFiTModel":
        raise NotImplementedError()

    def get_loss(self, **kwargs) -> Dict[str, Tensor]:
        all_labels = self.label_mapper.all_indices
        y_true = self.label_mapper.map_to_indices(kwargs["labels"])
        y_pred = self.predict_label_probabilities(texts=kwargs["texts"])
        return {
            "cross_entropy": log_loss(y_true=y_true,
                                      y_pred=y_pred,
                                      labels=all_labels)
        }

    @staticmethod
    def from_components(**kwargs) -> "ULMFiTModel":
        return ULMFiTModel(**kwargs)

    def get_components(self) -> dict:
        return dict(label_mapper=self.label_mapper,
                    rnn_learner=self.rnn_learner,
                    lang=self.lang)

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        return dict((name, parameter.detach().numpy())
                    for name, parameter in self.rnn_learner.model.named_parameters())


class IsolatedEvaluation:
    """
    copy of the evaluation callback, but instead of the model it returns the evaluation results as a dict
    """

    def __init__(self,
                 data_loder: KeyedBatchLoader,
                 junk_label: Label = None,
                 junk_threshold: float = None,
                 labels_to_evaluate: Set[int] = None,
                 text_kw: str = "clauses",
                 minibatch_size: int = 16):
        self.junk_label = junk_label
        self.junk_threshold = junk_threshold
        self.data_loader = data_loder
        self.labels_to_evaluate = labels_to_evaluate
        self.minibatch_size = minibatch_size
        self.text_kw = text_kw

    def __call__(self, model: ULMFiTModel, n_lm: int, n_clf: int,
                 add_loss_acc: bool = False, *args, **kwargs):

        print("isolated evaluation ...")
        y_true_per_message = []
        y_pred_per_message = []

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i_minibatch, minibatch in enumerate(self.data_loader.yield_minibatches(minibatch_size=self.minibatch_size)):
            message_lengths = [len(message_clauses) for message_clauses in minibatch[self.text_kw]]

            # flatten clauses
            clauses = [clause for message_clauses in minibatch.pop(self.text_kw) for clause in message_clauses]

            # add true labels (filter against excluded labels before)
            y_true_per_message.extend([list({int(label) for label in message_labels if
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
                    list(set(int(label) for label in predictions_per_message if int(label) != self.junk_label)))

                counter += message_length

        self.data_loader.reset()

        for y_true, y_pred in zip(y_true_per_message, y_pred_per_message):
            true_positives += len([y for y in y_true if y in y_pred])
            false_positives += len([y for y in y_pred if y not in y_true])
            false_negatives += len([y for y in y_true if y not in y_pred])

        precision = (true_positives / (true_positives + false_positives)) if (
                                                                                     true_positives + false_positives) != 0 else 0
        recall = (true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) != 0 else 0
        f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) != 0 else 0

        if add_loss_acc:
            train_loss, train_accuracy = model.rnn_learner.validate(dl=model.rnn_learner.data.train_dl)
            train_loss = float(train_loss)
            train_accuracy = float(train_accuracy.detach().item())

            valid_loss, valid_accuracy = model.rnn_learner.validate(dl=model.rnn_learner.data.valid_dl)
            valid_loss = float(valid_loss)
            valid_accuracy = float(valid_accuracy.detach().item())

            return dict(training_loss=train_loss,
                        train_accuracy=train_accuracy,
                        validation_loss=valid_loss,
                        validation_accuracies=valid_accuracy,
                        precision=precision,
                        recall=recall,
                        f1=f1,
                        n_lm=int(n_lm),
                        n_clf=int(n_clf)
                        )
        return dict(
            precision=precision,
            recall=recall,
            f1=f1,
            n_lm=int(n_lm),
            n_clf=int(n_clf)
        )


@dataclass
class TrainedLMConfig:
    model_paths: List[str]
    lm_iterations: List[int]
    training_losses: List[float]
    validation_losses: List[float]
    training_accuracies: List[float]
    validation_accuracies: List[float]
    vocab: Vocab


@dataclass
class TrainedClfConfig:
    model_paths: List[str]
    clf_iterations: List[int]
    training_losses: List[float]
    validation_losses: List[float]
    training_accuracies: List[float]
    validation_accuracies: List[float]


def train_ulmfit_language_models_english(texts: Iterable[str],
                                         X_test: Iterable[str],
                                         Y_test: Iterable[str],
                                         path: str = "/tmp",
                                         lm_iterations: int = 2,
                                         encoder_name: str = "encoder") -> TrainedLMConfig:
    """
    fine-tune pretrained model from scratch
    :param texts:
    :param X_test:
    :param Y_test:
    :param path:
    :param lm_iterations:
    :param encoder_to_load:
    :param encoder_name:
    :return:
    """

    def get_lr(iteration: int) -> float:
        """
        map iteration to learning rate
        :param iteration: 
        :return: 
        """
        if iteration == 0:
            return 1e-2
        return 1e-3

    training_data_lm = pd.DataFrame.from_dict(dict(text=texts, label=[0 for _ in texts]))
    val_data = pd.DataFrame.from_dict(dict(text=X_test, label=Y_test))
    data_lm = TextLMDataBunch.from_df(path=path,
                                      train_df=training_data_lm, valid_df=val_data, text_cols=["text"])
    learner = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
    iterations = []
    paths = []
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []

    print("evaluating original model")
    path = f"{encoder_name}_{0}"
    paths.append(path)
    iterations.append(0)
    train_loss, train_accuracy = learner.validate(dl=learner.data.train_dl)
    train_loss = float(train_loss)
    train_accuracy = float(train_accuracy.detach().item())

    valid_loss, valid_accuracy = learner.validate(dl=learner.data.valid_dl)
    valid_loss = float(valid_loss)
    valid_accuracy = float(valid_accuracy.detach().item())
    training_losses.append(train_loss)
    validation_losses.append(valid_loss)
    training_accuracies.append(train_accuracy)
    validation_accuracies.append(valid_accuracy)

    learner.save_encoder(path)

    print("starting very first training cycle with partially frozen model")
    learner.fit_one_cycle(1, get_lr(iteration=iterations[0]))
    path = f"{encoder_name}_{1}"
    paths.append(path)
    iterations.append(1)
    train_loss, train_accuracy = learner.validate(dl=learner.data.train_dl)
    train_loss = float(train_loss)
    train_accuracy = float(train_accuracy.detach().item())

    valid_loss, valid_accuracy = learner.validate(dl=learner.data.valid_dl)
    valid_loss = float(valid_loss)
    valid_accuracy = float(valid_accuracy.detach().item())
    training_losses.append(train_loss)
    validation_losses.append(valid_loss)
    training_accuracies.append(train_accuracy)
    validation_accuracies.append(valid_accuracy)

    learner.save_encoder(path)
    print("unfreezing")
    learner.unfreeze()
    learner.save_encoder(path)

    for iteration in range(lm_iterations - 1):
        print(f"lm training cycle {iteration}")
        learner.fit_one_cycle(1, get_lr(iteration))
        path = f"{encoder_name}_{iteration + 2}"
        paths.append(path)
        iterations.append(iteration + 2)
        train_loss, train_accuracy = learner.validate(dl=learner.data.train_dl)
        train_loss = float(train_loss)
        train_accuracy = float(train_accuracy.detach().item())

        valid_loss, valid_accuracy = learner.validate(dl=learner.data.valid_dl)
        valid_loss = float(valid_loss)
        valid_accuracy = float(valid_accuracy.detach().item())
        training_losses.append(train_loss)
        validation_losses.append(valid_loss)
        training_accuracies.append(train_accuracy)
        validation_accuracies.append(valid_accuracy)
        learner.save_encoder(path)
    return TrainedLMConfig(model_paths=paths,
                           vocab=data_lm.train_ds.vocab,
                           lm_iterations=iterations,
                           training_accuracies=training_accuracies,
                           validation_accuracies=validation_accuracies,
                           training_losses=training_losses,
                           validation_losses=validation_losses)


def train_ulmfit_language_models_german(texts: Iterable[str],
                                        X_test: Iterable[str],
                                        Y_test: Iterable[str],
                                        path_to_encoder: str,
                                        path: str,
                                        lm_iterations: int = 2,
                                        encoder_name: str = "encoder") -> TrainedLMConfig:
    """
    fine-tune pretrained model from scratch
    :param texts:
    :param X_test:
    :param Y_test:
    :param path:
    :param lm_iterations:
    :param encoder_to_load:
    :param encoder_name:
    :return:
    """

    def get_lr(iteration: int) -> float:
        """
        map iteration to learning rate
        :param iteration:
        :return:
        """
        if iteration == 0:
            return 1e-2
        return 1e-3

    # this will download the required model for sub-word tokenization
    bpemb_de = BPEmb(lang="de", vs=25000, dim=300)

    # contruct the vocabulary
    itos = dict(enumerate(bpemb_de.words + ['xxpad']))
    voc = Vocab(itos)

    # encode all tokens as IDs
    training_data_lm = pd.DataFrame.from_dict(dict(text=texts, label=[0 for _ in texts]))

    training_data_lm['text'] = training_data_lm['text'].apply(
        lambda x: bpemb_de.encode_ids_with_bos_eos(clean(x, stp_lang='german')))

    val_data = pd.DataFrame.from_dict(dict(text=X_test, label=Y_test))
    val_data['text'] = val_data['text'].apply(lambda x: bpemb_de.encode_ids_with_bos_eos(clean(x, stp_lang='german')))

    # setup language model data
    data_lm = TextLMDataBunch.from_ids(path=path,
                                       vocab=voc,
                                       train_ids=training_data_lm['text'],
                                       valid_ids=val_data['text'])

    # awd_lstm_lm_config = dict(emb_sz=400, n_hid=1152, n_layers=3, pad_token=1, qrnn=False, bidir=False, output_p=0.1,
    #                          hidden_p=0.15, input_p=0.25, embed_p=0.02, weight_p=0.2, tie_weights=True, out_bias=True)
    config = awd_lstm_lm_config.copy()
    config['n_hid'] = 1150

    # setup learner, load the model beforehand
    learner = language_model_learner(data_lm, AWD_LSTM, pretrained=False, config=config, drop_mult=0.5)
    learner.load(path_to_encoder)

    iterations = []
    paths = []
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []

    print("evaluating original model")

    path = f"{encoder_name}_{0}"
    paths.append(path)
    iterations.append(0)
    train_loss, train_accuracy = learner.validate(dl=learner.data.train_dl)
    train_loss = float(train_loss)
    train_accuracy = float(train_accuracy.detach().item())

    valid_loss, valid_accuracy = learner.validate(dl=learner.data.valid_dl)
    valid_loss = float(valid_loss)
    valid_accuracy = float(valid_accuracy.detach().item())
    training_losses.append(train_loss)
    validation_losses.append(valid_loss)
    training_accuracies.append(train_accuracy)
    validation_accuracies.append(valid_accuracy)

    learner.save_encoder(path)

    if lm_iterations > 0:
        print("starting very first training cycle with partially frozen model")
        learner.fit_one_cycle(1, get_lr(iteration=iterations[0]))
        path = f"{encoder_name}_{1}"
        paths.append(path)
        iterations.append(1)
        train_loss, train_accuracy = learner.validate(dl=learner.data.train_dl)
        train_loss = float(train_loss)
        train_accuracy = float(train_accuracy.detach().item())

        valid_loss, valid_accuracy = learner.validate(dl=learner.data.valid_dl)
        valid_loss = float(valid_loss)
        valid_accuracy = float(valid_accuracy.detach().item())
        training_losses.append(train_loss)
        validation_losses.append(valid_loss)
        training_accuracies.append(train_accuracy)
        validation_accuracies.append(valid_accuracy)

        learner.save_encoder(path)
        print("unfreezing")
        learner.unfreeze()
        learner.save_encoder(path)

    for iteration in range(lm_iterations - 1):
        print(f"lm training cycle {iteration}")
        learner.fit_one_cycle(1, get_lr(iteration))
        path = f"{encoder_name}_{iteration + 2}"
        paths.append(path)
        iterations.append(iteration + 2)
        train_loss, train_accuracy = learner.validate(dl=learner.data.train_dl)
        train_loss = float(train_loss)
        train_accuracy = float(train_accuracy.detach().item())

        valid_loss, valid_accuracy = learner.validate(dl=learner.data.valid_dl)
        valid_loss = float(valid_loss)
        valid_accuracy = float(valid_accuracy.detach().item())
        training_losses.append(train_loss)
        validation_losses.append(valid_loss)
        training_accuracies.append(train_accuracy)
        validation_accuracies.append(valid_accuracy)
        learner.save_encoder(path)
    return TrainedLMConfig(model_paths=paths,
                           vocab=data_lm.train_ds.vocab,
                           lm_iterations=iterations,
                           training_accuracies=training_accuracies,
                           validation_accuracies=validation_accuracies,
                           training_losses=training_losses,
                           validation_losses=validation_losses)


def continue_training_ulmfit_language_models(texts: Iterable[str],
                                             X_test: Iterable[str],
                                             Y_test: Iterable[str],
                                             path_encoder_to_load: str,
                                             start_iterations: int,
                                             path: str = "/tmp",
                                             lm_iterations: int = 2,
                                             encoder_name: str = "encoder") -> TrainedLMConfig:
    """
    continue fine-tuning pretrained model
    :param texts:
    :param X_test:
    :param Y_test:
    :param path:
    :param lm_iterations:
    :param encoder_to_load:
    :param encoder_name:
    :return:
    """

    def get_lr(iteration: int) -> float:
        """
        map iteration to learning rate
        :param iteration:
        :return:
        """
        if iteration == 0:
            return 1e-2
        return 1e-3

    training_data_lm = pd.DataFrame.from_dict(dict(text=texts, label=[0 for _ in texts]))
    val_data = pd.DataFrame.from_dict(dict(text=X_test, label=Y_test))
    data_lm = TextLMDataBunch.from_df(path=path,
                                      train_df=training_data_lm, valid_df=val_data, text_cols=["text"])
    learner = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
    iterations = []
    paths = []
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []

    print("loading fine-tuned model")
    learner.load_encoder(path_encoder_to_load)

    print("unfreezing")

    for iteration in range(start_iterations, start_iterations + lm_iterations):
        print(f"lm training cycle {iteration}")
        learner.fit_one_cycle(1, get_lr(iteration))
        path = f"{encoder_name}_{iteration + 1}"
        paths.append(path)
        iterations.append(iteration + 1)
        train_loss, train_accuracy = learner.validate(dl=learner.data.train_dl)
        train_loss = float(train_loss)
        train_accuracy = float(train_accuracy.detach().item())

        valid_loss, valid_accuracy = learner.validate(dl=learner.data.valid_dl)
        valid_loss = float(valid_loss)
        valid_accuracy = float(valid_accuracy.detach().item())
        training_losses.append(train_loss)
        validation_losses.append(valid_loss)
        training_accuracies.append(train_accuracy)
        validation_accuracies.append(valid_accuracy)
        learner.save_encoder(path)
    return TrainedLMConfig(model_paths=paths,
                           vocab=data_lm.train_ds.vocab,
                           lm_iterations=iterations,
                           training_accuracies=training_accuracies,
                           validation_accuracies=validation_accuracies,
                           training_losses=training_losses,
                           validation_losses=validation_losses)


def train_ulmfit_clfs_english(path: str,
                              X_train: Iterable[str],
                              Y_train: Iterable[str],
                              X_test: Iterable[str],
                              Y_test: Iterable[str],
                              vocab: Vocab,
                              path_encoder: str,
                              label_array: np.ndarray,
                              clf_iterations: int) -> TrainedClfConfig:
    training_data_clf = pd.DataFrame.from_dict(dict(text=X_train, label=Y_train))
    val_data = pd.DataFrame.from_dict(dict(text=X_test, label=Y_test))

    data_clas = TextClasDataBunch.from_df(path=path,
                                          train_df=training_data_clf,
                                          valid_df=val_data,
                                          text_cols=["text"],
                                          label_cols=["label"],
                                          vocab=vocab,
                                          classes=label_array
                                          )

    learner = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
    print(f"reusing encoder {path_encoder} for text classifier")
    learner.load_encoder(path_encoder)

    paths = []
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []

    # first iteration, only train last layer

    if clf_iterations > 0:
        learner.freeze_to(-2)
    learner.fit_one_cycle(1, 1e-2)
    path = os.path.join("models", f"ulmfit_model_{0}.pth")
    paths.append(path)

    train_loss, train_accuracy = learner.validate(dl=learner.data.train_dl)
    train_loss = float(train_loss)
    train_accuracy = float(train_accuracy.detach().item())

    valid_loss, valid_accuracy = learner.validate(dl=learner.data.valid_dl)
    valid_loss = float(valid_loss)
    valid_accuracy = float(valid_accuracy.detach().item())
    training_losses.append(train_loss)
    validation_losses.append(valid_loss)
    training_accuracies.append(train_accuracy)
    validation_accuracies.append(valid_accuracy)

    learner.export(path)

    # second iteration, unfreeze and train somewhat faster
    learner.unfreeze()
    if clf_iterations > 1:
        print(f"training clf {1}")
        learner.fit_one_cycle(1, slice(5e-3 / 2., 5e-3))
        path = os.path.join("models", f"ulmfit_model_{1}.pth")
        paths.append(path)

        train_loss, train_accuracy = learner.validate(dl=learner.data.train_dl)
        train_loss = float(train_loss)
        train_accuracy = float(train_accuracy.detach().item())

        valid_loss, valid_accuracy = learner.validate(dl=learner.data.valid_dl)
        valid_loss = float(valid_loss)
        valid_accuracy = float(valid_accuracy.detach().item())
        training_losses.append(train_loss)
        validation_losses.append(valid_loss)
        training_accuracies.append(train_accuracy)
        validation_accuracies.append(valid_accuracy)

        learner.export(path)

    # remaining iterations, train slower
    for i in range(2, clf_iterations):
        print(f"training clf {i}")
        learner.fit_one_cycle(1, slice(2e-3 / 100, 2e-3))
        path = os.path.join("models", f"ulmfit_model_{i}.pth")
        paths.append(path)

        train_loss, train_accuracy = learner.validate(dl=learner.data.train_dl)
        train_loss = float(train_loss)
        train_accuracy = float(train_accuracy.detach().item())

        valid_loss, valid_accuracy = learner.validate(dl=learner.data.valid_dl)
        valid_loss = float(valid_loss)
        valid_accuracy = float(valid_accuracy.detach().item())
        training_losses.append(train_loss)
        validation_losses.append(valid_loss)
        training_accuracies.append(train_accuracy)
        validation_accuracies.append(valid_accuracy)

        learner.export(path)
    return TrainedClfConfig(model_paths=paths,
                            clf_iterations=list(range(clf_iterations)),
                            training_losses=training_losses,
                            validation_losses=validation_losses,
                            training_accuracies=training_accuracies,
                            validation_accuracies=validation_accuracies)


def train_ulmfit_clfs_german(path: str,
                             X_train: Iterable[str],
                             Y_train: Iterable[str],
                             X_test: Iterable[str],
                             Y_test: Iterable[str],
                             vocab: Vocab,
                             path_encoder: str,
                             label_array: np.ndarray,
                             clf_iterations: int,
                             label_mapper: LabelMapper) -> TrainedClfConfig:
    # this will download the required model for sub-word tokenization
    bpemb_de = BPEmb(lang="de", vs=25000, dim=300)

    # contruct the vocabulary
    itos = dict(enumerate(bpemb_de.words + [transform.PAD]))
    voc = Vocab(itos)

    training_data = pd.DataFrame.from_dict(dict(text=X_train, label=Y_train))
    validation_data = pd.DataFrame.from_dict(dict(text=X_test, label=Y_test))

    training_data['text'] = training_data['text'].apply(
        lambda x: bpemb_de.encode_ids_with_bos_eos(clean(x, stp_lang='german')))
    validation_data['text'] = validation_data['text'].apply(
        lambda x: bpemb_de.encode_ids_with_bos_eos(clean(x, stp_lang='german')))

    training_data['label'] = training_data['label'].apply(lambda x: label_mapper.actuallabel2index[x])
    validation_data['label'] = validation_data['label'].apply(lambda x: label_mapper.actuallabel2index[x])

    data_clas = TextClasDataBunch.from_ids(path=path,
                                           vocab=voc,
                                           pad_idx=25000,
                                           train_ids=training_data['text'],
                                           valid_ids=validation_data['text'],
                                           train_lbls=training_data["label"],
                                           valid_lbls=validation_data["label"],
                                           classes=label_mapper.map_to_indices(label_array)
                                           )
    config = awd_lstm_clas_config.copy()
    config['n_hid'] = 1150
    learner = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5, config=config)
    print(f"reusing encoder {path_encoder} for text classifier")
    learner.load_encoder(path_encoder)

    paths = []
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []

    # first iteration, only train last layer

    if clf_iterations > 0:
        learner.freeze_to(-2)
    learner.fit_one_cycle(1, 1e-2)
    path = os.path.join("models", f"ulmfit_model_{0}.pth")
    paths.append(path)

    train_loss, train_accuracy = learner.validate(dl=learner.data.train_dl)
    train_loss = float(train_loss)
    train_accuracy = float(train_accuracy.detach().item())

    valid_loss, valid_accuracy = learner.validate(dl=learner.data.valid_dl)
    valid_loss = float(valid_loss)
    valid_accuracy = float(valid_accuracy.detach().item())
    training_losses.append(train_loss)
    validation_losses.append(valid_loss)
    training_accuracies.append(train_accuracy)
    validation_accuracies.append(valid_accuracy)

    learner.export(path)

    # second iteration, unfreeze and train somewhat faster
    learner.unfreeze()
    if clf_iterations > 1:
        print(f"training clf {1}")
        learner.fit_one_cycle(1, slice(5e-3 / 2., 5e-3))
        path = os.path.join("models", f"ulmfit_model_{1}.pth")
        paths.append(path)

        train_loss, train_accuracy = learner.validate(dl=learner.data.train_dl)
        train_loss = float(train_loss)
        train_accuracy = float(train_accuracy.detach().item())

        valid_loss, valid_accuracy = learner.validate(dl=learner.data.valid_dl)
        valid_loss = float(valid_loss)
        valid_accuracy = float(valid_accuracy.detach().item())
        training_losses.append(train_loss)
        validation_losses.append(valid_loss)
        training_accuracies.append(train_accuracy)
        validation_accuracies.append(valid_accuracy)

        learner.export(path)

    # remaining iterations, train slower
    for i in range(2, clf_iterations):
        print(f"training clf {i}")
        learner.fit_one_cycle(1, slice(2e-3 / 100, 2e-3))
        path = os.path.join("models", f"ulmfit_model_{i}.pth")
        paths.append(path)

        train_loss, train_accuracy = learner.validate(dl=learner.data.train_dl)
        train_loss = float(train_loss)
        train_accuracy = float(train_accuracy.detach().item())

        valid_loss, valid_accuracy = learner.validate(dl=learner.data.valid_dl)
        valid_loss = float(valid_loss)
        valid_accuracy = float(valid_accuracy.detach().item())
        training_losses.append(train_loss)
        validation_losses.append(valid_loss)
        training_accuracies.append(train_accuracy)
        validation_accuracies.append(valid_accuracy)

        learner.export(path)
    return TrainedClfConfig(model_paths=paths,
                            clf_iterations=list(range(clf_iterations)),
                            training_losses=training_losses,
                            validation_losses=validation_losses,
                            training_accuracies=training_accuracies,
                            validation_accuracies=validation_accuracies)


def evaluate_ulmfit_english(corpus: Corpus,
                            path_isolated_corpus: str,
                            path_additional_lm_data: str,
                            model_path: str = "/tmp",
                            results_path: str = "/tmp/results.json",
                            encoder_name: str = "encoder",
                            path_encoder_to_load: str = None,
                            start_iterations: int = None,
                            test_size: float = 0.1,
                            lm_iterations: int = 2,
                            clf_iterations: int = 1,
                            n_runs: int = 1, ):
    X_train, X_test, Y_train, Y_test = train_test_split(corpus.X,
                                                        corpus.Y,
                                                        test_size=test_size,
                                                        random_state=42,
                                                        stratify=corpus.Y)

    texts = [text for source in (X_train, yield_normalized_batched_texts(path_additional_lm_data)) for text in source]

    evaluator = IsolatedEvaluation(
        data_loder=KeyedBatchLoader(**load_messages(path=path_isolated_corpus)),
        labels_to_evaluate=corpus.label_mapper.labels,
        junk_label=-1,
        junk_threshold=0.85,
        minibatch_size=128)

    all_results = []
    for i_run in range(n_runs):
        print(f"starting run {i_run}")
        print("training language models ...")
        if path_encoder_to_load is None and start_iterations is None:
            lm_config = train_ulmfit_language_models_english(texts=texts,
                                                             X_test=X_test,
                                                             Y_test=Y_test,
                                                             path=model_path,
                                                             lm_iterations=lm_iterations,
                                                             encoder_name=f"{encoder_name}_run{i_run}")
        elif path_encoder_to_load is not None and start_iterations is not None:
            lm_config = continue_training_ulmfit_language_models(texts=texts,
                                                                 X_test=X_test,
                                                                 Y_test=Y_test,
                                                                 path=model_path,
                                                                 lm_iterations=lm_iterations,
                                                                 encoder_name=f"{encoder_name}_run{i_run}",
                                                                 path_encoder_to_load=path_encoder_to_load,
                                                                 start_iterations=start_iterations)
        else:
            raise Exception("path_encoder_to_load and start_iterations are either both None or not None")
        print(lm_config)
        for i_lm_config, (
                lm_path, lm_iterations, lm_train_loss, lm_valid_loss, lm_train_acc, lm_valid_acc) in enumerate(zip(
            lm_config.model_paths,
            lm_config.lm_iterations,
            lm_config.training_losses,
            lm_config.validation_losses,
            lm_config.training_accuracies,
            lm_config.validation_accuracies)):
            print(f"training classifiers for lm {i_lm_config} / {len(lm_config.model_paths)}")
            clf_config = train_ulmfit_clfs_english(path=model_path,
                                                   X_train=X_train,
                                                   X_test=X_test,
                                                   Y_train=Y_train,
                                                   Y_test=Y_test,
                                                   vocab=lm_config.vocab,
                                                   path_encoder=lm_path,
                                                   label_array=corpus.label_mapper.label_array,
                                                   clf_iterations=clf_iterations)

            for i_clf, (
                    clf_path, clf_iteration, clf_train_loss, clf_valid_loss, clf_train_acc, clf_valid_acc) in enumerate(
                zip(
                    clf_config.model_paths,
                    clf_config.clf_iterations,
                    clf_config.training_losses,
                    clf_config.validation_losses,
                    clf_config.training_accuracies,
                    clf_config.validation_accuracies
                )):
                print(f"evaluating classifier {i_clf} / {len(clf_config.model_paths)}")

                learner = load_learner(model_path, clf_path)

                model = ULMFiTModel(label_mapper=corpus.label_mapper,
                                    rnn_learner=learner,
                                    lang="english")

                results = evaluator(model=model,
                                    n_lm=lm_iterations,
                                    n_clf=clf_iteration,
                                    add_loss_acc=False)
                results["lm_training_loss"] = lm_train_loss
                results["lm_validation_loss"] = lm_valid_loss
                results["lm_training_accuracy"] = lm_train_acc
                results["lm_validation_accuracy"] = lm_valid_acc

                results["clf_training_loss"] = clf_train_loss
                results["clf_validation_loss"] = clf_valid_loss
                results["clf_training_accuracy"] = clf_train_acc
                results["clf_validation_accuracy"] = clf_valid_acc
                results["run"] = i_run

                all_results.append(results)
                if not os.path.exists(os.path.join(*os.path.split(results_path[:-1]))):
                    os.makedirs(os.path.join(*os.path.split(results_path[:-1])))
                with open(results_path, "w") as f:
                    json.dump(all_results, f, indent=2)


def evaluate_ulmfit_german(corpus: Corpus,
                           path_isolated_corpus: str,
                           path_additional_lm_data: str,
                           path_to_german_model: str,
                           model_path: str = "/tmp",
                           results_path: str = "/tmp/results.json",
                           encoder_name: str = "encoder",
                           path_encoder_to_reuse: str = None,
                           start_iterations: int = None,
                           test_size: float = 0.1,
                           lm_iterations: int = 2,
                           clf_iterations: int = 1,
                           n_runs: int = 1, ):
    X_train, X_test, Y_train, Y_test = train_test_split(corpus.X,
                                                        corpus.Y,
                                                        test_size=test_size,
                                                        random_state=42,
                                                        stratify=corpus.Y)

    texts = [text for source in (X_train, yield_normalized_batched_texts(path_additional_lm_data)) for text in source]

    evaluator = IsolatedEvaluation(
        data_loder=KeyedBatchLoader(**load_messages(path=path_isolated_corpus)),
        labels_to_evaluate=corpus.label_mapper.labels,
        junk_label=-1,
        junk_threshold=0.85)

    all_results = []
    for i_run in range(n_runs):
        print(f"starting run {i_run}")
        print("training language models ...")
        if path_encoder_to_reuse is None and start_iterations is None:
            lm_config = train_ulmfit_language_models_german(texts=texts,
                                                            X_test=X_test,
                                                            Y_test=Y_test,
                                                            path=model_path,
                                                            lm_iterations=lm_iterations,
                                                            encoder_name=f"{encoder_name}_run{i_run}",
                                                            path_to_encoder=path_to_german_model)
        elif path_encoder_to_reuse is not None and start_iterations is not None:
            raise NotImplementedError()

            # lm_config = continue_training_ulmfit_language_models(texts=texts,
            #                                                      X_test=X_test,
            #                                                      Y_test=Y_test,
            #                                                      path=model_path,
            #                                                      lm_iterations=lm_iterations,
            #                                                      encoder_name=f"{encoder_name}_run{i_run}",
            #                                                      path_encoder_to_load=path_encoder_to_reuse,
            #                                                      start_iterations=start_iterations)
        else:
            raise Exception("path_encoder_to_load and start_iterations are either both None or not None")
        print(lm_config)
        for i_lm_config, (
                lm_path, lm_iterations, lm_train_loss, lm_valid_loss, lm_train_acc, lm_valid_acc) in enumerate(zip(
            lm_config.model_paths,
            lm_config.lm_iterations,
            lm_config.training_losses,
            lm_config.validation_losses,
            lm_config.training_accuracies,
            lm_config.validation_accuracies)):
            print(f"training classifiers for lm {i_lm_config} / {len(lm_config.model_paths)}")
            clf_config = train_ulmfit_clfs_german(path=model_path,
                                                  X_train=X_train,
                                                  X_test=X_test,
                                                  Y_train=Y_train,
                                                  Y_test=Y_test,
                                                  vocab=lm_config.vocab,
                                                  path_encoder=lm_path,
                                                  label_array=corpus.label_mapper.label_array,
                                                  clf_iterations=clf_iterations,
                                                  label_mapper=corpus.label_mapper)

            for i_clf, (
                    clf_path, clf_iteration, clf_train_loss, clf_valid_loss, clf_train_acc, clf_valid_acc) in enumerate(
                zip(
                    clf_config.model_paths,
                    clf_config.clf_iterations,
                    clf_config.training_losses,
                    clf_config.validation_losses,
                    clf_config.training_accuracies,
                    clf_config.validation_accuracies
                )):
                print(f"evaluating classifier {i_clf} / {len(clf_config.model_paths)}")

                learner = load_learner(model_path, clf_path)

                model = ULMFiTModel(label_mapper=corpus.label_mapper,
                                    rnn_learner=learner,
                                    lang="german")

                results = evaluator(model=model,
                                    n_lm=lm_iterations,
                                    n_clf=clf_iteration,
                                    add_loss_acc=False)
                results["lm_training_loss"] = lm_train_loss
                results["lm_validation_loss"] = lm_valid_loss
                results["lm_training_accuracy"] = lm_train_acc
                results["lm_validation_accuracy"] = lm_valid_acc

                results["clf_training_loss"] = clf_train_loss
                results["clf_validation_loss"] = clf_valid_loss
                results["clf_training_accuracy"] = clf_train_acc
                results["clf_validation_accuracy"] = clf_valid_acc
                results["run"] = i_run

                all_results.append(results)
                if not os.path.exists(os.path.split(results_path[:-1])):
                    os.makedirs(os.path.join(*os.path.split(results_path[:-1])))
                with open(results_path, "w") as f:
                    json.dump(all_results, f, indent=2)


def train_ulmfit(corpus: Corpus,
                 path: str = "/tmp",
                 test_size: float = 0.1,
                 lm_iterations: int = 4,
                 clf_iterations: int = 4,
                 encoder_to_load: str = "encoder",
                 encoder_to_save: str = "encoder",
                 additional_lm_data: Iterable[str] = ()) -> ULMFiTModel:
    """
    trains ULMFiT via fast.ai implementation
    :param additional_lm_data:  iterable over additional data for language model to train on
    :param encoder_to_save: encoder model name to save after training language model
    :param encoder_to_load: encoder model to load before training text classifier
    :param path: path where fastai model is stored
    :param corpus: corpus to train on
    :param test_size: test size ratio
    :param lm_iterations: lm is trained at least once plus this
    :param clf_iterations: clf is trained at least twice plus this
    :return:
    """
    X_train, X_test, Y_train, Y_test = train_test_split(corpus.X,
                                                        corpus.Y,
                                                        test_size=test_size,
                                                        random_state=42,
                                                        stratify=corpus.Y)

    texts = [text for source in (X_train, additional_lm_data) for text in source]

    training_data_lm = pd.DataFrame.from_dict(dict(text=texts, label=[0 for _ in texts]))
    training_data_clf = pd.DataFrame.from_dict(dict(text=X_train, label=Y_train))
    val_data = pd.DataFrame.from_dict(dict(text=X_test, label=Y_test))
    data_lm = TextLMDataBunch.from_df(path=path,
                                      train_df=training_data_lm, valid_df=val_data, text_cols=["text"])
    del texts
    data_clas = TextClasDataBunch.from_df(path=path,
                                          train_df=training_data_clf, valid_df=val_data, text_cols=["text"],
                                          label_cols=["label"],
                                          vocab=data_lm.train_ds.vocab,
                                          classes=corpus.label_mapper.label_array
                                          )

    learner = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
    if encoder_to_load is not None:
        print(f"loading encoder {encoder_to_load}")
        learner.load_encoder(encoder_to_load)
    print("starting initial cycle")
    if lm_iterations > 0:
        learner.fit_one_cycle(1, 1e-2)
    print("unfreezing")
    learner.unfreeze()
    for _ in range(lm_iterations - 1):
        learner.fit_one_cycle(1, 1e-3)
    print(f"saving encoder as {encoder_to_save}")
    learner.save_encoder(encoder_to_save)
    learner = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
    print(f"reusing encoder {encoder_to_save} for text classifier")
    learner.load_encoder(encoder_to_save)
    if clf_iterations > 0:
        learner.freeze_to(-2)
    learner.fit_one_cycle(1, 1e-2)
    learner.unfreeze()
    if clf_iterations > 1:
        learner.fit_one_cycle(1, slice(5e-3 / 2., 5e-3))
    for i in range(clf_iterations - 2):
        learner.fit_one_cycle(1, slice(2e-3 / 100, 2e-3))
    learner.export(os.path.join("models", "ulmfit_model.pth"))

    model = ULMFiTModel(label_mapper=corpus.label_mapper,
                        rnn_learner=learner)
    return model

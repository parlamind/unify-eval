from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import torch as t
from keras import Sequential, utils, regularizers
from keras.layers import Embedding, GlobalAveragePooling1D, Dense
from keras_preprocessing import sequence
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import log_loss
from torch.nn import CrossEntropyLoss

from unify_eval.model.mixins.classification import Classifier, MessageLevelClassifier
from unify_eval.model.types import Tensor, Label
from unify_eval.utils.label_mapper import LabelMapper


class PytorchFastTextModule(t.nn.Module):
    """
    Pytorch implementation of FastText
    """

    def __init__(self, n_features: int, n_dims: int, n_classes: int):
        super().__init__()
        self.embedding_bag = t.nn.EmbeddingBag(num_embeddings=n_features, embedding_dim=n_dims)
        self.linear = t.nn.Linear(in_features=n_dims, out_features=n_classes)

    def sample_logits(self, n: int):
        """
        sample random logits,
        i.e. sample input following a standard normal distribution and push it through a linear transformation
        """
        z = t.randn(size=(n, self.linear.in_features))
        return self.linear(z)

    def embed(self, indices: t.Tensor) -> t.Tensor:
        """
        map padded token indices to averaged embeddings
        """
        return self.embedding_bag(indices)

    def __call__(self, indices: t.Tensor):
        averaged_embeddings = self.embed(indices)
        logits = self.linear(averaged_embeddings)
        return logits


class ShannonEntropy(t.nn.Module):
    """
    calculates shannon entropy of some categorical probability vector
    """

    def __call__(self, probs: t.Tensor) -> t.Tensor:
        return (-(probs * t.log(probs + 1e-8)).sum(dim=-1)).mean()


class PytorchFastText(Classifier):
    """
    Model wrapping around a pytorch implementation of FastText
    """

    def __init__(self,
                 label_mapper: LabelMapper,
                 hashing_vectorizer: HashingVectorizer,
                 pytorch_model: PytorchFastTextModule,
                 n_features_to_use: int,
                 l2_weight: float = 0.0,
                 junk_noise_entropy_weight: float = 0.0,
                 text_kw: str = "texts",
                 label_kw: str = "labels"):
        """
        :param label_mapper: LabelMapper instance mapping label name to index and vice versa
        :param hashing_vectorizer: HashingVectorizer instance that generates ngram profiles
        :param pytorch_model: actual pytorch module
        :param n_features_to_use: number of ngram features to actually use
        :param l2_weight: weight of l2 loss
        :param junk_noise_entropy_weight: weight of junk noise when added to loss
        :param text_kw: name of text data
        :param label_kw: name of label data
        """
        super().__init__(label_mapper)
        self.hashing_vectorizer = hashing_vectorizer
        self.pytorch_model = pytorch_model
        self.n_features_to_use = n_features_to_use
        self.junk_noise_entropy_weight = junk_noise_entropy_weight
        self.l2_weight = l2_weight
        self.text_kw = text_kw
        self.label_kw = label_kw

        self.xent = CrossEntropyLoss()
        self.shannon_entropy = ShannonEntropy()
        print(f"l2_weight {l2_weight}")
        self.optimizer = t.optim.Adam(params=list(self.pytorch_model.parameters()), weight_decay=l2_weight)
        self.optimizer.zero_grad()

    def predict_label_probabilities(self, **kwargs) -> Tensor:
        return self.get_logits(**kwargs).detach().numpy()

    def get_logits(self, **kwargs) -> Tensor:
        return self.pytorch_model(self.preprocess_texts(kwargs[self.text_kw]))

    def train(self, maximize_junk_entropy: bool = False,
              **kwargs) -> "PytorchFastText":
        loss_dict = self.get_loss(as_tensor=True, **kwargs)
        loss = loss_dict["cross_entropy"]
        if "junk_shannon_entropy" in loss_dict:
            loss -= loss_dict["junk_shannon_entropy"]
        if self.junk_noise_entropy_weight > 0.0 and "shannon_entropy" in loss_dict:
            loss -= self.junk_noise_entropy_weight * loss_dict["shannon_entropy"]
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return self

    def get_loss(self, as_tensor: bool = False,
                 maximize_junk_entropy: bool = False,
                 junk_label: Label = -1,
                 **kwargs) -> Dict[str, Tensor]:

        def get_true_data(texts: List[str], targets: List[Label]) -> Tuple[List[str], List[Label]]:
            true_indices = [i for i, label in enumerate(targets) if label != junk_label]
            return np.array(texts)[true_indices], np.array(targets)[true_indices]

        def get_junk_data(texts: List[str], targets: List[Label]) -> List[str]:
            true_indices = [i for i, label in enumerate(targets) if label == junk_label]
            return np.array(texts)[true_indices]

        texts = kwargs.pop(self.text_kw)
        labels = kwargs.pop(self.label_kw)

        if maximize_junk_entropy:
            clauses_true, labels_true = get_true_data(texts=texts, targets=labels)
            logits = self.get_logits(clauses=clauses_true, **kwargs)
            loss_dict = {"cross_entropy": self.xent(
                input=logits,
                target=t.from_numpy(self.label_mapper.map_to_indices(labels_true)).long())}

            texts_junk = get_junk_data(texts=texts, targets=labels)
            print(f"{len(texts_junk)} junk texts found")

            if len(texts_junk) > 0:
                loss_dict["junk_shannon_entropy"] = self.shannon_entropy(
                    t.softmax(self.pytorch_model(self.preprocess_texts(texts_junk)), dim=-1))
        else:

            logits = self.get_logits(clauses=texts, **kwargs)
            loss_dict = {"cross_entropy": self.xent(
                input=logits,
                target=t.from_numpy(self.label_mapper.map_to_indices(labels)).long())}

        if self.junk_noise_entropy_weight > 0.0:
            loss_dict["shannon_entropy"] = self.shannon_entropy(
                t.softmax(self.pytorch_model.sample_logits(n=logits.shape[0]), dim=-1))
        if not as_tensor:
            loss_dict = dict((k, v.detach().item()) for k, v in loss_dict.items())
        return loss_dict

    @staticmethod
    def from_components(**kwargs) -> "PytorchFastText":
        return PytorchFastText(**kwargs)

    def get_components(self) -> dict:
        return {
            "hashing_vectorizer": self.hashing_vectorizer,
            "label_mapper": self.label_mapper,
            "pytorch_model": self.pytorch_model,
            "l2_weight": self.l2_weight,
            "junk_noise_entropy_weight": self.junk_noise_entropy_weight,
            "n_features_to_use": self.n_features_to_use,
            "text_kw": self.text_kw,
            "label_kw": self.label_kw
        }

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        return dict((name, p.detach().numpy()) for name, p in self.pytorch_model.named_parameters())

    def preprocess_texts(self, texts: List[str]) -> t.Tensor:
        """
        maps list of texts to ngram profiles
        """
        onehots = self.hashing_vectorizer.transform(texts).toarray()
        # extract all the non-zero onehot indices
        sequences = np.array([np.arange(d.shape[-1])[d > 0.1] for d in onehots])
        return t.from_numpy(sequence.pad_sequences(sequences=sequences, maxlen=self.n_features_to_use)).long()

    @staticmethod
    def make(label_mapper: LabelMapper,
             hashing_vectorizer: HashingVectorizer,
             n_dims: int,
             n_classes: int,
             n_features_to_use: int,
             l2_weight: float = 0.0,
             junk_noise_entropy_weight: float = 0.0,
             text_kw: str = "texts",
             label_kw: str = "labels") -> "PytorchFastText":
        """
        Factory that generates new PytorchFastText instance
        :param label_mapper: LabelMapper instance mapping label name to index and vice versa
        :param hashing_vectorizer: HashingVectorizer instance that generates ngram profiles
        :param n_dims: dimensionality of embedding space
        :param n_classes: number of labels to classify
        :param n_features_to_use: number of ngram features to actually use
        :param l2_weight: weight of l2 loss
        :param junk_noise_entropy_weight: weight of junk noise when added to loss
        :param text_kw: name of text data
        :param label_kw: name of label data
        """
        model = PytorchFastTextModule(n_features=hashing_vectorizer.n_features,
                                      n_dims=n_dims,
                                      n_classes=n_classes)
        return PytorchFastText(label_mapper=label_mapper,
                               hashing_vectorizer=hashing_vectorizer,
                               pytorch_model=model,
                               l2_weight=l2_weight,
                               junk_noise_entropy_weight=junk_noise_entropy_weight,
                               n_features_to_use=n_features_to_use,
                               text_kw=text_kw,
                               label_kw=label_kw)


class MessageLevelFastTextModule(t.nn.Module):
    """
    Simple fast text model that runs on single clauses independently, without any interaction
    """

    def __init__(self, n_features: int, n_dims: int, n_classes: int):
        super().__init__()
        self.embedding = t.nn.Embedding(num_embeddings=n_features, embedding_dim=n_dims)
        self.linear = t.nn.Linear(in_features=n_dims, out_features=n_classes)

    def sample_logits(self, n: int, n_clauses: int):
        """
        sample random logits,
        i.e. sample input following a standard normal distribution and push it through a linear transformation
        """
        z = t.randn(size=(n, n_clauses, self.linear.in_features))
        return self.linear(z)

    def encode(self, indices_per_message: t.Tensor) -> t.Tensor:
        """
        map padded token indices to averaged embeddings
        """
        return self.embedding(indices_per_message).mean(dim=-2)

    def __call__(self, indices_per_message: t.Tensor):
        averaged_embeddings = self.encode(indices_per_message=indices_per_message)
        logits = self.linear(averaged_embeddings)
        return logits


class MessageLevelFastText(MessageLevelClassifier):
    """
    wrapper around MessageLevelFastTextModule, i.e. fasttext model that runs on single clauses independently
    """

    def __init__(self,
                 label_mapper: LabelMapper,
                 hashing_vectorizer: HashingVectorizer,
                 pytorch_model: MessageLevelFastTextModule,
                 n_features_to_use: int,
                 l2_weight: float = 0.0,
                 junk_noise_entropy_weight: float = 0.0,
                 text_kw: str = "texts",
                 label_kw: str = "labels"):
        """
        :param label_mapper: LabelMapper instance mapping label name to index and vice versa
        :param hashing_vectorizer: HashingVectorizer instance that generates ngram profiles
        :param pytorch_model: actual pytorch module
        :param n_features_to_use: number of ngram features to actually use
        :param l2_weight: weight of l2 loss
        :param junk_noise_entropy_weight: weight of junk noise when added to loss
        :param text_kw: name of text data
        :param label_kw: name of label data
        """
        super().__init__(label_mapper)
        self.hashing_vectorizer = hashing_vectorizer
        self.pytorch_model = pytorch_model
        self.n_features_to_use = n_features_to_use
        self.junk_noise_entropy_weight = junk_noise_entropy_weight
        self.l2_weight = l2_weight
        self.text_kw = text_kw
        self.label_kw = label_kw

        self.xent = CrossEntropyLoss(ignore_index=label_mapper.ignore_index)
        self.shannon_entropy = ShannonEntropy()
        self.optimizer = t.optim.Adam(params=list(self.pytorch_model.parameters()), weight_decay=l2_weight)
        self.optimizer.zero_grad()

    def predict_label_probabilities(self, **kwargs) -> Tensor:
        return np.array([logits.detach().numpy() for logits in self.get_logits(**kwargs)])

    def get_logits(self, clauses: List[List[str]], **kwargs) -> Tensor:
        return self.pytorch_model(indices_per_message=self.preprocess_clauses(clauses))

    def train(self, **kwargs) -> "MessageLevelFastText":
        loss_dict = self.get_loss(as_tensor=True, **kwargs)
        loss = loss_dict["cross_entropy"]
        if "junk_shannon_entropy" in loss_dict:
            loss -= loss_dict["junk_shannon_entropy"]
        if self.junk_noise_entropy_weight > 0.0 and "shannon_entropy" in loss_dict:
            loss -= self.junk_noise_entropy_weight * loss_dict["shannon_entropy"]

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return self

    def get_loss(self,
                 as_tensor: bool = False,
                 junk_label: Label = -1,
                 maximize_junk_entropy: bool = False,
                 **kwargs) -> Dict[str, Tensor]:

        def expand_targets_to_max_text_number(n_texts: int, targets: List[List[Label]]) -> List[List[Label]]:
            """
            add ignore_index to dummy clauses
            :param n_texts: maximum number of texts in a minibatch
            :param targets: original target label indices
            :return: updated list of list of labels, with ignore_index added for every dummy text
            """
            return [list(ys_pred) + [self.xent.ignore_index for _ in range(n_texts - len(ys_pred))] for ys_pred in
                    targets]

        def get_true_data(messages: List[List[str]], targets: List[List[Label]]) -> Tuple[
            List[List[str]], List[List[Label]]]:
            true_messages = []
            true_targets = []
            for message, target in zip(messages, targets):
                true_indices = [i for i, label in enumerate(target) if label != junk_label]
                true_messages.append(np.array(message)[true_indices])
                true_targets.append(np.array(target)[true_indices])
            return true_messages, true_targets

        def get_junk_data(messages: List[List[str]], targets: List[List[Label]]) -> List[List[str]]:
            true_messages = []
            true_targets = []
            for message, target in zip(messages, targets):
                true_indices = [i for i, label in enumerate(target) if label == junk_label]
                true_messages.append(np.array(message)[true_indices])
                true_targets.append(np.array(target)[true_indices])
            return true_messages

        messages = kwargs[self.text_kw]
        targets = kwargs[self.label_kw]
        if maximize_junk_entropy:

            true_messages, true_targets = get_true_data(messages=messages, targets=targets)

            logits_per_message = self.get_logits(clauses=true_messages)

            # highest count of clauses per message in the given minibatch
            max_text_number = logits_per_message.shape[-2]

            # target labels with dummy targets
            expanded_targets = expand_targets_to_max_text_number(n_texts=max_text_number, targets=true_targets)

            # target labels mapped to torch tensor containing label indices
            target_indices_per_message = t.stack(
                [t.from_numpy(self.label_mapper.map_to_indices(labels_per_message)).long()
                 for labels_per_message in expanded_targets], dim=0)

            # actual cross entropy loss
            xent = self.xent(input=logits_per_message.view((-1, self.label_mapper.n_labels)),
                             target=target_indices_per_message.view((-1,)))
            loss_dict = {"cross_entropy": xent}

            junk_messages = get_junk_data(messages=messages, targets=targets)
            if len(junk_messages) > 0:
                junk_shannon_entropy = self.shannon_entropy(t.softmax(
                    self.pytorch_model(
                        indices_per_message=self.preprocess_clauses(messages=junk_messages)),
                    dim=-1))
                loss_dict["junk_shannon_entropy"] = junk_shannon_entropy
        else:

            logits_per_message = self.get_logits(clauses=messages)

            # highest count of texts per message in the given minibatch
            max_text_number = logits_per_message.shape[-2]

            # target labels with dummy targets
            expanded_targets = expand_targets_to_max_text_number(n_texts=max_text_number, targets=targets)

            # target labels mapped to torch tensor containing label indices
            target_indices_per_message = t.stack(
                [t.from_numpy(self.label_mapper.map_to_indices(labels_per_message)).long()
                 for labels_per_message in expanded_targets], dim=0)

            # actual cross entropy loss
            xent = self.xent(input=logits_per_message.view((-1, self.label_mapper.n_labels)),
                             target=target_indices_per_message.view((-1,)))
            loss_dict = {"cross_entropy": xent}

        # add junk noise if desired
        if self.junk_noise_entropy_weight > 0.0:
            shannon_entropy = self.shannon_entropy(
                t.softmax(self.pytorch_model.sample_logits(*logits_per_message.shape[:-1]),
                          dim=-1))
            loss_dict["shannon_entropy"] = shannon_entropy

        if not as_tensor:
            loss_dict = dict((k, v.detach().item()) for k, v in loss_dict.items())
        return loss_dict

    @staticmethod
    def from_components(**kwargs) -> "MessageLevelFastText":
        return MessageLevelFastText(**kwargs)

    def get_components(self) -> dict:
        return {
            "hashing_vectorizer": self.hashing_vectorizer,
            "label_mapper": self.label_mapper,
            "pytorch_model": self.pytorch_model,
            "l2_weight": self.l2_weight,
            "junk_noise_entropy_weight": self.junk_noise_entropy_weight,
            "n_features_to_use": self.n_features_to_use,
            "text_kw": self.text_kw,
            "label_kw": self.label_kw
        }

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        return dict((name, p.detach().numpy()) for name, p in self.pytorch_model.named_parameters())

    def preprocess_clauses(self, messages: List[List[str]]) -> t.Tensor:
        """

        :param messages: list of messages, each containing list of clauses
        :return: long tensor of shape [batch_size, n_clauses,n_features_to_use]
        """

        def get_max_clause_number(messages: List[List[str]]) -> int:
            """
            get highest number of clauses per message
            :param messages: minibatch of messages, each containing a list of clauses
            :return:
            """
            return int(np.array([len(message) for message in messages]).max())

        def add_dummy_clauses(n_clauses: int, message: List[str]) -> List[str]:
            """
            fill shorted messages with dummy clauses
            :param n_clauses: max number of clauses per message in given minibatch
            :param message: list of single clauses
            :return: message with possibly added clauses
            """
            return message + ["#" for _ in range(n_clauses - len(message))]

        # get highest number of clauses per message in the given minibatch
        max_n_clauses = get_max_clause_number(messages=messages)

        # add dummy clauses
        messages = [add_dummy_clauses(n_clauses=max_n_clauses, message=list(message)) for message in messages]

        # transform clauses into onehots
        onehots_per_message = [self.hashing_vectorizer.transform(clause).toarray() for clause in messages]

        # retrieve onehot indices
        sequences_per_message = [np.array([np.arange(d.shape[-1])[d > 0.1] for d in onehots])
                                 for onehots in onehots_per_message]

        # pad index sequences to vectors of length n_features_to_use, and stack them clause-wise
        return t.stack([t.from_numpy(sequence.pad_sequences(sequences=sequences,
                                                            maxlen=self.n_features_to_use)).long()
                        for sequences in sequences_per_message], dim=0)

    @staticmethod
    def make(label_mapper: LabelMapper,
             hashing_vectorizer: HashingVectorizer,
             n_dims: int,
             n_classes: int,
             n_features_to_use: int = 1024,
             l2_weight: float = 0.0,
             junk_noise_entropy_weight: float = 0.0,
             text_kw: str = "text_kw",
             label_kw: str = "label_kw") -> "MessageLevelFastText":
        """
        Factory that generates new PytorchFastText instance
        :param label_mapper: LabelMapper instance mapping label name to index and vice versa
        :param hashing_vectorizer: HashingVectorizer instance that generates ngram profiles
        :param n_dims: dimensionality of embedding space
        :param n_classes: number of labels to classify
        :param n_features_to_use: number of ngram features to actually use
        :param l2_weight: weight of l2 loss
        :param junk_noise_entropy_weight: weight of junk noise when added to loss
        :param text_kw: name of text data
        :param label_kw: name of label data
        """
        model = MessageLevelFastTextModule(n_features=hashing_vectorizer.n_features,
                                           n_dims=n_dims,
                                           n_classes=n_classes)
        return MessageLevelFastText(label_mapper=label_mapper,
                                    hashing_vectorizer=hashing_vectorizer,
                                    pytorch_model=model,
                                    l2_weight=l2_weight,
                                    junk_noise_entropy_weight=junk_noise_entropy_weight,
                                    n_features_to_use=n_features_to_use,
                                    text_kw=text_kw,
                                    label_kw=label_kw)


class FastText(Classifier):
    "Wrapper around keras implementation of FastText"

    def __init__(self,
                 label_mapper: LabelMapper,
                 hashing_vectorizer: HashingVectorizer,
                 keras_model: Sequential,
                 n_features_to_use: int,
                 text_kw: str = "texts",
                 label_kw: str = "labels"):
        """
        :param label_mapper: LabelMapper instance mapping label name to index and vice versa
        :param hashing_vectorizer: HashingVectorizer instance that generates ngram profiles
        :param keras_model: actual keras model
        :param n_features_to_use: number of ngram features to actually use
        :param text_kw: name of text data
        :param label_kw: name of label data
        """
        super().__init__(label_mapper)
        self.hashing_vectorizer = hashing_vectorizer
        self.keras_model = keras_model
        self.n_features_to_use = n_features_to_use
        self.text_kw = text_kw
        self.label_kw = label_kw

    def preprocess_clauses(self, texts: List[str]) -> np.ndarray:
        onehots = self.hashing_vectorizer.transform(texts).toarray()
        sequences = np.array([np.arange(d.shape[-1])[d > 0.1] for d in onehots])
        return sequence.pad_sequences(sequences=sequences, maxlen=self.n_features_to_use)

    def predict_label_probabilities(self, **kwargs) -> Tensor:
        indices = self.preprocess_clauses(texts=kwargs[self.text_kw])
        return self.keras_model.predict_proba(x=indices)

    def _logit(self, x):
        return np.log(x / (np.clip(x, 1e-8, (1.0 - 1e-8)) - x))

    def get_logits(self, **kwargs) -> Tensor:
        indices = self.preprocess_clauses(texts=kwargs[self.text_kw])
        return self._logit(self.keras_model.predict_proba(x=indices))

    def train(self, **kwargs) -> "FastText":
        self.keras_model.train_on_batch(x=self.preprocess_clauses(kwargs[self.text_kw]),
                                        y=utils.to_categorical(
                                            y=self.label_mapper.map_to_indices(kwargs[self.label_kw]),
                                            num_classes=self.label_mapper.n_labels))
        return self

    def get_loss(self, **kwargs) -> Dict[str, Tensor]:
        all_labels = self.label_mapper.all_indices
        y_true = self.label_mapper.map_to_indices(kwargs[self.label_kw])
        y_pred = self.predict_label_probabilities(clauses=kwargs[self.text_kw])
        return {
            "cross_entropy": log_loss(y_true=y_true,
                                      y_pred=y_pred,
                                      labels=all_labels)
        }

    @staticmethod
    def from_components(**kwargs) -> "FastText":
        return FastText(**kwargs)

    def get_components(self) -> dict:
        return {
            "label_mapper": self.label_mapper,
            "hashing_vectorizer": self.hashing_vectorizer,
            "keras_model": self.keras_model,
            "max_len": self.n_features_to_use,
            "text_kw": self.text_kw,
            "label_kw": self.label_kw
        }

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        names = [weight.name for layer in self.keras_model.layers for weight in layer.weights]
        weights = self.keras_model.get_weights()

        return dict((name, weight) for name, weight in zip(names, weights))

    @staticmethod
    def make(label_mapper: LabelMapper,
             hashing_vectorizer: HashingVectorizer,
             n_dims: int,
             n_classes: int,
             n_features_to_use: int,
             label_smoothing: float = 0.0,
             l2_weight: float = 0.0,
             text_kw: str = "texts",
             label_kw: str = "labels") -> "FastText":
        """
        Factory that generates new PytorchFastText instance
        :param label_mapper: LabelMapper instance mapping label name to index and vice versa
        :param hashing_vectorizer: HashingVectorizer instance that generates ngram profiles
        :param n_dims: dimensionality of embedding space
        :param n_classes: number of labels to classify
        :param n_features_to_use: number of ngram features to actually use
        :param label_smoothing: float in [0, 1]. When > 0, label values are smoothed
        :param l2_weight: weight of l2 loss
        :param text_kw: name of text data
        :param label_kw: name of label data
        """
        ff = Dense(n_classes, activation='softmax')

        model = Sequential()

        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        model.add(Embedding(hashing_vectorizer.n_features,
                            n_dims,
                            input_length=n_features_to_use,
                            embeddings_regularizer=regularizers.l2(l2_weight)))

        # we add a GlobalAveragePooling1D, which will average the embeddings
        # of all words in the document
        model.add(GlobalAveragePooling1D())

        model.add(ff)

        loss_function = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

        model.compile(loss=loss_function,
                      optimizer='adam',
                      metrics=[])

        return FastText(label_mapper=label_mapper,
                        hashing_vectorizer=hashing_vectorizer,
                        keras_model=model,
                        n_features_to_use=n_features_to_use,
                        text_kw=text_kw,
                        label_kw=label_kw)

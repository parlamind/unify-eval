import abc
import warnings
from abc import ABC
from collections import defaultdict
from typing import List, Set, Dict, Callable

warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import accuracy_score
from skopt import gp_minimize
from skopt.space import Dimension

from unifyeval.model.deep_model import DeepModel
from unifyeval.model.mixins.classification import Classifier
from unifyeval.model.types import Label, Tensor
from unifyeval.training.callback import TrainerCallback
from unifyeval.training.trainer import Trainer
from unifyeval.utils.load_data import KeyedBatchLoader


class RawIsolatedEvaluation:
    """
    copy of the evaluation callback, but instead of the model it returns the evaluation results as a dict
    """

    def __init__(self,
                 data_loader: KeyedBatchLoader,
                 junk_label: Label = None,
                 junk_threshold: float = None,
                 labels_to_evaluate: Set[int] = None,
                 text_kw: str = "clauses",
                 minibatch_size: int = 16):
        self.junk_label = junk_label
        self.junk_threshold = junk_threshold
        self.data_loader = data_loader
        self.labels_to_evaluate = labels_to_evaluate
        self.minibatch_size = minibatch_size
        self.text_kw = text_kw

    def __call__(self, model: Classifier, *args, **kwargs):

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

        return dict(
            precision=precision,
            recall=recall,
            f1=f1
        )


class OptimizableTrainer(Trainer, DeepModel):

    def __init__(self,
                 data_loader: KeyedBatchLoader,
                 minibatch_callbacks: List[TrainerCallback],
                 batch_callbacks: List[TrainerCallback],
                 hyper_params_initialization: List[Dimension],
                 hyper_params_training: List[Dimension],
                 initiate_model: Callable[..., DeepModel],
                 evaluate_model: Callable[..., float],
                 initial_hyper_params: List[object]
                 ):
        super().__init__(data_loader, minibatch_callbacks, batch_callbacks)
        self.hyper_params_initialization = hyper_params_initialization
        self.hyper_params_training = hyper_params_training
        self.initiate_model = initiate_model
        self.evaluate_model = evaluate_model
        self.current_result = dict()
        self.current_hyper_params = initial_hyper_params if initial_hyper_params else []
        self.run_name_counts = defaultdict(lambda: -1)

    def train(self, acquisition_function: str,
              n_hyperparam_iterations: int,
              **kwargs) -> "OptimizableTrainer":
        def func(hyperparam_values) -> float:
            # group hyper param values by usage
            hyper_param_values_initialization = hyperparam_values[:len(self.hyper_params_initialization)]
            hyper_param_values_training = hyperparam_values[len(self.hyper_params_initialization):]

            # turn list of hyper param values into dictionaries so they can be kwarg-ed
            hyper_param_values_initialization = dict((dim.name, value) for dim, value in
                                                     zip(self.hyper_params_initialization,
                                                         hyper_param_values_initialization))

            hyper_param_values_training = dict((dim.name, value) for dim, value in
                                               zip(self.hyper_params_training,
                                                   hyper_param_values_training))

            # initiate model with given hyper param values
            initial_model = self.initiate_model(**hyper_param_values_initialization, **kwargs)

            # prepare name of run for callbacks
            run_name = "_".join(
                [f"{name}{value}" for d in (hyper_param_values_initialization, hyper_param_values_training)
                 for name, value in d.items()])

            # in case of a hyperparam setting that is repeated, increase run index (with -1 as default)
            self.run_name_counts[run_name] += 1
            run_name += f"_run{self.run_name_counts[run_name]}"

            # train the model
            trained_model = self.train_model(model=initial_model,
                                             run_name=run_name,
                                             **kwargs,
                                             **hyper_param_values_training)
            # set global_step back to 0 for every callback
            for callbacks in (self.minibatch_callbacks, self.batch_callbacks):
                for callback in callbacks:
                    callback.global_step = 0
            # evaluate model
            return self.evaluate_model(model=trained_model, **kwargs)

        print(self.current_hyper_params)

        search_result = gp_minimize(
            func=func,
            dimensions=self.hyper_params_initialization + self.hyper_params_training,
            acq_func=acquisition_function,
            n_calls=n_hyperparam_iterations,
            x0=self.current_hyper_params if len(self.current_hyper_params) > 0 else None,
            verbose=True
        )

        # update hyper param search results
        self.current_result = search_result
        self.current_hyper_params = search_result["x"]
        return self

    def get_loss(self, **kwargs) -> Dict[str, Tensor]:
        model: DeepModel = kwargs.pop("model")
        return model.get_loss(**kwargs)

    def get_components(self) -> dict:
        return dict(
            data_loader=self.data_loader,
            minibatch_callbacks=self.minibatch_callbacks,
            batch_callbacks=self.batch_callbacks,
            hyper_params_initialization=self.hyper_params_initialization,
            hyper_params_training=self.hyper_params_training
        )

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        return dict()

    @classmethod
    def from_components(cls, **kwargs) -> "OptimizableTrainer":
        return cls(**kwargs)


class ModelEvaluator(ABC):
    """
    ABC for functions that need some data loader
    """

    def __init__(self, data_loader: KeyedBatchLoader):
        self.data_loader = data_loader

    @abc.abstractmethod
    def __call__(self, model: DeepModel, **kwargs) -> float:
        """
        evaluate some model with the provided data
        :param model:
        :return:
        """
        pass


class CheckLoss(ModelEvaluator):
    """
    Evaluates the cross entropy of a given model and data
    """

    def __init__(self, data_loader: KeyedBatchLoader, loss_name: str):
        super().__init__(data_loader)
        self.loss_name = loss_name

    def __call__(self, model: DeepModel, **kwargs) -> float:
        losses = []
        minibatch_sizes = []
        text_kw = kwargs["text_kw"]
        for minibatch in self.data_loader.yield_minibatches(minibatch_size=128):
            losses.append(model.get_loss(**minibatch)[self.loss_name])
            minibatch_sizes.append(len(minibatch[text_kw]))
        n_data = sum(minibatch_sizes)
        self.data_loader.reset()
        return float(((np.array(minibatch_sizes) / n_data) * np.array(losses)).sum())


class CheckCrossEntropy(CheckLoss):
    """
    Evaluates the cross entropy of a given model and data
    """

    def __init__(self, data_loader: KeyedBatchLoader):
        super().__init__(data_loader, loss_name="cross_entropy")


class CheckAccuracy(ModelEvaluator):
    """
    Evaluates the accuracy of a given model and data
    """

    def __call__(self, model: Classifier, **kwargs) -> float:
        y_true = []
        y_pred = []
        label_kw = kwargs["label_kw"]

        for minibatch in self.data_loader.yield_minibatches(minibatch_size=128):
            y_true.extend(minibatch[label_kw])
            y_pred.extend(model.predict_label(**minibatch))
        self.data_loader.reset()
        # negate accuracy as we minimize the loss
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        return -acc

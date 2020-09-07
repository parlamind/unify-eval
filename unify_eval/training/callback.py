import abc
import io
import os
from abc import ABC
from typing import List, Dict, Union

import imageio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler

from unify_eval.model.deep_model import DeepModel
from unify_eval.model.mixins.adversarial import SaliencyModel
from unify_eval.model.mixins.classification import Classifier, MessageLevelClassifier
from unify_eval.model.mixins.embedding import EmbeddingModel
from unify_eval.model.types import Label
from unify_eval.training.model_saver import QueuedModelSaver
from unify_eval.utils.corpus import group
from unify_eval.utils.data_collection import DataCollector
from unify_eval.utils.load_data import KeyedBatchLoader, KeyedSubsampledBatchLoader


class TrainerCallback(ABC):
    """
    ABC for callbacks during training
    """
    global_step: int = 0

    def prepare_run(self, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, model: DeepModel, i_minibatch: int, iteration: int, *args, **kwargs) -> DeepModel:
        pass


class ScheduleCallback(TrainerCallback):

    def __init__(self, scheduler: lr_scheduler._LRScheduler):
        self.scheduler = scheduler

    def __call__(self, model: DeepModel, *args, **kwargs):
        self.scheduler.step()
        return model


class TensorboardCallback(TrainerCallback, ABC):
    def __init__(self,
                 folder_path: str,
                 relative_path: str =""):
        self.folder_path = folder_path
        self.relative_path = relative_path
        self.writer = None

    def prepare_run(self, run_name: str, **kwargs):
        path = os.path.join(self.folder_path, run_name, self.relative_path)
        self.writer = SummaryWriter(logdir=path)


class PrintStatus(TrainerCallback):

    def __call__(self, model: DeepModel, i_minibatch: int, iteration: int, *args, **kwargs) -> DeepModel:
        print(f"iteration {iteration} minibatch {i_minibatch}")
        return model


class ModelSaverCallback(TrainerCallback):

    def __init__(self, model_saver: QueuedModelSaver):
        self.model_saver = model_saver

    def __call__(self, model: DeepModel, i_minibatch: int, iteration: int, *args, **kwargs) -> DeepModel:
        self.model_saver.add_model(model=model, tags=[str(iteration)])
        return model


class EvaluationMetric(ABC):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abc.abstractmethod
    def __call__(self, y_true: np.array, y_pred: np.array = None, **kwargs) -> float:
        pass


class Accuracy(EvaluationMetric):

    def __call__(self, y_true: np.array, y_pred: np.array = None, **kwargs) -> float:
        return accuracy_score(y_true=y_true, y_pred=y_pred, **self.kwargs, **kwargs)


class Precision(EvaluationMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs

    def __call__(self, y_true: np.array, y_pred: np.array = None, **kwargs) -> float:
        return precision_score(y_true=y_true, y_pred=y_pred, **self.kwargs, **kwargs)


class F1(EvaluationMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs

    def __call__(self, y_true: np.array, y_pred: np.array = None, **kwargs) -> float:
        return f1_score(y_true=y_true, y_pred=y_pred, **self.kwargs, **kwargs)


class Recall(EvaluationMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs

    def __call__(self, y_true: np.array, y_pred: np.array = None, **kwargs) -> float:
        return recall_score(y_true=y_true, y_pred=y_pred, **self.kwargs, **kwargs)


class MonitoringMetric(ABC):
    """
    ABC for metrics that also depend on a respective model
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abc.abstractmethod
    def __call__(self, model: DeepModel, **kwargs) -> Dict[str, float]:
        pass


class Loss(MonitoringMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs

    def __call__(self, model: DeepModel, **kwargs) -> Dict[str, float]:
        return model.get_loss(**self.kwargs, **kwargs)


class PlotParameters(TensorboardCallback):
    """
    Callback plotting histograms of all model parameters via tensorboard
    """

    def __init__(self, folder_path: str, relative_path: str = ""):
        """
        :param folder_path: path to folder containing all the tensorboard logs
        """
        super().__init__(folder_path, relative_path)

    def __call__(self, model: DeepModel, i_minibatch: int, iteration: int, *args, **kwargs) -> DeepModel:
        for name, parameter in model.get_numpy_parameters().items():
            self.writer.add_histogram(tag=name, values=parameter, global_step=self.global_step)
        return model


class PlotClassificationEmbeddings(TensorboardCallback):
    """
    Callback that embeds text classification data. Adds the following as meta data:
        - all input data
        - true label
        - predicted label
        - if prediction is correct for the given label
        - normalized entropy of label likelihood
    Data is stored for every call independently in a respective folder containing metadata and embeddings as tsv files.
    """

    def __init__(self, folder_path: str,
                 data_loader: KeyedBatchLoader,
                 relative_path: str = "",
                 minibatch_size: int = 512,
                 label_kw: str = "labels", progress_bar: bool = True):
        """
        :param folder_path: path to folder storing tensorboard data
        :param data_loader: loader for data to embed
        :param minibatch_size: minibatch size of embedding
        :param label_kw: name of label to evaluate
        :param progress_bar: if true, shows progressbar over minibatches

        """
        super().__init__(folder_path, relative_path)
        self.data_loader = data_loader
        self.header = list(data_loader.data.keys())
        self.label_kw = label_kw
        self.minibatch_size = minibatch_size
        self.progress_bar = progress_bar

    def normalized_entropy(self, prob_matrix: np.ndarray):
        return entropy(prob_matrix.transpose()) / np.log(prob_matrix.shape[-1])

    def __call__(self, model: Union[Classifier, EmbeddingModel], i_minibatch: int, iteration: int, *args,
                 **kwargs) -> DeepModel:
        print("embedding ...")

        data_keys = list((key for key in self.data_loader.data.keys() if key != self.label_kw))
        additional_keys = ["correctly_classified",
                           f"{self.label_kw}_true",
                           f"{self.label_kw}_pred",
                           "normalized_entropy"]

        sorted_keys = sorted(data_keys + additional_keys)

        keyed_meta_data = dict((key, []) for key in sorted_keys)

        embeddings = []

        for keyed_minibatch in self.data_loader.yield_minibatches(minibatch_size=self.minibatch_size,
                                                                  progress_bar=self.progress_bar):
            for key, data in keyed_minibatch.items():
                if key != self.label_kw:
                    keyed_meta_data[key].extend(data)
                else:
                    keyed_meta_data[f"{self.label_kw}_true"].extend(data)
            embeddings.extend(model.embed(**keyed_minibatch, **kwargs))

            y_pred_tmp_probs = model.predict_label_probabilities(**keyed_minibatch, **kwargs)
            y_pred_tmp = model.label_mapper.map_to_actual_labels(np.argmax(y_pred_tmp_probs, axis=-1))
            keyed_meta_data[f"{self.label_kw}_pred"].extend(y_pred_tmp)
            keyed_meta_data["correctly_classified"].extend(keyed_minibatch[self.label_kw] == y_pred_tmp)
            keyed_meta_data["normalized_entropy"].extend(self.normalized_entropy(prob_matrix=y_pred_tmp_probs))

        meta_data = list(zip(*[keyed_meta_data[key] for key in sorted_keys]))

        self.writer.add_embedding(mat=embeddings,
                                  metadata_header=sorted_keys,
                                  metadata=meta_data,
                                  global_step=self.global_step,
                                  tag="classification_embeddings")

        self.data_loader.reset()
        return model


class EvaluationCallBack(TensorboardCallback):
    """
    Runs evaluation on a simple text classification.
    Plots line plots with results over time via tensorboard and additionally saves results to a json
    """

    def __init__(self,
                 folder_path: str,
                 data_loader: Union[KeyedBatchLoader, KeyedSubsampledBatchLoader],
                 monitoring_metrics: List[MonitoringMetric],
                 evaluation_metrics: Dict[str, EvaluationMetric],
                 relative_path: str = "",
                 minibatch_size: int = 512,
                 label_kw: str = "labels",
                 progress_bar: bool = True,
                 junk_threshold: float = None,
                 junk_label_index: int = None):
        """
        :param folder_path: path to evaluation results
        :param data_loader: data loader for test data
        :param monitoring_metrics: list of MonitoringMetric objects to evaluate in that order
        :param evaluation_metrics: dict from names to EvaluationMetric objects to evaluate.
        :param minibatch_size: minibatch size to use when evaluating models
        :param label_kw: name of label to evaluate
        :param progress_bar: if true, shows progressbar over minibatches
        :param junk_threshold: if not None, predictions with label probabilities lower than this threshold are turned into junk predictions
        :param junk_label_index: if not None, predictions with label probabilities lower than junk_threshold are mapped into this label index
        """
        super().__init__(folder_path, relative_path)
        if not os.path.exists(folder_path):
            print(f"writing folder for evaluation callback {folder_path}")
            os.makedirs(folder_path)
        self.folder_path = folder_path
        self.data_loader = data_loader
        self.monitoring_metrics = monitoring_metrics
        self.evaluation_metrics = evaluation_metrics
        self.data_collector = DataCollector()
        self.minibatch_size = minibatch_size
        self.label_kw = label_kw
        self.progress_bar = progress_bar
        self.junk_threshold = junk_threshold
        self.junk_label_index = junk_label_index

    def __call__(self, model: Union[Classifier, EmbeddingModel], i_minibatch: int, iteration: int, *args,
                 **kwargs) -> DeepModel:

        print("evaluating ...")

        # initialize dictionary of metric keys to list of single metric scores for every minibatch
        collected_metrics = dict()
        full_data_size = 0

        # fill dict with evaluation scores
        # metrics are calculated for minibatches and then weighted according to the respective minibatch size
        # i.e. sum(p(i_minibatch)*metric)

        for keyed_minibatch in self.data_loader.yield_minibatches(minibatch_size=self.minibatch_size,
                                                                  progress_bar=self.progress_bar):
            y_true_tmp = keyed_minibatch[self.label_kw]
            full_data_size += len(y_true_tmp)
            for name, metric in self.evaluation_metrics.items():
                if name not in collected_metrics:
                    collected_metrics[name] = {
                        "collected": [],
                        "minibatch_size": []
                    }
                collected_metrics[name]["collected"].append(metric(y_true=y_true_tmp,
                                                                   y_pred=model.predict_label(
                                                                       junk_threshold=self.junk_threshold,
                                                                       junk_label_index=self.junk_label_index,
                                                                       **kwargs,
                                                                       **keyed_minibatch)))
                collected_metrics[name]["minibatch_size"].append(len(y_true_tmp))

            # fill dict with monitoring scores
            for metric in self.monitoring_metrics:
                for name, value in metric(model=model, **keyed_minibatch, **kwargs).items():
                    if name not in collected_metrics:
                        collected_metrics[name] = {
                            "collected": [],
                            "minibatch_size": []
                        }
                    collected_metrics[name]["collected"].append(value)
                    collected_metrics[name]["minibatch_size"].append(len(y_true_tmp))

        # average minibatch data to get batch expectation
        evaluation_results = dict((k,
                                   float(((np.array(v["minibatch_size"]) / full_data_size) * np.array(
                                       v["collected"])).sum()))
                                  for k, v in collected_metrics.items())

        # add every update to tensorboard viz
        for name, value in evaluation_results.items():
            self.writer.add_scalar(tag=name, scalar_value=value, global_step=self.global_step)

        # add all to evaluation_results.json
        self.data_collector.add_data(**evaluation_results).to_json(
            path=os.path.join(self.folder_path, "evaluation_results.json"))

        # reset data loader (duh)
        self.data_loader.reset()

        return model

    @staticmethod
    def default(folder_path: str,
                data_loader: Union[KeyedBatchLoader, KeyedSubsampledBatchLoader],
                label_indices: Union[List[int], np.ndarray],
                relative_path: str = "",
                minibatch_size: int = 265,
                junk_threshold: float = None,
                junk_label_index: int = None,
                ) -> "EvaluationCallBack":

        return EvaluationCallBack(
            folder_path=folder_path,
            relative_path=relative_path,
            data_loader=data_loader,
            monitoring_metrics=[
                Loss(all_labels=label_indices)
            ],
            evaluation_metrics={
                "accuracy": Accuracy(),
                "macro_precision": Precision(average="macro", labels=label_indices),
                "weighted_precision": Precision(average="weighted", labels=label_indices),
                "macro_recall": Recall(average="macro", labels=label_indices),
                "weighted_recall": Recall(average="weighted", labels=label_indices),
                "macro_f1": F1(average="macro", labels=label_indices),
                "weighted_f1": F1(average="weighted", labels=label_indices),
            },
            minibatch_size=minibatch_size,
            junk_threshold=junk_threshold,
            junk_label_index=junk_label_index
        )


class MessageLevelEvaluationCallBack(TensorboardCallback):
    """
    Runs evaluation on text classification over entire message, with one label per clause.
    Plots line plots with results over time via tensorboard and additionally saves results to a json
    """

    def __init__(self,
                 folder_path: str,
                 data_loader: Union[KeyedBatchLoader, KeyedSubsampledBatchLoader],
                 monitoring_metrics: List[MonitoringMetric],
                 evaluation_metrics: Dict[str, EvaluationMetric],
                 relative_path: str = "",
                 minibatch_size: int = 512,
                 label_kw: str = "labels",
                 progress_bar: bool = True,
                 junk_threshold: float = None,
                 junk_label_index: int = None):
        """
        :param folder_path: path to evaluation results
        :param data_loader: data loader for test data
        :param monitoring_metrics: list of MonitoringMetric objects to evaluate in that order
        :param evaluation_metrics: dict from names to EvaluationMetric objects to evaluate.
        :param minibatch_size: minibatch size to use when evaluating models
        :param label_kw: name of label to evaluate
        :param progress_bar: if true, shows progressbar over minibatches
        :param junk_threshold: if not None, predictions with label probabilities lower than this threshold are turned into junk predictions
        :param junk_label_index: if not None, predictions with label probabilities lower than junk_threshold are mapped into this label index
        """
        super().__init__(folder_path, relative_path)
        if not os.path.exists(folder_path):
            print(f"writing folder for evaluation callback {folder_path}")
            os.makedirs(folder_path)
        self.folder_path = folder_path
        self.data_loader = data_loader
        self.monitoring_metrics = monitoring_metrics
        self.evaluation_metrics = evaluation_metrics
        self.data_collector = DataCollector()
        self.minibatch_size = minibatch_size
        self.label_kw = label_kw
        self.progress_bar = progress_bar
        self.junk_threshold = junk_threshold
        self.junk_label_index = junk_label_index

    def __call__(self, model: Union[MessageLevelClassifier], i_minibatch: int, iteration: int, *args,
                 **kwargs) -> DeepModel:

        print("evaluating ...")

        # initialize dictionary of metric keys to list of single metric scores for every minibatch
        collected_metrics = dict()
        full_data_size = 0

        # fill dict with evaluation scores
        # metrics are calculated for minibatches and then weighted according to the respective minibatch size
        # i.e. sum(p(i_minibatch)*metric)

        for keyed_minibatch in self.data_loader.yield_minibatches(minibatch_size=self.minibatch_size,
                                                                  progress_bar=self.progress_bar):
            flattened_target = [label for labels_per_message in keyed_minibatch[self.label_kw]
                                for label in labels_per_message]
            full_data_size += len(flattened_target)
            flattened_predictions = [label for labels_per_message in model.predict_labels(
                junk_threshold=self.junk_threshold,
                junk_label_index=self.junk_label_index,
                **kwargs,
                **keyed_minibatch) for label in labels_per_message]
            for name, metric in self.evaluation_metrics.items():
                if name not in collected_metrics:
                    collected_metrics[name] = {
                        "collected": [],
                        "minibatch_size": []
                    }
                collected_metrics[name]["collected"].append(metric(y_true=flattened_target,
                                                                   y_pred=flattened_predictions))
                collected_metrics[name]["minibatch_size"].append(len(flattened_target))

            # fill dict with monitoring scores
            for metric in self.monitoring_metrics:
                for name, value in metric(model=model, **keyed_minibatch, **kwargs).items():
                    if name not in collected_metrics:
                        collected_metrics[name] = {
                            "collected": [],
                            "minibatch_size": []
                        }
                    collected_metrics[name]["collected"].append(value)
                    collected_metrics[name]["minibatch_size"].append(len(flattened_target))

        # average minibatch data to get batch expectation
        evaluation_results = dict((k,
                                   float(((np.array(v["minibatch_size"]) / full_data_size) * np.array(
                                       v["collected"])).sum()))
                                  for k, v in collected_metrics.items())

        # add every update to tensorboard viz
        for name, value in evaluation_results.items():
            self.writer.add_scalar(tag=name, scalar_value=value, global_step=self.global_step)

        # add all to evaluation_results.json
        self.data_collector.add_data(**evaluation_results).to_json(
            path=os.path.join(self.folder_path, "evaluation_results.json"))

        # reset data loader (duh)
        self.data_loader.reset()

        return model

    @staticmethod
    def default(folder_path: str,
                data_loader: Union[KeyedBatchLoader, KeyedSubsampledBatchLoader],
                label_indices: Union[List[int], np.ndarray],
                relative_path: str = "",
                minibatch_size: int = 265,
                junk_threshold: float = None,
                junk_label_index: int = None,
                ) -> "MessageLevelEvaluationCallBack":

        return MessageLevelEvaluationCallBack(
            folder_path=folder_path,
            relative_path=relative_path,
            data_loader=data_loader,
            monitoring_metrics=[
                Loss(all_labels=label_indices)
            ],
            evaluation_metrics={
                "accuracy": Accuracy(),
                "macro_precision": Precision(average="macro", labels=label_indices),
                "weighted_precision": Precision(average="weighted", labels=label_indices),
                "macro_recall": Recall(average="macro", labels=label_indices),
                "weighted_recall": Recall(average="weighted", labels=label_indices),
                "macro_f1": F1(average="macro", labels=label_indices),
                "weighted_f1": F1(average="weighted", labels=label_indices),
            },
            minibatch_size=minibatch_size,
            junk_threshold=junk_threshold,
            junk_label_index=junk_label_index
        )


class CheckNaN(TrainerCallback):
    """
    Callback that checks if any parameter contains NaNs. If so, it stops the current training.
    """

    def __call__(self, model: DeepModel, i_minibatch: int, iteration: int, *args, **kwargs) -> DeepModel:
        for key, parameter in model.get_numpy_parameters().items():
            assert not np.isnan(parameter).any(), f"parameter {key} contains NaN values!"
        return model


class PlottingCallback(TensorboardCallback):
    """
    Subclass that registers a tensorboard summary writer
    """

    def __init__(self, folder_path: str, relative_path: str = ""):
        super().__init__(folder_path, relative_path)

    def _current_plot_to_image(self):
        """Converts the current matplotlib plot to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = imageio.imread(buf)
        buf.close()
        plt.close()
        return image

    @abc.abstractmethod
    def __call__(self, model: DeepModel, i_minibatch: int, iteration: int, *args, **kwargs) -> DeepModel:
        pass


class ConfusionMatrixCallback(PlottingCallback):
    """
    Callback that writes a temporary png of a confusion matrix and than adds it to tensorboard
    """

    def __init__(self, folder_path: str,
                 data_loader: KeyedBatchLoader,
                 label_array: np.array,
                 label_kw: str = "labels",
                 relative_path: str = "",
                 minibatch_size: int = 256,
                 junk_threshold: float = None,
                 junk_label_index: int = None,
                 tag: str = "confusion matrix",
                 title: str = None,
                 progress_bar: bool = True):
        super().__init__(folder_path=folder_path, relative_path=relative_path)
        self.folder_path = folder_path
        self.data_loader = data_loader
        self.label_array = label_array
        self.label_kw = label_kw
        self.minibatch_size = minibatch_size
        self.junk_threshold = junk_threshold
        self.junk_label_index = junk_label_index
        self.tag = tag
        self.title = self.tag if title is None else title
        self.progress_bar = progress_bar

    def __call__(self, model: Classifier, i_minibatch: int, iteration: int, *args, **kwargs) -> DeepModel:
        print("calculating confusion matrix ...")
        y_true = []
        y_pred = []

        for keyed_batch in self.data_loader.yield_minibatches(minibatch_size=self.minibatch_size,
                                                              progress_bar=self.progress_bar):
            y_true.extend(keyed_batch[self.label_kw])
            y_pred.extend(model.predict_label(junk_label_index=self.junk_label_index,
                                              junk_threshold=self.junk_threshold,
                                              **keyed_batch, **kwargs))

        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=self.label_array)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(self.title)
        plt.colorbar()
        tick_marks = np.arange(len(self.label_array))
        plt.xticks(tick_marks, self.label_array, rotation=90, fontsize=8)
        plt.yticks(tick_marks, self.label_array, fontsize=8)
        image = self._current_plot_to_image()
        self.writer.add_image(tag=self.tag,
                              img_tensor=image,
                              dataformats="HWC",
                              global_step=self.global_step)
        self.data_loader.reset()
        return model


class MessageLevelConfusionMatrixCallback(PlottingCallback):
    """
    Callback that writes a temporary png of a confusion matrix and than adds it to tensorboard
    """

    def __init__(self,
                 folder_path: str,
                 data_loader: KeyedBatchLoader,
                 label_array: np.array,
                 label_kw: str = "labels",
                 minibatch_size: int = 256,
                 junk_threshold: float = None,
                 junk_label_index: int = None,
                 tag: str = "confusion matrix",
                 title: str = None,
                 progress_bar: bool = True):
        super().__init__(folder_path)
        self.folder_path = folder_path
        self.data_loader = data_loader
        self.label_array = label_array
        self.label_kw = label_kw
        self.minibatch_size = minibatch_size
        self.junk_threshold = junk_threshold
        self.junk_label_index = junk_label_index
        self.tag = tag
        self.title = self.tag if title is None else title
        self.progress_bar = progress_bar

    def __call__(self, model: MessageLevelClassifier, i_minibatch: int, iteration: int, *args, **kwargs) -> DeepModel:
        print("calculating confusion matrix ...")
        y_true = []
        y_pred = []

        for keyed_batch in self.data_loader.yield_minibatches(minibatch_size=self.minibatch_size,
                                                              progress_bar=self.progress_bar):
            true_labels = [label for labels_per_message in keyed_batch[self.label_kw] for label in labels_per_message]
            predicted_labels = [label for labels_per_message in model.predict_labels(
                junk_label_index=self.junk_label_index,
                junk_threshold=self.junk_threshold,
                **keyed_batch, **kwargs) for label in labels_per_message
                                ]
            y_true.extend(true_labels)
            y_pred.extend(predicted_labels)

        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=self.label_array)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(self.title)
        plt.colorbar()
        tick_marks = np.arange(len(self.label_array))
        plt.xticks(tick_marks, self.label_array, rotation=90, fontsize=8)
        plt.yticks(tick_marks, self.label_array, fontsize=8)
        image = self._current_plot_to_image()
        self.writer.add_image(tag=self.tag,
                              img_tensor=image,
                              dataformats="HWC",
                              global_step=self.global_step)
        self.data_loader.reset()
        return model


class TestForMinimalPairs(PlottingCallback):
    """
    Callback to test minimal pairs (duh). Expects a category name (such as negation, tense, etc.)
    and a dictionary of batch loaders, with its keys being the respective labels to check the category for.
    For every example, plots probability of junk label, actual label and other labels combined as a horizontal barplot.
    Also draws line plots of accuracies over the respective data.
    """

    def __init__(self,
                 folder_path: str,
                 category_name: str,
                 data_loaders_per_label: Dict[Label, KeyedBatchLoader],
                 relative_path: str = "",
                 text_kw: str = "texts",
                 label_kw: str = "labels",
                 opposing_label: str = "-1",
                 minibatch_size=256,
                 progress_bar: bool = True):
        """
        :param folder_path: path to folder where tensorboard checkpoints are stored
        :param folder_path: relative path from folder to checkpoint files
        :param category_name: category of minimal pairs, e.g. negation, tense, ...
        :param data_loaders_per_label: dictionary from labels to corresponding data loader
        :param text_kw: keyword in data loader for input text (used for plotting)
        :param label_kw: keyword in data loader for label
        :param opposing_label: label to use as opponent to the given true label. Only used if data does not contain any such datapoints.
        :param minibatch_size: minibatch size during inference of label probability
        :param progress_bar: if true, print tqdm progress bar during inference
        """
        super().__init__(folder_path,relative_path)
        self.folder_path = folder_path
        self.category_name = category_name
        self.data_loaders_per_label = data_loaders_per_label
        self.text_kw = text_kw
        self.label_kw = label_kw
        self.opposing_label = opposing_label
        self.minibatch_size = minibatch_size
        self.progress_bar = progress_bar
        self._writer = SummaryWriter(logdir=folder_path)

    def _plot_minimal_pair_data(self,
                                texts: List[str],
                                labels: List[str],
                                y_probs: np.ndarray,
                                positive_label: str,
                                negative_label: str = "-1",
                                other_label: str = "other",
                                fontsize: int = 8,
                                title: str = "minimal pairs"):
        def plot(texts, y_probs, category_names):
            y_probs_cum = y_probs.cumsum(axis=1)
            category_colors = plt.get_cmap('RdYlGn')(
                np.linspace(0.15, 0.85, y_probs.shape[1]))

            fig, ax = plt.subplots(figsize=(10, 1 + 0.4 * len(y_probs)))
            ax.invert_yaxis()
            ax.xaxis.set_visible(False)
            ax.set_xlim(0, np.sum(y_probs, axis=1).max())
            ax.set_ylim(len(texts), -0.5)

            for i, (colname, color) in enumerate(zip(category_names, category_colors)):
                widths = y_probs[:, i]
                starts = y_probs_cum[:, i] - widths
                ax.barh(np.arange(len(texts)) + 0.5, widths, left=starts, height=0.5,
                        label=colname, color=color)
                xcenters = starts + widths / 2

                r, g, b, _ = color
                text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
                for y, (x, w) in enumerate(zip(xcenters, widths)):
                    ax.text(x, y + 0.5, f"{np.round(float(w), 2)}", ha='center', va='center',
                            color=text_color,
                            fontsize=fontsize)
                for i, (text, label) in enumerate(zip(texts, labels)):
                    ax.annotate(text, (0.5, i),
                                ha="center",
                                va="center",
                                color="black" if label == positive_label else "red")
            ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
                      loc='lower left', fontsize='small')
            ax.set_xticks([], [])
            ax.set_title(title)
            return fig, ax

        plot(texts=texts,
             y_probs=y_probs,
             category_names=[negative_label, other_label, positive_label])

    def __call__(self, model: Classifier, i_minibatch: int, iteration: int, *args, **kwargs) -> DeepModel:
        print(f"checking minimal pairs for category {self.category_name} ...")
        for name, data_loader in self.data_loaders_per_label.items():
            print(f"checking minimal pairs for label {name} ...")
            y_pred_probs = []
            y_pred = []
            y_true = []
            texts = []
            for keyed_batch in data_loader.yield_minibatches(minibatch_size=self.minibatch_size,
                                                             progress_bar=self.progress_bar):
                y_true.extend(str(label) for label in keyed_batch[self.label_kw])
                texts.extend(keyed_batch[self.text_kw])
                y_pred_probs.extend(model.predict_label_probabilities(**keyed_batch, **kwargs))
                y_pred.extend(model.predict_label(**keyed_batch, **kwargs))
            y_pred_probs = np.array(y_pred_probs)
            other_labels = list({y for y in y_true if str(y) != name})
            negative_label = str(other_labels[0]) if len(other_labels) != 0 else self.opposing_label
            positive_label_index = model.label_mapper.actuallabel2index[int(name)]
            negative_label_index = model.label_mapper.actuallabel2index[int(negative_label)]
            other_indices = list(
                {i for i in model.label_mapper.all_indices if i not in {positive_label_index, negative_label_index}})
            positive_probs = y_pred_probs[:, positive_label_index].reshape((-1, 1))
            negative_probs = y_pred_probs[:, negative_label_index].reshape((-1, 1))
            other_probs = y_pred_probs[:, other_indices].sum(axis=-1).reshape((-1, 1))
            merged_probs = np.hstack((negative_probs, other_probs, positive_probs))

            self._writer.add_scalar(tag=f"{self.category_name}_{name}_accuracy",
                                    scalar_value=accuracy_score(y_true=[int(y) for y in y_true], y_pred=y_pred),
                                    global_step=self.global_step)

            self._plot_minimal_pair_data(
                texts=texts,
                labels=y_true,
                y_probs=merged_probs,
                positive_label=str(name),
                negative_label=negative_label,
                other_label="other",
                title=f"minimal pairs {self.category_name}\n{name}"
            )
            img = self._current_plot_to_image()
            self._writer.add_image(tag=f"minimal_pairs_{self.category_name}_{name}",
                                   img_tensor=img,
                                   dataformats="HWC",
                                   global_step=self.global_step)

            data_loader.reset()
            return model


class MessageLevelTestForMinimalPairs(PlottingCallback):
    """
    Callback to test minimal pairs (duh). Expects a category name (such as negation, tense, etc.)
    and a dictionary of batch loaders, with its keys being the respective labels to check the category for.
    For every example, plots probability of junk label, actual label and other labels combined as a horizontal barplot.
    Also draws line plots of accuracies over the respective data.
    """

    def __init__(self,
                 folder_path: str,
                 category_name: str,
                 data_loaders_per_label: Dict[Label, KeyedBatchLoader],
                 text_kw: str = "texts",
                 label_kw: str = "labels",
                 opposing_label: str = "-1",
                 minibatch_size=256,
                 progress_bar: bool = True):
        """
        :param folder_path: path to folder where tensorboard checkpoints are stored
        :param category_name: category of minimal pairs, e.g. negation, tense, ...
        :param data_loaders_per_label: dictionary from labels to corresponding data loader
        :param text_kw: keyword in data loader for input text (used for plotting)
        :param label_kw: keyword in data loader for label
        :param opposing_label: label to use as opponent to the given true label. Only used if data does not contain any such datapoints.
        :param minibatch_size: minibatch size during inference of label probability
        :param progress_bar: if true, print tqdm progress bar during inference
        """
        super().__init__(folder_path)
        self.folder_path = folder_path
        self.category_name = category_name
        self.data_loaders_per_label = data_loaders_per_label
        self.text_kw = text_kw
        self.label_kw = label_kw
        self.opposing_label = opposing_label
        self.minibatch_size = minibatch_size
        self.progress_bar = progress_bar

    def _plot_minimal_pair_data(self,
                                texts: List[str],
                                labels: List[str],
                                y_probs: np.ndarray,
                                positive_label: str,
                                negative_label: str = "-1",
                                other_label: str = "other",
                                fontsize: int = 8,
                                title: str = "minimal pairs"):
        def plot(texts, y_probs, category_names):
            y_probs_cum = y_probs.cumsum(axis=1)
            category_colors = plt.get_cmap('RdYlGn')(
                np.linspace(0.15, 0.85, y_probs.shape[1]))

            fig, ax = plt.subplots(figsize=(10, 1 + 0.4 * len(y_probs)))
            ax.invert_yaxis()
            ax.xaxis.set_visible(False)
            ax.set_xlim(0, np.sum(y_probs, axis=1).max())
            ax.set_ylim(len(texts), -0.5)

            for i, (colname, color) in enumerate(zip(category_names, category_colors)):
                widths = y_probs[:, i]
                starts = y_probs_cum[:, i] - widths
                ax.barh(np.arange(len(texts)) + 0.5, widths, left=starts, height=0.5,
                        label=colname, color=color)
                xcenters = starts + widths / 2

                r, g, b, _ = color
                text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
                for y, (x, w) in enumerate(zip(xcenters, widths)):
                    ax.text(x, y + 0.5, f"{np.round(float(w), 2)}", ha='center', va='center',
                            color=text_color,
                            fontsize=fontsize)
                for i, (text, label) in enumerate(zip(texts, labels)):
                    ax.annotate(text, (0.5, i),
                                ha="center",
                                va="center",
                                color="black" if label == positive_label else "red")
            ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
                      loc='lower left', fontsize='small')
            ax.set_xticks([], [])
            ax.set_title(title)
            return fig, ax

        plot(texts=texts,
             y_probs=y_probs,
             category_names=[negative_label, other_label, positive_label])

    def __call__(self, model: MessageLevelClassifier, i_minibatch: int, iteration: int, *args, **kwargs) -> DeepModel:
        print(f"checking minimal pairs for category {self.category_name} ...")
        for name, data_loader in self.data_loaders_per_label.items():
            print(f"checking minimal pairs for label {name} ...")
            y_pred_probs = []
            y_pred = []
            y_true = []
            texts = []
            for keyed_batch in data_loader.yield_minibatches(minibatch_size=self.minibatch_size,
                                                             progress_bar=self.progress_bar):
                keyed_batch[self.text_kw] = [[text] for text in keyed_batch[self.text_kw]]
                y_true.extend(str(label) for label in keyed_batch[self.label_kw])
                texts.extend(keyed_batch[self.text_kw])
                y_pred_probs.extend(probs_per_message[0] for probs_per_message in
                                    model.predict_label_probabilities(**keyed_batch, **kwargs))
                # take first prediction for every message (as input is always a singleton clause anyway)
                y_pred.extend(
                    labels_per_message[0] for labels_per_message in model.predict_labels(**keyed_batch, **kwargs))
            y_pred_probs = np.array(y_pred_probs)
            other_labels = list({y for y in y_true if str(y) != name})
            negative_label = str(other_labels[0]) if len(other_labels) != 0 else self.opposing_label
            positive_label_index = model.label_mapper.actuallabel2index[int(name)]
            negative_label_index = model.label_mapper.actuallabel2index[int(negative_label)]
            other_indices = list(
                {i for i in model.label_mapper.all_indices if i not in {positive_label_index, negative_label_index}})
            positive_probs = y_pred_probs[:, positive_label_index].reshape((-1, 1))
            negative_probs = y_pred_probs[:, negative_label_index].reshape((-1, 1))
            other_probs = y_pred_probs[:, other_indices].sum(axis=-1).reshape((-1, 1))
            merged_probs = np.hstack((negative_probs, other_probs, positive_probs))

            self.writer.add_scalar(tag=f"{self.category_name}_{name}_accuracy",
                                   scalar_value=accuracy_score(y_true=[int(y) for y in y_true], y_pred=y_pred),
                                   global_step=self.global_step)

            self._plot_minimal_pair_data(
                texts=texts,
                labels=y_true,
                y_probs=merged_probs,
                positive_label=str(name),
                negative_label=negative_label,
                other_label="other",
                title=f"minimal pairs {self.category_name}\n{name}"
            )
            img = self._current_plot_to_image()
            self.writer.add_image(tag=f"minimal_pairs_{self.category_name}_{name}",
                                  img_tensor=img,
                                  dataformats="HWC",
                                  global_step=self.global_step)

            data_loader.reset()
            return model


class LabelSpecificEvaluation(PlottingCallback):
    """
    Callback that evaluates performance label per label.
    Draws line plots for precision, recall and f1 for every label.
    Also draws a scatter plot of labels, with precision and recall as axes.
    """

    def __init__(self,
                 folder_path: str,
                 data_loader: KeyedBatchLoader,
                 relative_path: str = "",
                 label_kw: str = "labels",
                 minibatch_size=256,
                 progress_bar: bool = True,
                 print_results: bool = False):
        """
        :param folder_path: path to tensorboard checkpoint files
        :param data_loader: data loader yielding some evaluation data
        :param label_kw: keyword of label to predict
        :param minibatch_size: mibatch size during inference
        :param progress_bar: if true, print tqdm progress bar during inference
        """
        super().__init__(folder_path=folder_path,
                         relative_path=relative_path)
        self.folder_path = folder_path
        self.data_loader = data_loader
        self.label_kw = label_kw
        self.minibatch_size = minibatch_size
        self.progress_bar = progress_bar
        self.print_results = print_results

    def __call__(self, model: Classifier, i_minibatch: int, iteration: int, *args, **kwargs) -> DeepModel:
        print(f"creating scatter plot for labels ...")
        y_pred = []
        y_true = []
        for keyed_batch in self.data_loader.yield_minibatches(minibatch_size=self.minibatch_size,
                                                              progress_bar=self.progress_bar):
            y_true.extend(keyed_batch[self.label_kw])
            y_pred.extend(model.predict_label(**keyed_batch, **kwargs))
        report = classification_report(y_true=y_true,
                                       y_pred=y_pred,
                                       output_dict=True,
                                       labels=model.label_mapper.labels)
        labels = []
        precisions = []
        recalls = []
        support = []

        for label in model.label_mapper.labels:
            precision = report[str(label)]["precision"]
            recall = report[str(label)]["recall"]
            f1 = report[str(label)]["f1-score"]

            self.writer.add_scalar(tag=f"precision {label}", scalar_value=precision, global_step=self.global_step)
            self.writer.add_scalar(tag=f"recall {label}", scalar_value=recall, global_step=self.global_step)
            self.writer.add_scalar(tag=f"f1 {label}", scalar_value=f1, global_step=self.global_step)
            labels.append(label)
            precisions.append(precision)
            recalls.append(recall)
            support.append(report[str(label)]["support"])

        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=precisions,
                        y=recalls,
                        size=support,
                        hue=support,
                        sizes=(min(support), max(support)),
                        palette="coolwarm")
        plt.grid()
        for label, x, y in zip(labels, precisions, recalls):
            plt.annotate(label, (x, y))
            if self.print_results:
                f1 = 0 if x == 0 or y == 0 else 2 * ((x * y) / (x + y))
                print(f"label {label} precision {x} recall {y} f1 {f1}")
        plt.xlabel("precision")
        plt.ylabel("recall")
        plt.title("precision and recall per label")
        plt.xlim((0.0, 1.0))
        plt.ylim((0.0, 1.0))
        self.writer.add_image(tag=f"labelwise_evaluation",
                              img_tensor=self._current_plot_to_image(),
                              dataformats="HWC",
                              global_step=self.global_step)
        self.data_loader.reset()
        return model


class MessageLevelLabelSpecificEvaluation(PlottingCallback):
    """
    Callback that evaluates performance label per label.
    Draws line plots for precision, recall and f1 for every label.
    Also draws a scatter plot of labels, with precision and recall as axes.
    """

    def __init__(self,
                 folder_path: str,
                 data_loader: KeyedBatchLoader,
                 relative_path: str = "",
                 label_kw: str = "labels",
                 minibatch_size=256,
                 progress_bar: bool = True):
        """
        :param folder_path: path to tensorboard checkpoint files
        :param data_loader: data loader yielding some evaluation data
        :param label_kw: keyword of label to predict
        :param minibatch_size: mibatch size during inference
        :param progress_bar: if true, print tqdm progress bar during inference
        """
        super().__init__(folder_path=folder_path,
                         relative_path=relative_path)
        self.folder_path = folder_path
        self.data_loader = data_loader
        self.label_kw = label_kw
        self.minibatch_size = minibatch_size
        self.progress_bar = progress_bar

    def __call__(self, model: MessageLevelClassifier, i_minibatch: int, iteration: int, *args, **kwargs) -> DeepModel:
        print(f"creating scatter plot for labels ...")
        y_pred = []
        y_true = []
        for keyed_batch in self.data_loader.yield_minibatches(minibatch_size=self.minibatch_size,
                                                              progress_bar=self.progress_bar):
            y_true.extend(label for labels_per_message in keyed_batch[self.label_kw]
                          for label in labels_per_message)
            y_pred.extend(label for labels_per_message in model.predict_labels(**keyed_batch, **kwargs)
                          for label in labels_per_message)
        report = classification_report(y_true=y_true,
                                       y_pred=y_pred,
                                       output_dict=True,
                                       labels=model.label_mapper.labels)
        labels = []
        precisions = []
        recalls = []
        support = []

        for label in model.label_mapper.labels:
            precision = report[str(label)]["precision"]
            recall = report[str(label)]["recall"]
            f1 = report[str(label)]["f1-score"]

            self.writer.add_scalar(tag=f"precision {label}", scalar_value=precision, global_step=self.global_step)
            self.writer.add_scalar(tag=f"recall {label}", scalar_value=recall, global_step=self.global_step)
            self.writer.add_scalar(tag=f"f1 {label}", scalar_value=f1, global_step=self.global_step)
            labels.append(label)
            precisions.append(precision)
            recalls.append(recall)
            support.append(report[str(label)]["support"])

        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=precisions,
                        y=recalls,
                        size=support,
                        hue=support,
                        sizes=(min(support), max(support)),
                        palette="coolwarm")
        plt.grid()
        for label, x, y in zip(labels, precisions, recalls):
            plt.annotate(label, (x, y))
        plt.xlabel("precision")
        plt.ylabel("recall")
        plt.title("precision and recall per label")
        self.writer.add_image(tag=f"labelwise_evaluation",
                              img_tensor=self._current_plot_to_image(),
                              dataformats="HWC",
                              global_step=self.global_step)
        self.data_loader.reset()
        return model


class PrintSaliency(TrainerCallback):
    """
    Really just used to test saliency stuff.
    """

    def __init__(self,
                 data_loaders_per_label: Dict[Label, KeyedBatchLoader],
                 text_kw: str = "text",
                 minibatch_size=256,
                 ):
        self.data_loaders = data_loaders_per_label
        self.text_kw = text_kw
        self.minibatch_size = minibatch_size

    def __call__(self, model: SaliencyModel, i_minibatch: int, iteration: int, *args, **kwargs) -> DeepModel:
        for label, data_loader in self.data_loaders.items():
            for keyed_batch in data_loader.yield_minibatches(minibatch_size=self.minibatch_size,
                                                             progress_bar=False):
                model.print_saliency(texts=keyed_batch[self.text_kw],
                                     label=label,
                                     **kwargs)
            data_loader.reset()
        return model


class SaveWrongPredictions(PlottingCallback):

    def __init__(self,
                 folder_path: str,
                 data_loader: KeyedBatchLoader,
                 text_kw: str = "clauses",
                 label_kw: str = "labels",
                 minibatch_size=256,
                 progress_bar: bool = True):
        super().__init__(folder_path)
        self.folder_path = folder_path
        self.data_loader = data_loader
        self.text_kw = text_kw
        self.label_kw = label_kw
        self.minibatch_size = minibatch_size
        self.progress_bar = progress_bar

    def __call__(self, model: Classifier, i_minibatch: int, iteration: int, *args, **kwargs) -> DeepModel:
        print(f"collecting wrong predictions...")
        texts = []
        y_pred = []
        y_true = []
        for keyed_batch in self.data_loader.yield_minibatches(minibatch_size=self.minibatch_size,
                                                              progress_bar=self.progress_bar):
            y_true.extend(keyed_batch[self.label_kw])
            y_pred.extend(model.predict_label(**keyed_batch, **kwargs))
            texts.extend(keyed_batch[self.text_kw])
        self.data_loader.reset()

        wrong_texts = []
        predictions = []
        for text, y_p, y_t in zip(texts, y_pred, y_true):
            if y_p != y_t:
                wrong_texts.append(text)
                predictions.append((y_t, y_p))

        grouped = group(X_data=wrong_texts, Y_data=predictions)

        for prediction, texts in grouped.items():
            merged_texts = "  \n".join(texts)
            self.writer.add_text(text_string=merged_texts,
                                 tag=f"test_data true {prediction[0]} prediction {prediction[1]}",
                                 global_step=self.global_step)
        return model

import abc
import importlib
import json
import os
from enum import Enum
from typing import Dict, Tuple, Iterator

import joblib
import keras
import numpy as np
import torch as t
from fastai.text import RNNLearner, load_learner

from unify_eval.model.types import Tensor
from unify_eval.utils.serialization import TarWriter, TarReader


class DeepModelBase(abc.ABC):
    """
    Absolute base class for all kinds of model objects.
    Can have optimizable parameters, but particular optimization is undefined yet
    (e.g. a single layer in a model can have optimizable parameters, but actual loss is defined for entire model only)
    """

    @abc.abstractmethod
    def get_components(self) -> dict:
        """
        get single components of models, e.g. preprocessing, actual model, ...
        :return: dictionary mapping from component name to actual name
        """
        pass

    @abc.abstractmethod
    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary mapping from parameter name to parameter value
        """

    def get_optimizable_parameters(self) -> Iterator[Tensor]:
        """
        Return an iterator over optimizable parameters.
        Should be overridden in case of pytorch or tensorflow models where parameters are passed to optimizer objects.
        For instance, a pytorch module would yield an iterator over torch.Tensor.
        Unless overridden, parameter values are numpy arrays.
        """
        for _, parameter in self.get_numpy_parameters():
            yield parameter

    def train_mode(self) -> "DeepModelBase":
        """
        Sets the model into training mode. Important when collecting statistics e.g. for batch norm, otherwise can be ignored
        :return: current model
        """
        return self

    def eval_mode(self) -> "DeepModelBase":
        """
        Sets the model into evaluation / inference mode. Important when collecting statistics e.g. for batch norm, otherwise can be ignored
        :return: current model
        """
        return self

    @classmethod
    @abc.abstractmethod
    def from_components(cls, **kwargs) -> "DeepModelBase":
        """
        factory to build a model from given components
        :param kwargs: named parts of the model
        :return: new model as combination of its parts
        """
        pass

    def save(self, path: str) -> "DeepModelBase":
        """
        save model into a single file
        :param path: path to single serialized model
        :return: current model
        """

        def save_components_and_meta(components: dict, meta_data: dict, path: str):
            with TarWriter(path):
                for name, component in components.items():
                    Backend.get_backend(component).save_component(name=name, component=component, folder_path=path)
                    with open(os.path.join(path, "metadata.json"), "w") as f:
                        json.dump(meta_data, f, indent=2)

        save_components_and_meta(
            components=self.get_components(),
            meta_data={
                "module": self.__class__.__module__,
                "cls": self.__class__.__name__
            },
            path=path)
        return self


class DeepModel(DeepModelBase):
    """
    High-Level abstract base class for all models to be evaluated.
    Models are supposed to be end2end, so they expect some raw (e.g. textual) data and preprocess it themselves.
    Can be trained via a given loss function.
    """

    @abc.abstractmethod
    def train(self, **kwargs) -> "DeepModel":
        """
        train model on data
        :param kwargs: training data
        :return: updated model
        """

    @abc.abstractmethod
    def get_loss(self, **kwargs) -> Dict[str, Tensor]:
        """
        get current training loss given data
        :param kwargs: labelled data
        :return: dictionary mapping from loss name to respective loss value.
        """
        pass


def load_model(path: str) -> DeepModel:
    """
    load model from a single file
    :param path: path to model
    :param cls: subclass of IntentModel that implements load_parts and from_components methods
    :return: deserialized model
    """

    def load_components_and_meta(path: str) -> Tuple[dict, dict]:
        components = dict()
        metadata = dict()
        with TarReader(path_to_model_file=path) as tar_reader:
            for file in os.listdir(tar_reader.path_extracted):
                file_abs = os.path.join(tar_reader.path_extracted, file)

                if os.path.isdir(file_abs):
                    backend = Backend[file]
                    for component_file in os.listdir(file_abs):
                        component_file_abs = os.path.join(file_abs, component_file)
                        name = component_file.split(".")[0]
                        components[name] = backend.load_component(path=component_file_abs)
                if file == "metadata.json":
                    with open(file_abs, "r") as f:
                        metadata = json.load(f)

        return components, metadata

    def load_cls(module, cls):
        return getattr(importlib.import_module(module), cls)

    components, meta = load_components_and_meta(path=path)
    cls = load_cls(module=meta["module"], cls=meta["cls"])
    return cls.from_components(**components)


class Backend(Enum):
    """
    Enum to represent supported backends.
    """
    fastai_rnn_learner = "fastai_rnn_learner"
    pytorch = "pytorch"
    keras = "keras"
    deep_model = "deep_model"
    other = "other"

    def _make_backend_folder(self, folder_path: str) -> str:
        """
        create a backend folder from folter_path and backend name and return that path
        """
        backend_folder = os.path.join(folder_path, self.value)
        if not os.path.exists(backend_folder):
            os.makedirs(backend_folder)
        return backend_folder

    def save_component(self, name: str, component, folder_path: str):
        """
        saves a given component by its name
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        backend_folder = self._make_backend_folder(folder_path)
        if self == Backend.fastai_rnn_learner:
            component: RNNLearner = component
            component.export(os.path.join(backend_folder, f"{name}"))
        if self == Backend.pytorch:
            t.save(component, os.path.join(backend_folder, f"{name}.pt"))
        if self == Backend.keras:
            component.save(os.path.join(backend_folder, f"{name}.h5"))
        if self == Backend.deep_model:
            component.save(path=os.path.join(backend_folder, name))
        if self == Backend.other:
            joblib.dump(component, os.path.join(backend_folder, f"{name}.pkl"), compress=True)

    def load_component(self, path: str) -> object:
        """
        load a single component stored under the given path
        """
        if self == Backend.fastai_rnn_learner:
            split_path = os.path.split(path)
            return load_learner(path=os.path.join(*split_path[:-1]), file=split_path[-1])
        if self == Backend.pytorch:
            return t.load(path)
        if self == Backend.keras:
            return keras.models.load_model(path)
        if self == Backend.deep_model:
            return load_model(path=path)
        if self == Backend.other:
            return joblib.load(path)

    @classmethod
    def get_backend(cls, obj) -> "Backend":
        """
        given some actual object, try to find the respective Backend enum
        """
        if isinstance(obj, RNNLearner):
            return Backend.fastai_rnn_learner
        if isinstance(obj, t.nn.Module):
            return Backend.pytorch
        if isinstance(obj, keras.Model):
            return Backend.keras
        if isinstance(obj, DeepModel):
            return Backend.deep_model
        return Backend.other

from typing import List

from unifyeval.model.deep_model import DeepModel
from unifyeval.model.mixins.classification import StatefulTextClassifier
from unifyeval.training.callback import TrainerCallback
from unifyeval.utils.load_data import KeyedBatchLoader
from unifyeval.model.mixins.stateful import StatefulModel


class Trainer:
    """
    Class that trains a model (duh).
    Builds a pipeline with callbacks called after every minibatch / full batch respectively.
    """

    def __init__(self,
                 data_loader: KeyedBatchLoader,
                 minibatch_callbacks: List[TrainerCallback],
                 batch_callbacks: List[TrainerCallback]):
        """
        :param data_loader: data loadering yielding training data
        :param minibatch_callbacks: List of TrainerCallback objects that are called after every minibatch update
        :param batch_callbacks: List of TrainerCallback objects that are called after every full iteration
        """
        self.data_loader = data_loader
        self.minibatch_callbacks = minibatch_callbacks
        self.batch_callbacks = batch_callbacks

    def train_on_minibatch(self, model: DeepModel, keyed_minibatch: dict, i_minibatch: int, iteration: int,
                           **kwargs) -> DeepModel:
        """
        trains a model on a minibatch and than calls minibatch callbacks
        :param model: model to train
        :param keyed_minibatch:  dictionary from data name to actual data
        :param kwargs: keyword arguments for downstream components (can be model-specific)
        :return: updated model
        """
        model.train_mode()
        model.train(**keyed_minibatch, **kwargs)
        model.eval_mode()
        for call_back in self.minibatch_callbacks:
            model = call_back(model=model,
                              iteration=iteration,
                              i_minibatch=i_minibatch,
                              **kwargs)
            call_back.global_step += 1
        return model

    def train_on_full_batch(self, model, minibatch_size: int, iteration: int, progress_bar: bool = True,
                            batch_callback_step: int = 1, **kwargs) -> DeepModel:
        """
        trains a model on full batch of data and than calls batch callbacks
        :param model: model to train
        :param minibatch_size: size of minibatches to use
        :param iteration: current iteration
        :param progress_bar: true if tqdm progress bar should be used for minibatch training. Defaults to false if data loader is lazy
        :param batch_callback_step: defines when batch callbacks are called (every n-th time)
        :param kwargs: keyword arguments for downstream components (can be model-specific)
        :return: updated model
        """

        print(f"training iteration {iteration}")
        i_minibatch = 0

        for i_minibatch, minibatch in enumerate(self.data_loader.yield_minibatches(minibatch_size=minibatch_size,
                                                                                   progress_bar=progress_bar)):
            model = self.train_on_minibatch(model=model,
                                            keyed_minibatch=minibatch,
                                            i_minibatch=i_minibatch,
                                            iteration=iteration,
                                            **kwargs)

        if batch_callback_step < 1:
            batch_callback_step = 1

        if iteration % batch_callback_step == 0:
            for call_back in self.batch_callbacks:
                model = call_back(model=model, iteration=iteration, i_minibatch=i_minibatch, **kwargs)
                call_back.global_step += batch_callback_step
        return model

    def train_model(self, model: DeepModel, n_iterations: int, minibatch_size: int, progress_bar: bool = True,
                    initial_iteration: int = 0,
                    run_name: str = "run0",
                    full_batch_callback_step: int = 1,
                    **kwargs) -> DeepModel:
        """
        trains a model on entire dataset for a given number of iterations and runs callbacks respectively
        :param model: model to train
        :param n_iterations: iterations over dataset
        :param minibatch_size: minibatch size to use
        :param progress_bar true if tqdm progress bar should be used for minibatch training. Defaults to false if data loader is lazy
        :param initial_iteration: initial value of iteration index, defaults to 0. Useful if pretrained model is trained further
        :param full_batch_callback_step: defines when batch callbacks are called (every n-th time)
        :param run_name: name of run
        :param kwargs: keyword arguments for downstream components (can be model-specific)
        :return: model trained on data for several iterations
        """

        print("preparing callbacks ...")
        for callbacks in (self.minibatch_callbacks, self.batch_callbacks):
            for callback in callbacks:
                callback.prepare_run(run_name=run_name, **kwargs)

        print("training ...")
        progress_bar = False if self.data_loader.is_lazy() else progress_bar

        for iteration in range(initial_iteration, n_iterations + initial_iteration):
            model = self.train_on_full_batch(model=model,
                                             minibatch_size=minibatch_size,
                                             iteration=iteration,
                                             progress_bar=progress_bar,
                                             batch_callback_step=full_batch_callback_step,
                                             **kwargs)
            self.data_loader.reset()
        return model


class TextClassifierTrainer:
    """
    Class that trains a model (duh).
    Builds a pipeline with callbacks called after every minibatch / full batch respectively.
    """

    def __init__(self,
                 data_loader: KeyedBatchLoader,
                 minibatch_callbacks: List[TrainerCallback],
                 batch_callbacks: List[TrainerCallback]):
        """
        :param data_loader: data loadering yielding train¥ing data
        :param minibatch_callbacks: List of TrainerCallback objects that are called after every minibatch update
        :param batch_callbacks: List of TrainerCallback objects that are called after every full iteration
        """
        self.data_loader = data_loader
        self.minibatch_callbacks = minibatch_callbacks
        self.batch_callbacks = batch_callbacks

    def train_on_minibatch(self, model: StatefulTextClassifier, keyed_minibatch: dict, i_minibatch: int, iteration: int,
                           **kwargs) -> StatefulTextClassifier:
        """
        trains a model on a minibatch and than calls minibatch callbacks
        :param model: model to train
        :param keyed_minibatch:  dictionary from data name to actual data
        :param kwargs: keyword arguments for downstream components (can be model-specific)
        :return: updated model
        """
        model.train_mode()
        model.reset()
        model.train(**keyed_minibatch, **kwargs)
        model.reset()
        model.eval_mode()
        for call_back in self.minibatch_callbacks:
            model: StatefulTextClassifier = call_back(model=model,
                                                      iteration=iteration,
                                                      i_minibatch=i_minibatch,
                                                      **kwargs)
            model.reset()
            call_back.global_step += 1
        return model

    def train_on_full_batch(self, model: StatefulTextClassifier, minibatch_size: int, iteration: int,
                            progress_bar: bool = True,
                            callback_step: int = 1,
                            **kwargs) -> StatefulTextClassifier:
        """
        trains a model on full batch of data and than calls batch callbacks
        :param model: model to train
        :param minibatch_size: size of minibatches to use
        :param iteration: current iteration
        :param progress_bar true if tqdm progress bar should be used for minibatch training. Defaults to false if data loader is lazy
                :param callback_step: defines when callbacks are called (every n-th time)
        :param kwargs: keyword arguments for downstream components (can be model-specific)
        :return: updated model
        """

        print(f"training iteration {iteration}")
        i_minibatch = 0


        for i_minibatch, minibatch in enumerate(self.data_loader.yield_minibatches(minibatch_size=minibatch_size,
                                                                                   progress_bar=progress_bar)):
            model = self.train_on_minibatch(model=model,
                                            keyed_minibatch=minibatch,
                                            i_minibatch=i_minibatch,
                                            iteration=iteration,
                                            **kwargs)
        if callback_step < 1:
            callback_step = 1
        model.eval_mode()
        model.reset()
        for call_back in self.batch_callbacks:
            model: StatefulTextClassifier = call_back(model=model, iteration=iteration, i_minibatch=i_minibatch,
                                                      **kwargs)
            model.reset()
            call_back.global_step += 1
        return model

    def train_model(self, model: StatefulTextClassifier, n_iterations: int, minibatch_size: int,
                    progress_bar: bool = True,
                    initial_iteration: int = 0,
                    **kwargs) -> StatefulTextClassifier:
        """
        trains a model on entire dataset for a given number of iterations and runs callbacks respectively
        :param model: model to train
        :param n_iterations: iterations over dataset
        :param minibatch_size: minibatch size to use
        :param progress_bar true if tqdm progress bar should be used for minibatch training. Defaults to false if data loader is lazy
        :param initial_iteration: initial value of iteration index, defaults to 0. Useful if pretrained model is trained further
        :param kwargs: keyword arguments for downstream components (can be model-specific)
        :return: model trained on data for several iterations
        """

        print("preparing callbacks ...")
        for callbacks in (self.minibatch_callbacks, self.batch_callbacks):
            for callback in callbacks:
                callback.prepare_run()

        print("training ...")
        progress_bar = False if self.data_loader.is_lazy() else progress_bar

        for iteration in range(initial_iteration, n_iterations + initial_iteration):
            model: StatefulTextClassifier = self.train_on_full_batch(model=model,
                                                                     minibatch_size=minibatch_size,
                                                                     iteration=iteration,
                                                                     progress_bar=progress_bar,
                                                                     **kwargs)
            self.data_loader.reset()

        return model.reset()

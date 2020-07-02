from random import shuffle
from typing import List, Union

from unifyeval.model.deep_model import DeepModel
from unifyeval.model.mixins.sequences.language_models import LayeredLanguageModel
from unifyeval.training.callback import TrainerCallback
from unifyeval.training.seq2seq.seq2seq_data import Seq2SeqData
from unifyeval.training.trainer import Trainer
from unifyeval.utils.load_data import KeyedBatchLoader, FiniteKeyedLazyDataLoader, KeyedLazyDataLoader


class Seq2SeqModelTrainer(Trainer):
    """
    Same as Trainer, but resets model state between calls
    """

    def __init__(self,
                 data_loader: Union[KeyedBatchLoader, FiniteKeyedLazyDataLoader, KeyedLazyDataLoader],
                 minibatch_callbacks: List[TrainerCallback],
                 batch_callbacks: List[TrainerCallback],
                 text_kw: str = "texts"):
        super().__init__(data_loader, minibatch_callbacks, batch_callbacks)
        self.text_kw = text_kw

    def train_on_minibatch(self, model: LayeredLanguageModel, keyed_minibatch: dict, i_minibatch: int, iteration: int,
                           backprop_length: int = 150, minibatch_size: int = 64,
                           **kwargs) -> DeepModel:

        tokenized_texts = model.layers.preprocessing.layers.tokenizer.tokenize_all(
            texts=list(keyed_minibatch.pop(self.text_kw)))

        # shuffle texts
        shuffle(tokenized_texts)

        lm_data = Seq2SeqData.generate_stateful_lm_data(
            sequence_mapper=model.layers.preprocessing.layers.sequence_mapper,
            tokenized_texts=tokenized_texts,
            minibatch_size=minibatch_size,
            backprop_length=backprop_length)
        model.train_mode()
        with model:
            model.train(data=lm_data, **keyed_minibatch, **kwargs)

        model.eval_mode()

        for call_back in self.minibatch_callbacks:
            model = call_back(model=model,
                              iteration=iteration,
                              i_minibatch=i_minibatch,
                              **keyed_minibatch, **kwargs,
                              minibatch_size=minibatch_size,
                              backprop_length=backprop_length)
            call_back.global_step += 1
            model.reset()
        return model

    def train_on_full_batch(self,
                            model: LayeredLanguageModel,
                            minibatch_size: int,
                            iteration: int,
                            progress_bar: bool = True,
                            backprop_length: int = 150,
                            text_batch_size: int = 10000,
                            **kwargs) -> DeepModel:
        print(f"training iteration {iteration}")
        i_minibatch = 0

        for i_minibatch, minibatch in enumerate(self.data_loader.yield_minibatches(minibatch_size=text_batch_size,
                                                                                   progress_bar=progress_bar)):
            self.train_on_minibatch(model=model,
                                    keyed_minibatch=minibatch,
                                    i_minibatch=i_minibatch,
                                    iteration=iteration,
                                    backprop_length=backprop_length,
                                    minibatch_size=minibatch_size,
                                    **kwargs)
        model.reset()
        for call_back in self.batch_callbacks:
            model = call_back(model=model,
                              iteration=iteration,
                              i_minibatch=i_minibatch,
                              minibatch_size=minibatch_size,
                              backprop_length=backprop_length,
                              **kwargs)
            call_back.global_step += 1
        return model

    def train_model(self, model: LayeredLanguageModel, n_iterations: int, minibatch_size: int, progress_bar: bool = True,
                    initial_iteration: int = 0, backprop_length: int = 150, text_batch_size: int = 10000,
                    **kwargs) -> DeepModel:

        print("training ...")
        for iteration in range(initial_iteration, n_iterations + initial_iteration):
            self.train_on_full_batch(model=model,
                                     minibatch_size=minibatch_size,
                                     iteration=iteration,
                                     progress_bar=progress_bar,
                                     backprop_length=backprop_length,
                                     **kwargs)
            self.data_loader.reset()
        return model


class CompSeq2SeqModelTrainer(Trainer):
    """
    Same as Trainer, but resets model state between calls
    """

    def __init__(self,
                 data_loader: Union[KeyedBatchLoader, FiniteKeyedLazyDataLoader, KeyedLazyDataLoader],
                 minibatch_callbacks: List[TrainerCallback],
                 batch_callbacks: List[TrainerCallback],
                 text_kw: str = "texts"):
        super().__init__(data_loader, minibatch_callbacks, batch_callbacks)
        self.text_kw = text_kw

    def train_on_minibatch(self, model: LayeredLanguageModel, keyed_minibatch: dict, i_minibatch: int, iteration: int,
                           backprop_length: int = 150, minibatch_size: int = 64,
                           **kwargs) -> LayeredLanguageModel:

        tokenized_texts = model.tokenizer.tokenize_all(texts=list(keyed_minibatch.pop(self.text_kw)))

        # shuffle texts
        shuffle(tokenized_texts)

        lm_data = Seq2SeqData.generate_stateful_lm_data(sequence_mapper=model.sequence_mapper,
                                                        tokenized_texts=tokenized_texts,
                                                        minibatch_size=minibatch_size,
                                                        backprop_length=backprop_length)
        model.train_mode()
        with model:
            model.train(data=lm_data, **keyed_minibatch)

        model.eval_mode()

        for call_back in self.minibatch_callbacks:
            model = call_back(model=model,
                              iteration=iteration,
                              i_minibatch=i_minibatch,
                              **keyed_minibatch, **kwargs,
                              minibatch_size=minibatch_size,
                              backprop_length=backprop_length)
            call_back.global_step += 1
            model.reset()
        return model

    def train_on_full_batch(self,
                            model: LayeredLanguageModel,
                            minibatch_size: int,
                            iteration: int,
                            progress_bar: bool = True,
                            backprop_length: int = 150,
                            text_batch_size: int = 10000,
                            **kwargs) -> LayeredLanguageModel:
        print(f"training iteration {iteration}")
        i_minibatch = 0

        for i_minibatch, minibatch in enumerate(self.data_loader.yield_minibatches(minibatch_size=text_batch_size,
                                                                                   progress_bar=progress_bar)):
            self.train_on_minibatch(model=model,
                                    keyed_minibatch=minibatch,
                                    i_minibatch=i_minibatch,
                                    iteration=iteration,
                                    backprop_length=backprop_length,
                                    minibatch_size=minibatch_size,
                                    **kwargs)
        model.reset()
        for call_back in self.batch_callbacks:
            model = call_back(model=model,
                              iteration=iteration,
                              i_minibatch=i_minibatch,
                              minibatch_size=minibatch_size,
                              backprop_length=backprop_length,
                              **kwargs)
            call_back.global_step += 1
        return model

    def train_model(self,
                    model: LayeredLanguageModel,
                    n_iterations: int,
                    minibatch_size: int,
                    progress_bar: bool = True,
                    initial_iteration: int = 0,
                    backprop_length: int = 150,
                    text_batch_size: int = 10000,
                    **kwargs) -> LayeredLanguageModel:

        print("training ...")
        for iteration in range(initial_iteration, n_iterations + initial_iteration):
            self.train_on_full_batch(model=model,
                                     minibatch_size=minibatch_size,
                                     iteration=iteration,
                                     progress_bar=progress_bar,
                                     backprop_length=backprop_length,
                                     **kwargs)
            self.data_loader.reset()
        return model

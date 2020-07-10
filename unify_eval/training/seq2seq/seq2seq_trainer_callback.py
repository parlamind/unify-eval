import gc
import os
from typing import Callable

import numpy as np
from sklearn.model_selection import train_test_split

from unify_eval.model.deep_model import load_model
from unify_eval.model.mixins.stateful_classification import StatefulTextClassifier
from unify_eval.model.mixins.sequences.language_models import LayeredLanguageModel
from unify_eval.training.callback import CheckNaN, EvaluationCallBack, TensorboardCallback
from unify_eval.training.seq2seq.seq2seq_data import Seq2SeqData
from unify_eval.training.trainer import Trainer
from unify_eval.utils.corpus import Corpus
from unify_eval.utils.load_data import FiniteKeyedLazyDataLoader, KeyedBatchLoader, KeyedSubsampledBatchLoader


class LanguageModelEvaluation(TensorboardCallback):

    def __init__(self,
                 folder_path: str,
                 relative_path: str,
                 data_loader: FiniteKeyedLazyDataLoader,
                 text_minibatch_size: int = 10000):
        super().__init__(folder_path, relative_path)
        self.data_loader = data_loader
        self.text_minibatch_size = text_minibatch_size

    def __call__(self, model: LayeredLanguageModel, i_minibatch: int, iteration: int, *args,
                 **kwargs) -> LayeredLanguageModel:
        print(f"evaluating language model with {self.data_loader.n_datapoints} datapoints")
        cross_entropies = []
        perplexities = []
        for i_texts, minibatch in enumerate(
                self.data_loader.yield_minibatches(minibatch_size=self.text_minibatch_size)):
            texts = minibatch[kwargs["text_kw"]]
            # max_length = np.max([len(text) for text in texts])

            data = Seq2SeqData.generate_stateful_lm_data(
                sequence_mapper=model.sequence_mapper,
                tokenized_texts=model.tokenizer.tokenize_all(texts),
                minibatch_size=kwargs["minibatch_size"],
                backprop_length=kwargs["backprop_length"])

            cross_entropy = model.get_cross_entropy(data=data,
                                                    **kwargs)
            cross_entropies.append(cross_entropy)
            # calculate perplexity here since we already have cross entropy
            perplexity = np.exp(cross_entropy)
            perplexities.append(perplexity)
        self.data_loader.reset()
        cross_entropy = np.mean(cross_entropies)
        perplexity = np.mean(perplexities)
        self.writer.add_scalar(tag="lm_cross_entropy", scalar_value=cross_entropy, global_step=self.global_step)
        self.writer.add_scalar(tag="lm_perplexity", scalar_value=perplexity, global_step=self.global_step)
        return model


class GenerateText(TensorboardCallback):
    def __init__(self,
                 folder_path: str,
                 relative_path: str,
                 seed_string: str,
                 n_tokens: int,
                 temperature: float = 1.0,
                 delimiter: str = " ",
                 add_newlines: str = "\n"):
        super().__init__(folder_path, relative_path)
        self.seed_string = seed_string
        self.n_tokens = n_tokens
        self.temperature = temperature
        self.delimiter = delimiter
        self.add_newlines = add_newlines

    def __call__(self, model: LayeredLanguageModel, i_minibatch: int, iteration: int, *args,
                 **kwargs) -> LayeredLanguageModel:
        generated = model.generate(text=self.seed_string,
                                   n_tokens=self.n_tokens,
                                   temperature=self.temperature,
                                   delimiter=self.delimiter)
        if self.add_newlines is not None:
            generated = generated.replace(f"{self.delimiter}xxbos", "  \nxxbos")
        self.writer.add_text(tag=f"generated_text temperature {self.temperature} {self.seed_string}",
                             text_string=generated,
                             global_step=self.global_step)
        return model


class TextClassifierEvaluation(TensorboardCallback):
    def __init__(self,
                 folder_path: str,
                 relative_path: str,
                 corpus: Corpus,
                 build_classifier: Callable[[LayeredLanguageModel], StatefulTextClassifier],
                 n_iterations: int):
        super().__init__(folder_path, relative_path)
        self.corpus = corpus

        X_train, X_test, Y_train, Y_test = train_test_split(corpus.X,
                                                            corpus.Y,
                                                            test_size=0.1,
                                                            random_state=42,
                                                            stratify=corpus.Y)
        self._X_train = X_train
        self._X_test = X_test
        self._Y_train = Y_train
        self._Y_test = Y_test
        self.folder_path = folder_path
        self.training_data_loader = KeyedBatchLoader(raw_texts=X_train, labels=Y_train)
        self.test_data_loader = KeyedBatchLoader(raw_texts=X_test, labels=Y_test)
        self.test_data_loader_85 = KeyedBatchLoader(raw_texts=X_test, labels=Y_test)
        self.build_classifier = build_classifier
        self.n_iterations = n_iterations

    def __call__(self, model: LayeredLanguageModel, i_minibatch: int, iteration: int, *args,
                 **kwargs) -> LayeredLanguageModel:
        trainer = Trainer(data_loader=KeyedBatchLoader(raw_texts=self._X_train, labels=self._Y_train),
                          minibatch_callbacks=[CheckNaN()],
                          batch_callbacks=[
                              # run evaluation on subsampled training data (for bigger training sets that might take ages)
                              EvaluationCallBack.default(
                                  folder_path=self.folder_path,
                                  relative_path=os.path.join("train", f"{self.global_step}"),
                                  data_loader=KeyedSubsampledBatchLoader(n_subsampled=1024,
                                                                         raw_texts=self._X_train,
                                                                         labels=self._Y_train),
                                  label_indices=self.corpus.label_mapper.all_indices),
                              # run evaluation on test data
                              EvaluationCallBack.default(
                                  folder_path=self.folder_path,
                                  relative_path=os.path.join("test", f"{self.global_step}"),
                                  data_loader=KeyedBatchLoader(raw_texts=self._X_test,
                                                               labels=self._Y_test),
                                  label_indices=self.corpus.label_mapper.all_indices,
                                  minibatch_size=512),
                              # run evaluation on test data, but now with a junk threshold
                              EvaluationCallBack.default(
                                  folder_path=self.folder_path,
                                  relative_path=os.path.join("test_85", f"{self.global_step}"),
                                  data_loader=KeyedBatchLoader(raw_texts=self._X_test,
                                                               labels=self._Y_test),
                                  label_indices=self.corpus.label_mapper.all_indices,
                                  junk_threshold=0.85,
                                  junk_label_index=self.corpus.label_mapper.actuallabel2index[
                                      -1],
                                  minibatch_size=512), ])

        classifier: StatefulTextClassifier = self.build_classifier(model)

        path_lm = "/tmp/text_classifier_evaluation/tmp_lm.model"
        path_clf = "/tmp/text_classifier_evaluation/tmp_clf.model"
        print(f"saving language model for text classifier evaluation under {path_lm}")
        model.save(path_lm)
        # make model space is released
        del model
        gc.collect()
        print(f"saving classifier for text classifier evaluation under {path_lm}")
        classifier.save(path_clf)
        print(f"loading model for text classifier evaluation from {path_clf}")
        classifier: StatefulTextClassifier = load_model(path_clf)
        trainer.train_model(model=classifier, n_iterations=self.n_iterations, **kwargs)
        print(f"loading language model again from {path_lm}")
        model: LayeredLanguageModel = load_model(path_lm)
        return model

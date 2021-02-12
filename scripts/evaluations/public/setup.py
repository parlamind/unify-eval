from sklearn.model_selection import train_test_split
import torch as t

from unify_eval.training.callback import *
from unify_eval.training.isolated_evaluation import IsolatedEvaluation, load_messages
from unify_eval.training.trainer import Trainer
from unify_eval.utils.load_data import KeyedBatchLoader
from scripts.evaluations.public import setup_data, setup_model


class DefaultSetup:
    """
    A class to represent an experimental setup, where the user specifies the dataset and model,
    along with all the desired hyperparamter settings, in order to then get a trainer object
    that can be used to conduct training and evaluation.
    To experiment with a new model, the user needs to create a new DefaultSetup object.
    """

    def __init__(self, user_name, data, model_name="model",
                 technique="distilbert", architecture="lstm", num_layers=1,
                 activation=t.nn.ReLU, lr=0.001, weight_decay=0.01, dropout=0.1, skip_connect=False, using_gpu=True):
        """
        :param user_name: the dir with the user name on the
        :param data: one of {"snips", "AskUbuntuCorpus", "ChatbotCorpus", "WebApplicationsCorpus"}
        :param model_name:
        :param technique: one of {"bert", "distilbert", "roberta", "gpt2"}
        :param architecture: one of {"mlp", "attention", "lstm", "gru", "rnn"}
        :param num_layers:
        :param activation:
        :param lr:
        :param weight_decay:
        :param dropout:
        :param skip_connect: if true, the word embedding is fed as input to the final (output) layer of the classifier
        :param using_gpu:
        """
        self.data = data
        self.eval_folder = os.path.join(f"/home/{user_name}/results", "deep_models")
        self.model_name = model_name
        self.model_folder = os.path.join(self.eval_folder, self.model_name)
        self.train_corpus = setup_data.get_data(data, test_data=False)
        self.test_corpus = setup_data.get_data(data, test_data=True)
        self.model = setup_model.get_model(self.train_corpus, technique, "en", architecture, num_layers, activation, lr,
                                           weight_decay, dropout, skip_connect, using_gpu)

    def get_corpus(self, test=False):
        # get the desired set (train / test) of the indicated corpus (e.g. SNIPS)
        return self.train_corpus if not test else self.test_corpus

    def get_model(self):
        return self.model

    def get_trainer(self, include_isolated=False, include_plotting=False, extra_callbacks=None):
        # this is the trainer object that can be  used to conduct the training and evaluation
        trainer = Trainer(
            data_loader=self.get_data_loader(),
            minibatch_callbacks=self.get_minibatch_callbacks(),
            batch_callbacks=self.get_batch_callbacks(include_isolated=include_isolated,
                                                     include_plotting=include_plotting,
                                                     extra_callbacks=extra_callbacks)
        )
        return trainer

    def get_data_loader(self, subsample=False, testing=False):
        """
        :param subsample: if true, the loader will contain only 512 instances instead of the whole dataset
        :param testing: if true, the loader will fetch the test set instances, otherwise the training set instances
        :return: an object to load the data inside the callbacks
        """
        clauses = self.test_corpus.X if testing else self.train_corpus.X
        labels = self.test_corpus.Y if testing else self.train_corpus.Y

        if subsample:
            return KeyedSubsampledBatchLoader(n_subsampled=512, clauses=clauses, labels=labels)
        else:
            return KeyedBatchLoader(clauses=clauses, labels=labels)

    def get_isolated_data_loader(self):
        """
        :return: get a data loader object for the isolated evaluation
        """
        kbd = KeyedBatchLoader(**load_messages(
            path=os.path.join("data", "public", "snips", "test_isolated.json"),
            text_kw="clauses"))
        return kbd

    def get_minibatch_callbacks(self):
        # checked after each mini-batch of data
        # check that no parameter is NaN
        minibatch_callbacks = [CheckNaN()]
        return minibatch_callbacks

    def get_batch_callbacks(self, include_isolated=False, include_plotting=False, extra_callbacks=None):
        """
        checked after each iteration over the whole batch of data
        :param include_isolated: if true, includes a callback to conduct isolated evaluation (F1, prec and recall)
        :param include_plotting: if true, includes a callback to visualize the results (using tensorboard)
        :param extra_callbacks: provide a list of any additional callbacks and it will be included in the final list
        :return: a list of callbacks
        """
        batch_callbacks = self.get_standard_eval_callbacks()
        if include_isolated:
            batch_callbacks.extend(self.get_isolated_eval_callbacks())
        if include_plotting:
            batch_callbacks.extend(self.get_plotting_callbacks())
        if extra_callbacks is not None:
            batch_callbacks.extend(extra_callbacks)
        batch_callbacks.extend(self.get_model_saver_callback())
        return batch_callbacks

    def get_standard_eval_callbacks(self):
        """
        :return: callbacks to measure model accuracy on train and test set
        note: for the SNIPS dataset, we use only a sample of the data when measuring the score
              on the training set in order to reduce the time needed for evaluation
        """
        return [
            EvaluationCallBack.default(folder_path=os.path.join(self.model_folder, "evaluation_data", "train"),
                                       data_loader=self.get_data_loader(subsample=(self.data == "snips")),
                                       label_indices=self.train_corpus.label_mapper.all_indices,
                                       minibatch_size=32),
            EvaluationCallBack.default(folder_path=os.path.join(self.model_folder, "evaluation_data", "test"),
                                       data_loader=self.get_data_loader(testing=True),
                                       label_indices=self.train_corpus.label_mapper.all_indices,
                                       minibatch_size=32)
        ]

    def get_isolated_eval_callbacks(self):
        return [
            IsolatedEvaluation(folder_path=os.path.join(self.model_folder, "isolated_evaluation"),
                               data_loader=self.get_isolated_data_loader(),
                               labels_to_evaluate=self.train_corpus.label_mapper.labels),
            IsolatedEvaluation(folder_path=os.path.join(self.model_folder, "isolated_evaluation_85"),
                               data_loader=self.get_isolated_data_loader(),
                               labels_to_evaluate=self.train_corpus.label_mapper.labels,
                               junk_threshold=0.85)
        ]

    def get_plotting_callbacks(self):
        return [
            # save parameter values for tensorboard
            PlotParameters(folder_path=os.path.join(self.model_folder, "parameters")),
            # save embeddings for tensorboard
            PlotClassificationEmbeddings(
                folder_path=os.path.join(self.model_folder, "embeddings"),
                data_loader=self.get_data_loader(testing=True),
                label_kw="labels"
            ),
            # plot labelwise precision and recall
            LabelSpecificEvaluation(folder_path=os.path.join(self.model_folder, "label_scatter_plot"),
                                    label_kw="labels",
                                    data_loader=self.get_data_loader(testing=True)),
        ]

    def get_model_saver_callback(self):
        """
        At any given point, only a maximum of N best performing models are stored.
        Currently we use N = 5
        :return:
        """
        return [
            ModelSaverCallback(
                model_saver=QueuedModelSaver(
                    path_to_folder=os.path.join(self.model_folder, "saved_models"),
                    model_name=self.model_name,
                    queue_size=5))
        ]

    def initialize_training(self, n_iterations=30, mini_batch_size=32, plot=False):
        # kick off the training and evaluation process
        self.get_trainer(include_isolated=True, include_plotting=plot)\
            .train_model(model=self.get_model(),
                         classes=self.get_corpus().label_mapper.all_indices,
                         n_iterations=n_iterations,
                         minibatch_size=mini_batch_size,
                         epochs=1,
                         full_batch_callback_step=1,
                         verbose=0)

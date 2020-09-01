from scripts.evaluations.public.setup import DefaultSetup
import torch as t


"""
We evaluate 2 different models on the AskUbuntu corpus (one of the Sebischair corpora) to see which one performs better. 
For each model, we fo through 2 steps: 
1. select the model we want to use and specify its hyperparams
2. start the training
"""

# the first model we try is DistilBert
# user_name refers to the user on the operating sys, assuming that a folder called /home/<user_name> exists
first_setup = DefaultSetup(user_name="mohamed", data="AskUbuntuCorpus", model_name="ubuntu_distilbert",
                           technique="distilbert", architecture="lstm", num_layers=1,
                           activation=t.nn.ELU, lr=0.001, weight_decay=0.0001, dropout=0.1, skip_connect=False,
                           using_gpu=False)

# start the training of the first model
first_setup.initialize_training(n_iterations=40, mini_batch_size=32, plot=False)

# the second model we try is Bert
# as it is more complex, we use a smaller learning rate, but otherwise keep the hyperparams as before
second_setup = DefaultSetup(user_name="mohamed", data="AskUbuntuCorpus", model_name="ubuntu_bert",
                            technique="bert", architecture="lstm", num_layers=1,
                            activation=t.nn.ELU, lr=0.0007, weight_decay=0.0001, dropout=0.1, skip_connect=False,
                            using_gpu=False)

# start the training of the second model
second_setup.initialize_training(n_iterations=40, mini_batch_size=32, plot=False)

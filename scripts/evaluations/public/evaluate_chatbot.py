from scripts.evaluations.public.setup import DefaultSetup
import torch as t


"""
We evaluate DistilBert on the Chatbot corpus (one of the Sebischair corpora).
To evaluate a model, we fo through 2 steps: 
1. select the model we want to use and specify its hyperparams
2. start the training
Any other model can be evaluated analogously
"""

setup = DefaultSetup(user_name="mohamed", data="ChatbotCorpus", model_name="chatbot_distilbert",
                     technique="distilbert", architecture="lstm", num_layers=1,
                     activation=t.nn.ELU, lr=0.001, weight_decay=0.0001, dropout=0.1, skip_connect=False,
                     using_gpu=False)

# start the training
setup.initialize_training(n_iterations=40, mini_batch_size=32, plot=False)
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel, \
    GPT2Model, GPT2Tokenizer, RobertaModel, RobertaTokenizer
from sentence_transformers import SentenceTransformer
from unify_eval.model.transformer_clf import MLP, SelfAttentionClassifier
from unify_eval.model.rnn_clf import RecurrentClassifier
from unify_eval.model.transformer_model import *
from unify_eval.model.sbert import SbertClassifier, SbertClassificationModel


def get_model(corpus, technique, lang, architecture, num_layers, activation, lr, weight_decay, dropout, skip_connect, using_gpu):
    """
    :param corpus: must have a label_mapper attribute
    :param technique: one of {"bert", "distilbert", "roberta", "gpt2"}
    :param lang: de or en
    :param architecture: one of {"mlp", "attention", "lstm", "gru", "rnn"}
    :param num_layers: num encoder units (attention), recurrent layers (lstm, gru, rnn) or feedforward layers (mlp)
    :param activation: used only in the (final) mlp component of the classifier
    :param lr: learning rate
    :param weight_decay: lambda for regularization in AdamW optimizer
    :param dropout: the same dropout is used across all layers of the classifier
    :param skip_connect: if true, the original input to the classifier gets added to the output of the penultimate layer
    :param using_gpu: boolean
    :return:
    """
    num_in = 768
    num_out = corpus.label_mapper.n_labels
    device = "cuda:0" if using_gpu else "cpu"

    if architecture == "mlp":
        layer_sizes = [num_in] * num_layers
        layer_sizes.extend([num_out])
        clf = MLP(layer_sizes=layer_sizes, activation=activation, dropout=dropout).to(device)
    elif architecture == "attention":
        clf = SelfAttentionClassifier(dim_model=num_in, n_output_units=num_out, n_encoder_units=num_layers,
                                      activation=activation, dropout=dropout, skip_connect=skip_connect).to(device)
    else:  # recurrence
        clf = RecurrentClassifier(input_dim=num_in, output_dim=num_out, recurrence_type=architecture,
                                  num_recurrent_layers=num_layers, activation=activation, dropout=dropout,
                                  skip_connect=skip_connect).to(device)

    if technique == "bert":
        model = get_bert(corpus, lang, architecture, clf, device, lr, weight_decay)
    elif technique == "distilbert":
        model = get_distilbert(corpus, lang, architecture, clf, device, lr, weight_decay)
    elif technique == "gpt2":
        model = get_gpt2(corpus, architecture, clf, device, lr, weight_decay)
    elif technique == "roberta":
        model = get_roberta(corpus, architecture, clf, device, lr, weight_decay)
    elif technique == "sbert":
        model = get_sbert(corpus, lang, clf, device, lr, weight_decay)
    else:
        model = None
    return model


def get_sbert(corpus, lang, clf, device, lr, weight_decay):
    pretrained_model_name = "paraphrase-xlm-r-multilingual-v1" if lang == "de" else "paraphrase-distilroberta-base-v1"
    model = SbertClassificationModel(label_mapper=corpus.label_mapper,
                                     sbert_classifier=SbertClassifier(
                                        pretrained_model_name=pretrained_model_name,
                                        clf=clf),
                                     lr=lr,
                                     weight_decay=weight_decay).to_device(device)
    return model


def get_bert(corpus, lang, architecture, clf, device, lr, weight_decay):
    # the pre-trained model name is hardcoded here
    # to select a different one: refer to https://huggingface.co/transformers/pretrained_models.html
    pretrained_model_name = "bert-base-german-dbmdz-uncased" if lang == "de" else "bert-base-uncased"
    model = BertClassificationModel(label_mapper=corpus.label_mapper,
                                    transformer_classifier=TransformerClassifier(
                                        encoder=BertModel.from_pretrained(pretrained_model_name),
                                        clf=clf, clf_architecture=architecture),
                                    tokenizer=BertTokenizer.from_pretrained(pretrained_model_name),
                                    lr=lr,
                                    weight_decay=weight_decay).to_device(device)
    return model


def get_distilbert(corpus, lang, architecture, clf, device, lr, weight_decay):
    # the pre-trained model name is hardcoded here
    # to select a different one: refer to https://huggingface.co/transformers/pretrained_models.html
    pretrained_model_name = "distilbert-base-german-cased" if lang == "de" else "distilbert-base-uncased"
    model = BertClassificationModel(label_mapper=corpus.label_mapper,
                                    transformer_classifier=TransformerClassifier(
                                        encoder=DistilBertModel.from_pretrained(pretrained_model_name),
                                        clf=clf, clf_architecture=architecture),
                                    tokenizer=DistilBertTokenizer.from_pretrained(pretrained_model_name),
                                    distilling=True,
                                    lr=lr,
                                    weight_decay=weight_decay).to_device(device)
    return model


def get_gpt2(corpus, architecture, clf, device, lr, weight_decay):
    # the pre-trained model name is hardcoded here
    # to select a different one: refer to https://huggingface.co/transformers/pretrained_models.html
    pretrained_model_name = "gpt2"
    model = TransformerClassificationModel(label_mapper=corpus.label_mapper,
                                           transformer_classifier=TransformerClassifier(
                                               encoder=GPT2Model.from_pretrained(pretrained_model_name),
                                               clf=clf, clf_architecture=architecture),
                                           tokenizer=GPT2Tokenizer.from_pretrained(pretrained_model_name),
                                           lr=lr,
                                           weight_decay=weight_decay).to_device(device)
    return model


def get_roberta(corpus, architecture, clf, device, lr, weight_decay):
    # the pre-trained model name is hardcoded here
    # to select a different one: refer to https://huggingface.co/transformers/pretrained_models.html
    pretrained_model_name = "roberta-base"
    model = RobertaClassificationModel(label_mapper=corpus.label_mapper,
                                       transformer_classifier=TransformerClassifier(
                                           encoder=RobertaModel.from_pretrained(pretrained_model_name),
                                           clf=clf, clf_architecture=architecture),
                                       tokenizer=RobertaTokenizer.from_pretrained(pretrained_model_name),
                                       lr=lr,
                                       weight_decay=weight_decay).to_device(device)
    return model

from typing import List, Callable, Dict

import numpy as np
import torch as t
import torch.nn as nn
from torch.nn import Embedding

from unifyeval.model.mixins.adversarial import PytorchSaliencyModel
from unifyeval.model.mixins.embedding import EmbeddingModel
from unifyeval.model.types import Tensor, ListOfRawTexts, ListOfTokenizedTexts
from unifyeval.utils.label_mapper import LabelMapper
from unifyeval.utils.text_sequence import SequenceMapper, Tokenizer
from unifyeval.model.mixins.classification import Classifier

"""
EXAMPLE FOR SOME DEEP MODEL 
REALLY JUST USED FOR PROTOTYPING THE WHOLE API

"""


def sliding(l, slide_size: int, step_size=1):
    for i in range(0, len(l) - slide_size + step_size, step_size):
        result = l[i:i + slide_size]
        if len(result) == slide_size:
            yield result


class MLP(t.nn.Module):

    def __init__(self, layer_sizes: List[int], activation: Callable = t.nn.ELU):
        super(MLP, self).__init__()
        self.activation = activation()
        self.layer_sizes = layer_sizes
        self.layers = t.nn.ModuleList([t.nn.Linear(in_features=n_in, out_features=n_out)
                                       for (n_in, n_out) in sliding(layer_sizes, slide_size=2, step_size=1)])
        self.dropouts = t.nn.ModuleList([t.nn.Dropout(p=0.1) for l in layer_sizes[:-1]])

    def forward(self, x):
        for layer, dropout in zip(self.layers[:-1], self.dropouts):
            x = self.activation(layer(dropout(x)))
        return self.layers[-1](x)


class RawSingleAttention(nn.Module):

    def get_attention(self,
                      queries: t.Tensor,
                      keys: t.Tensor,
                      attention_mask: t.Tensor) -> t.Tensor:
        d_keys = t.tensor(keys.shape[-1], requires_grad=False).float()
        attention_logits = queries.matmul(keys.permute((0, 2, 1))) / t.sqrt(d_keys)
        attention_logits_masked = t.where(attention_mask < -1.0, attention_mask, attention_logits)
        if t.isnan(attention_logits_masked).any():
            pass
        attention = t.softmax(attention_logits_masked, dim=-1)
        # if key is padding, corresponding attention will be nan, so turn those into zeros
        final_attention = t.where(t.isnan(attention), t.tensor(0.0), attention)
        return final_attention

    def forward(self,
                queries: t.Tensor,
                keys: t.Tensor,
                values: t.Tensor,
                attention_mask: t.Tensor) -> t.Tensor:
        """
        single batch-wise attention
        :param queries: [bach_size,n_queries,query_dim]
        :param keys: [bach_size,n_keys,query_dim]
        :param values: [bach_size,n_keys,value_dim]
        :return: attended values [batch_size,n_queries,value_dim]
        """

        return self.get_attention(queries=queries, keys=keys, attention_mask=attention_mask).matmul(values)


class ProjectedSingleAttention(nn.Module):
    def __init__(self, dim_model: int, dim_queries: int, dim_keys: int, dim_values: int):
        super().__init__()
        self.dim_model = dim_model
        self.dim_queries = dim_queries
        self.dim_keys = dim_keys
        self.dim_values = dim_values
        self.raw_single_attention = RawSingleAttention()

        self.W_queries = nn.Linear(in_features=dim_model, out_features=dim_queries, bias=False)
        self.W_keys = nn.Linear(in_features=dim_model, out_features=dim_keys, bias=False)
        self.W_values = nn.Linear(in_features=dim_model, out_features=dim_values, bias=False)

    def forward(self, queries: t.Tensor,
                keys: t.Tensor,
                values: t.Tensor,
                attention_mask: t.Tensor) -> t.Tensor:
        """
       single batch-wise attention
       :param queries: [bach_size,n_queries,dim_model]
       :param keys: [bach_size,n_keys,dim_model]
       :param values: [bach_size,n_keys,dim_model]
       :return: attended values [batch_size,n_queries,model_dim]
       """
        return self.raw_single_attention.forward(
            queries=self.W_queries.forward(queries),
            keys=self.W_keys.forward(keys),
            values=self.W_values.forward(values),
            attention_mask=attention_mask)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, dim_model: int, dim_queries: int, dim_keys: int, dim_values: int):
        super().__init__()
        self.heads: nn.ModuleList = \
            nn.ModuleList([ProjectedSingleAttention(dim_model=dim_model,
                                                    dim_queries=dim_queries,
                                                    dim_keys=dim_keys,
                                                    dim_values=dim_values) for _ in range(n_heads)])
        self.W_out = nn.Linear(in_features=n_heads * dim_values, out_features=dim_model, bias=False)

    def forward(self,
                queries: t.Tensor,
                keys: t.Tensor,
                values: t.Tensor,
                attention_mask: t.Tensor) -> t.Tensor:
        """
                batch-wise multi-head attention
               :param queries: [bach_size,n_queries,dim_model]
               :param keys: [bach_size,n_keys,dim_model]
               :param values: [bach_size,n_keys,dim_model]
               :return: attended values [batch_size,n_queries,model_dim]
               """
        return self.W_out.forward(t.cat([head.forward(queries=queries,
                                                      keys=keys,
                                                      values=values,
                                                      attention_mask=attention_mask)
                                         for head in self.heads], dim=-1))


class TransformerSublayer(nn.Module):
    def __init__(self,
                 n_heads: int,
                 dim_model: int,
                 dim_queries: int,
                 dim_keys: int,
                 dim_values: int,
                 ff_hidden_dims: List[int]):
        super().__init__()
        self.dim_model = dim_model
        self.multihead_attention: MultiHeadAttention = MultiHeadAttention(n_heads=n_heads,
                                                                          dim_model=dim_model,
                                                                          dim_queries=dim_queries,
                                                                          dim_keys=dim_keys,
                                                                          dim_values=dim_values)
        self.ff = MLP(layer_sizes=[self.dim_model] + ff_hidden_dims + [self.dim_model])
        self.dropout = t.nn.Dropout(p=0.1)
        self.lnorm0 = nn.LayerNorm(self.dim_model)
        self.lnorm1 = nn.LayerNorm(self.dim_model)

    def forward(self, queries: t.Tensor,
                keys: t.Tensor,
                values: t.Tensor,
                attention_mask: t.Tensor) -> t.Tensor:
        intermediate = self.lnorm0(
            queries + self.multihead_attention.forward(queries=queries,
                                                       keys=keys,
                                                       values=values,
                                                       attention_mask=attention_mask))
        output = self.ff.forward(intermediate)
        dropout_output = self.dropout(output)
        return self.lnorm1(dropout_output + intermediate)


class SelfAttentionLayer(nn.Module):
    def __init__(self,
                 n_encoder_units: int,
                 n_heads: int,
                 dim_model: int,
                 dim_queries: int,
                 dim_keys: int,
                 dim_values: int,
                 ff_hidden_dims: List[int]):
        super().__init__()
        self.sublayers = nn.ModuleList([TransformerSublayer(n_heads=n_heads,
                                                            dim_model=dim_model,
                                                            dim_queries=dim_queries,
                                                            dim_keys=dim_keys,
                                                            dim_values=dim_values,
                                                            ff_hidden_dims=ff_hidden_dims) for _ in
                                        range(n_encoder_units)])

    def forward(self, queries: t.Tensor, attention_mask: t.Tensor) -> t.Tensor:
        queries, keys, values = queries, queries, queries
        for i_layer, layer in enumerate(self.sublayers):
            out = layer.forward(queries=queries,
                                keys=keys,
                                values=values,
                                attention_mask=attention_mask)
            queries, keys, values = out, out, out
        return queries


class SelfAttentionClassifier(nn.Module):
    def __init__(self,
                 dim_model: int,
                 n_output_units: int,
                 n_encoder_units: int = 1,
                 n_heads: int = 12,
                 transformer_ff_hidden_dims: List[int] = None,
                 activation: Callable = t.nn.ReLU) -> None:
        super().__init__()
        if n_encoder_units < 1:
            n_encoder_units = 1
        # common heuristics to set the dimensionality of queries and feed-forward layers
        dim_queries = int(dim_model / n_heads)
        if transformer_ff_hidden_dims is None:
            transformer_ff_hidden_dims = [dim_model*4]
        self.self_attention_layer = SelfAttentionLayer(n_encoder_units, n_heads, dim_model, dim_queries,
                                                       dim_queries, dim_queries, transformer_ff_hidden_dims)
        self.intermediate_dropout = t.nn.Dropout(p=0.1)
        self.clf = MLP([dim_model, n_output_units], activation)

    def forward(self, queries: t.Tensor, attention_mask: t.Tensor) -> t.Tensor:
        embedded = self.self_attention_layer(queries, attention_mask)
        embedded = embedded.mean(axis=-2)
        dropout_output = self.intermediate_dropout(embedded)
        return self.clf(dropout_output)


class TransformerClassifier(nn.Module):
    def __init__(self,
                 n_encoder_units: int,
                 n_heads: int,
                 dim_model: int,
                 transformer_ff_hidden_dims: List[int],
                 n_labels: int,
                 vocab_size: int = 10000,
                 max_text_length: int = 50,
                 clf_hidden: int = 64, ) -> None:
        super().__init__()
        self.embedding_layer = Embedding(num_embeddings=vocab_size, embedding_dim=dim_model)
        self.position_embedding_layer = Embedding(num_embeddings=max_text_length, embedding_dim=dim_model)
        self.self_attention_layer = SelfAttentionLayer(
            n_encoder_units=n_encoder_units,
            n_heads=n_heads,
            dim_model=dim_model,
            dim_queries=dim_model,
            dim_keys=dim_model,
            dim_values=dim_model,
            ff_hidden_dims=transformer_ff_hidden_dims)
        self.clf = MLP(layer_sizes=[dim_model, clf_hidden, n_labels])

    def get_combined_embeddings(self, indices: t.Tensor, onehots: t.Tensor = None) -> t.Tensor:
        def get_token_embeddings():
            if onehots is None:
                return self.embedding_layer.forward(indices)
            else:
                return onehots @ self.embedding_layer.weight.data

        token_embeddings = get_token_embeddings()
        position_embeddings = \
            self.position_embedding_layer.forward(
                t.LongTensor(np.arange(0, indices.shape[-1])).view(-1, indices.shape[-1]))
        combined_embeddings = token_embeddings + position_embeddings
        return combined_embeddings

    def embed(self, indices: t.Tensor, combined_embeddings: t.Tensor = None, onehots: t.Tensor = None) -> t.Tensor:

        def get_combined_embeddings():
            if combined_embeddings is None:
                return self.get_combined_embeddings(indices=indices, onehots=onehots)
            else:
                return combined_embeddings

        combined_embeddings = get_combined_embeddings()
        signed = indices.sign().float()
        outer_product = t.einsum("bx,by->bxy", signed, signed)
        mask = t.where(outer_product > 0.5, t.tensor(0.0), t.tensor(float("-inf")))
        return self.self_attention_layer.forward(combined_embeddings, attention_mask=mask).mean(dim=-2)

    def forward(self, indices: t.Tensor, combined_embeddings: t.Tensor = None, onehots: t.tensor = None) -> t.Tensor:
        return self.clf(self.embed(indices=indices, combined_embeddings=combined_embeddings, onehots=onehots))


class TransformerModel(PytorchSaliencyModel, EmbeddingModel, Classifier):
    """
    Model used to prototype the whole api.
    Transformer with final mlp as softmax classifier
    """

    def __init__(self,
                 tokenizer: Tokenizer,
                 sequence_mapper: SequenceMapper,
                 transformer_classifier: TransformerClassifier,
                 label_mapper: LabelMapper):
        EmbeddingModel.__init__(self)
        PytorchSaliencyModel.__init__(self,
                                      tokenizer=tokenizer,
                                      sequence_mapper=sequence_mapper)
        Classifier.__init__(self, label_mapper=label_mapper)
        self.transformer_classifier = transformer_classifier
        self.label_mapper = label_mapper
        self._opt = t.optim.Adam(params=transformer_classifier.parameters())
        self._opt.zero_grad()
        self._xent = nn.CrossEntropyLoss()

    def predict_label_probabilities(self, **kwargs):

        logits = self.get_logits(**kwargs)
        softmax_probs = t.softmax(logits, dim=-1)
        if "as_tensor" in kwargs and kwargs["as_tensor"] == True:
            return softmax_probs
        return softmax_probs.detach().numpy()

    def train(self, **kwargs) -> "TransformerModel":
        loss = self.get_loss(as_tensor=True, **kwargs)["cross_entropy"]
        loss.backward()
        self._opt.step()
        self._opt.zero_grad()
        return self

    def preprocess(self, **kwargs) -> t.Tensor:
        return t.tensor(
            self.map_to_index_sequence(tokenized_texts=kwargs["clauses"], max_length=kwargs["max_length"])).long()

    def map_to_embedding_sequence(self,
                                  max_length: int,
                                  raw_texts: ListOfRawTexts = None,
                                  tokenized_texts: ListOfTokenizedTexts = None) -> Tensor:
        return self.transformer_classifier.get_combined_embeddings(
            indices=t.from_numpy(self.map_to_index_sequence(tokenized_texts=tokenized_texts,
                                                            max_length=max_length)).long())

    def get_logits(self, **kwargs):
        indices = self.preprocess(**kwargs)

        if "onehots" in kwargs and kwargs["onehots"] is not None:
            logits = self.transformer_classifier.forward(indices=indices, onehots=kwargs["onehots"])
        elif "embeddings" in kwargs and kwargs["embeddings"] is not None:
            logits = self.transformer_classifier.forward(indices=indices, combined_embeddings=kwargs["embeddings"])
        else:
            logits = self.transformer_classifier.forward(indices=indices)

        return logits

    def get_loss(self, **kwargs) -> Dict[str, Tensor]:
        logits = self.get_logits(**kwargs) if "logits" not in kwargs else kwargs["logits"]
        loss = self._xent(input=logits, target=t.tensor(self.label_mapper.map_to_indices(kwargs["labels"])).long())
        if "as_tensor" in kwargs and kwargs["as_tensor"] == True:
            return {"cross_entropy": loss}
        return {"cross_entropy": loss.detach().item()}

    @classmethod
    def from_components(cls, **kwargs) -> "TransformerModel":
        return cls(**kwargs)

    def get_components(self) -> dict:
        return {
            "tokenizer": self.tokenizer,
            "sequence_mapper": self.sequence_mapper,
            "transformer_classifier": self.transformer_classifier,
            "label_mapper": self.label_mapper}

    def get_numpy_parameters(self) -> Dict[str, np.ndarray]:
        return dict((name, p.detach().numpy())
                    for name, p in self.transformer_classifier.named_parameters())

    def embed(self, **kwargs) -> np.ndarray:
        return self.transformer_classifier.embed(indices=self.preprocess(**kwargs)).detach().numpy()

    def get_saliency_matrix(self, texts: ListOfRawTexts, label: int, max_length: int = None, **kwargs) -> Tensor:
        indices = t.from_numpy(self.map_to_index_sequence(tokenized_texts=texts, max_length=max_length)).long()
        embeddings = self.map_to_embedding_sequence(texts=texts, max_length=max_length)

        loss = self.get_loss_from_embeddings(embeddings=embeddings,
                                             indices=indices,
                                             labels=np.array([label for _ in texts]))["cross_entropy"]

        gradients = self.get_gradients(tensor=loss, with_respect_to=embeddings)
        self.transformer_classifier.zero_grad()
        saliency = -t.einsum("abc,acd -> ad", list(gradients), list(embeddings.permute((0, 2, 1)))).detach().numpy()
        return saliency

    def get_loss_from_embeddings(self, embeddings: t.Tensor, **kwargs) -> Dict[str, t.Tensor]:
        loss = self.get_loss(logits=self.transformer_classifier.forward(indices=kwargs["indices"],
                                                                        combined_embeddings=embeddings),
                             labels=kwargs["labels"],
                             as_tensor=True)
        return loss

    def get_loss_from_onehots(self, onehots: Tensor, **kwargs) -> Dict[str, t.Tensor]:
        loss = self.get_loss(logits=self.transformer_classifier.forward(indices=kwargs["indices"],
                                                                        onehots=onehots),
                             labels=kwargs["labels"],
                             as_tensor=True)
        return loss

    def get_module(self) -> t.nn.Module:
        return self.transformer_classifier

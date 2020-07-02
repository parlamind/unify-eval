import abc
import sys
from abc import ABC
from typing import List, Iterator, Dict, Callable

import numpy as np
from tqdm import tqdm


class DataLoader(ABC):
    @abc.abstractmethod
    def next_minibatch(self, minibatch_size: int = 16):
        pass

    @abc.abstractmethod
    def is_exhausted(self) -> bool:
        pass

    def yield_minibatches(self, minibatch_size: int = 16):
        while not self.is_exhausted():
            yield self.next_minibatch(minibatch_size=minibatch_size)

    @abc.abstractmethod
    def reset(self) -> "DataLoader":
        pass

    @abc.abstractmethod
    def is_lazy(self) -> bool:
        pass


class EagerDataLoader(DataLoader):

    def is_lazy(self) -> bool:
        return False

    @abc.abstractmethod
    def n_datapoints(self) -> int:
        pass

    def n_minibatches(self, minibatch_size: int) -> int:
        return int(np.ceil(self.n_datapoints() / minibatch_size))

    def yield_minibatches(self, minibatch_size: int = 16, progress_bar: bool = True):
        raw_yield = super().yield_minibatches(minibatch_size)
        return tqdm(iterable=raw_yield,
                    total=self.n_minibatches(minibatch_size=minibatch_size),
                    file=sys.stdout) if progress_bar else raw_yield


class LazyDataLoader(DataLoader):

    def is_lazy(self) -> bool:
        return True


class RandomIndicesLoader(EagerDataLoader):
    """
    samples random indices given a batch size and max index
    """

    def __init__(self, max_index: int):
        self.max_index = max_index
        self.n_minibatches = 0

    def is_exhausted(self):
        return False

    def next_minibatch(self, minibatch_size: int = 16) -> np.ndarray:
        minibatch_size = np.min([self.max_index, minibatch_size])
        return np.random.choice(self.max_index, minibatch_size, replace=False)

    def reset(self) -> "DataLoader":
        return self

    def n_datapoints(self) -> int:
        return self.max_index


class IndicesLoader(EagerDataLoader):
    """
    loads indices given max index and batch size. If exhausted, it can be reset
    """

    def __init__(self, max_index: int):
        self.max_index = max_index
        self.current_start_index = 0

    def next_minibatch(self, minibatch_size: int = 16) -> np.ndarray:
        upper_bound = np.min([self.current_start_index + minibatch_size, self.max_index])
        indices = np.arange(self.current_start_index, upper_bound)
        self.current_start_index = upper_bound
        return indices

    def is_exhausted(self) -> bool:
        return self.max_index == self.current_start_index

    def reset(self) -> "DataLoader":
        self.current_start_index = 0
        return self

    def n_datapoints(self) -> int:
        return self.max_index


class BatchLoader(EagerDataLoader):
    """
    loads samples from n different sources, given a batch size. If exhausted, it can be reset
    """

    def __init__(self, *data: np.ndarray):
        self.data = data
        self._indices_loader = IndicesLoader(data[0].shape[0])

    def next_minibatch(self, minibatch_size: int = 16) -> List[np.ndarray]:
        indices = self._indices_loader.next_minibatch(minibatch_size=minibatch_size)
        return [d[indices] for d in self.data]

    def reset(self) -> DataLoader:
        self._indices_loader.reset()
        return self

    def is_exhausted(self) -> bool:
        return self._indices_loader.is_exhausted()

    def n_datapoints(self) -> int:
        return self._indices_loader.max_index


class KeyedBatchLoader(EagerDataLoader):
    """
    loads samples from n different sources, given a batch size. If exhausted, it can be reset
    """

    def __init__(self, **data: np.ndarray):
        self.data = data
        self._indices_loader = IndicesLoader(data[list(data.keys())[0]].shape[0])

    def next_minibatch(self, minibatch_size: int = 16) -> Dict[str, np.ndarray]:
        indices = self._indices_loader.next_minibatch(minibatch_size=minibatch_size)
        return dict((k, d[indices]) for k, d in self.data.items())

    def reset(self) -> DataLoader:
        self._indices_loader.reset()
        return self

    def is_exhausted(self) -> bool:
        return self._indices_loader.is_exhausted()

    def n_datapoints(self) -> int:
        return self._indices_loader.max_index


class KeyedSubsampledBatchLoader(EagerDataLoader):
    """
    same as KeyedCorpusBatchLoader, but yielded full batch only consists of subset of actual full batch
    """

    def __init__(self, n_subsampled: int, **data: np.ndarray):
        self.data = data
        data_length = data[list(data.keys())[0]].shape[0]
        self.n_subsampled = np.min([n_subsampled, data_length])
        self.current_subsampled_indices = np.random.choice(data_length, n_subsampled, replace=False)
        self._current_data = dict((k, v[self.current_subsampled_indices]) for k, v in self.data.items())
        self._indices_loader = IndicesLoader(self.current_subsampled_indices.shape[0])

    def next_minibatch(self, minibatch_size: int = 16) -> Dict[str, np.ndarray]:
        indices = self.current_subsampled_indices[self._indices_loader.next_minibatch(minibatch_size=minibatch_size)]
        return dict((k, d[indices]) for k, d in self.data.items())

    def reset(self) -> DataLoader:
        self.current_subsampled_indices = np.random.choice(self.data[list(self.data.keys())[0]].shape[0],
                                                           self.n_subsampled,
                                                           replace=False)
        self._current_data = dict((k, v[self.current_subsampled_indices]) for k, v in self.data.items())
        self._indices_loader = IndicesLoader(self.current_subsampled_indices.shape[0])
        return self

    def is_exhausted(self) -> bool:
        return self._indices_loader.is_exhausted()

    def n_datapoints(self) -> int:
        return self._indices_loader.max_index


class KeyedLazyDataLoader(LazyDataLoader):
    """
    lazily loads data from some iterators. Iterators might be cyclic / infinitely long
    """

    def __init__(self, **data_iterators: object) -> object:
        self.data_iterators = data_iterators

    def next_minibatch(self, minibatch_size: int = 16) -> Dict[str, np.ndarray]:
        data = dict((key, []) for key in self.data_iterators)
        for i in range(minibatch_size):
            try:
                for key, data_iterator in self.data_iterators.items():
                    data[key].append(next(data_iterator))
            except:
                break

        return dict((key, np.array(d)) for key, d in data.items())

    def is_exhausted(self) -> bool:
        return False

    def reset(self) -> "KeyedLazyDataLoader":
        return self


class FiniteKeyedLazyDataLoader(KeyedLazyDataLoader):
    """
    same as KeyedLazyDataLoader, but expects data iterators to be finite.
    Useful for instance for finite but big data sets to be loaded lazily
    """

    def __init__(self, n_datapoints: int, **data_iterator_factories: Callable[[], Iterator]):
        super().__init__(**dict((k, v()) for k, v in data_iterator_factories.items()))
        self.data_iterator_factories = data_iterator_factories
        self.n_datapoints = n_datapoints
        self._is_exhausted = False

    def n_minibatches(self, minibatch_size: int) -> int:
        return int(np.ceil(self.n_datapoints / minibatch_size))

    def next_minibatch(self, minibatch_size: int = 16) -> Dict[str, np.ndarray]:

        data = dict((key, []) for key in self.data_iterators)
        for i in range(minibatch_size):
            try:
                for key, data_iterator in self.data_iterators.items():
                    data[key].append(next(data_iterator))
                    self._is_exhausted = False
            except:
                self._is_exhausted = True
                break
        return dict((key, np.array(d)) for key, d in data.items())

    def is_exhausted(self) -> bool:
        return self._is_exhausted

    def reset(self) -> "FiniteKeyedLazyDataLoader":
        self.data_iterators = dict((k, v()) for k, v in self.data_iterator_factories.items())
        self._is_exhausted = False
        return self

    def yield_minibatches(self, minibatch_size: int = 16, progress_bar: bool = True):
        raw_yield = super().yield_minibatches(minibatch_size)
        if progress_bar:
            return tqdm(iterable=raw_yield,
                        total=self.n_minibatches(minibatch_size=minibatch_size),
                        file=sys.stdout)
        return raw_yield


class CyclicIterator(Iterator):
    def __init__(self, iterator_factory):
        self.iterator_factory = iterator_factory
        self.iterator = self.iterator_factory()

    def __next__(self):
        try:
            return next(self.iterator)
        except:
            self.iterator = self.iterator_factory()
            return next(self.iterator)

    def __iter__(self) -> Iterator:
        return self

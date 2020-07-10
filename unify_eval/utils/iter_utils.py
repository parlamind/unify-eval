from itertools import chain, islice
from typing import Iterable


def flatmap(f, xs):
    return (y for ys in f(xs) for y in ys)


def intersperse(xs, delimiter):
    it = iter(xs)
    yield next(it)
    for x in it:
        yield delimiter
        yield x


def join(*iters):
    return (x for xs in iters for x in xs)


class IterableIterator(object):
    def __init__(self, iterable: Iterable):
        self.iterable: Iterable = iterable

    def __iter__(self):
        return self

    def __next__(self):
        try:
            v = next(self.iterable)
            return v
        except Exception:
            raise StopIteration


def sliding(l, slide_size: int, step_size=1):
    for i in range(0, len(l) - slide_size + step_size, step_size):
        result = l[i:i + slide_size]
        if len(result) == slide_size:
            yield result


def lazy_chunking(iterable, chunk_size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, chunk_size - 1))


def chunking(l, chunk_size: int):
    """
    returns a generator over equal-sizes chunks. remainder is cut off
    :param l:
    :param chunk_size:
    :return:
    """
    slices = sliding(l, slide_size=chunk_size, step_size=chunk_size)
    for slice in slices:
        if len(slice) == chunk_size:
            yield slice

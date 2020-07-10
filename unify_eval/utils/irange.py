class Irange():
    def __init__(self, start: int = 0, step_size: int = 1):
        self.i = start
        self.step_size = step_size

    def __next__(self):
        i = self.i
        self.i += self.step_size
        return i

    def __iter__(self):
        return self


def irange(start: int = 0, step_size: int = 1) -> Irange:
    return Irange(start=start, step_size=step_size)

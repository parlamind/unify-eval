def auto_repr(cls):
    """
    decorator that implements __repr__ for a given class
    :param cls:
    :return:
    """

    def repr(self):
        return f"{cls.__name__}({', '.join(f'{key}={value}' for key, value in vars(self).items())})"

    cls.__repr__ = repr
    return cls

# debugy.py

from functools import wraps, partial


def debug(func=None, *, prefix=''):
    # func is function to be wrapped
    if func is None:
        # func wasn't passed
        return partial(debug, prefix=prefix)
    msg = prefix + func.__qualname__
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(msg)
        return func(*args, **kwargs)
    return wrapper
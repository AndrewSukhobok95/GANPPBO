import numpy as np
from datetime import datetime


def verbose_info(verbose: bool, msg: str, verbose_endl: str = "    "):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if verbose:
                print(msg, end=verbose_endl)
                start_t = datetime.now()
            func(*args, **kwargs)
            if verbose:
                dur = (datetime.now() - start_t).total_seconds()
                print("DONE", "in {} min. {:.2f} sec.".format(int(dur // 60), dur - 60 * (dur // 60)))
        return wrapper
    return decorator


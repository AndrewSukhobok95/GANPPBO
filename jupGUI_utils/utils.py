import numpy as np
from datetime import datetime
from io import BytesIO
from PIL import Image



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


def widget_image_to_bytes(img):
    image = Image.fromarray((img * 255).astype(np.uint8), mode='RGB')
    f = BytesIO()
    image.save(f, 'png')
    return f.getvalue()


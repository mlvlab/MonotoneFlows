import numpy as np
from numpy.random import default_rng


# Dataset iterator
def inf_train_gen(data, batch_size=200, linspace=False):
    rng = default_rng()

    if data == "multiplestepfunc":
        def t(w):
            alpha = 0.05
            return np.minimum(np.maximum(np.maximum(0, alpha*w), np.minimum((1/alpha)*(w-(1-alpha)), 1)), 1)
        if linspace:
            x = np.linspace(-2., 2., batch_size).reshape((batch_size,1))
        else:
            x = rng.uniform(low=-2., high=2., size=(batch_size,1))
        y = t(x+2)+t(x+1)+t(x)+t(x-1)
        return x,y
    elif data == "impflowfunc":
        def t(w):
            alpha = 0.1
            return np.maximum(0, w/alpha) + np.minimum(0, w*alpha)
        if linspace:
            x = np.linspace(-1., 1., batch_size).reshape((batch_size,1))
        else:
            x = rng.uniform(low=-1., high=1., size=(batch_size,1))
        y = t(x)
        return x,y
    else:
        raise Exception('Unknown 1D dataset %s' % (data))
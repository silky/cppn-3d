"""
We follow basically the same structure of the JS version.
"""

import tensorflow as tf
import numpy      as np

from collections            import namedtuple
from tensorflow.python.keras        import Sequential
from tensorflow.python.keras.layers import Dense


Config = namedtuple("Config", 
                    [ "net_size"
                    , "num_dense"
                    , "input_size"
                    , "latent_dim"
                    , "activation_function"
                    , "colours"
                    , "norm"
                    ])


def build_model (config):
    model = Sequential()
    init  = tf.random_normal_initializer(mean=0, stddev=1)

    for k in range(config.num_dense):
        model.add(Dense( config.net_size
                       , batch_input_shape  = (None, config.input_size)
                       , kernel_initializer = init
                       , bias_initializer   = init
                       , activation         = config.activation_function))

    model.add(Dense(config.colours, activation = "sigmoid"))
    return model


def get_input_data (config, width, height):
    x = np.linspace(-1, 1, num = width)
    y = np.linspace(-1, 1, num = height)
    return get_input_data_(config, x, y)


def get_input_data_ (config, x, y):
    xx, yy = np.meshgrid(x, y)
    zz     = config.norm([xx, yy])
    r      = np.array([ xx.ravel(), yy.ravel(), zz.ravel() ])
    return np.transpose(r)


def stitch_together (yss):
    """ Given that we had to compute the things separately, let's stich them
        together.

        We know that our loop builds things like so:

        yss = [a, b, c, d]

        image (400x400)
             = 
                 a | b
                 -----
                 c | d

        and that everything will be a square, so e.g.

        yss = [a, b, c, d, e, f, g, h, i, j]
        image (600x600)
             = 

                 a | b | c
                 ---------
                 e | f | g
                 --------
                 h | i | j

        So a simple candidate plan is just to concat along rows first,
        then concat the rows horiztonally.
    """

    n      = len(yss)
    rows   = int(np.sqrt(n))
    result = []

    for r in range(rows):
        elts = np.take(yss, range(r*rows, (r+1)*rows), axis=0)
        elts = np.concatenate(elts, axis=1)
        result.append(elts)

    result = np.array(result)
    result = np.concatenate(result, axis=0)

    return result


def forward (config, model, z, width, height):
    # For convenience, let's make quite harsh restrictions.
    assert width == height

    max_size = 200
    if width > max_size:
        assert width  % max_size == 0
        # We just want to call "get_input_data_" with a subset
        # of x's and y's.

        # We know we want to go from -1 to 1 ultimately. So that's 2. So we
        # want to divide this into k = width / max_size blocks.
        start     = -1
        end       = 1
        r         = end - start
        k         = width // max_size
        step      = r / k
        step_size = step / max_size
        results   = []

        for i in range(k):
            y0 = start + (i * step)
            y  = np.arange(y0, y0 + step, step_size)

            for j in range(k):
                x0 = start + (j * step)
                x  = np.arange(x0, x0 + step, step_size)

                xs = get_input_data_(config, x, y)
                ys = forward_(config, model, z, max_size, max_size, xs)
                results.append(ys)

        return np.array(results)
    else:
        xs = get_input_data(config, width, height)
        ys = forward_(config, model, z, width, height, xs)
        return np.array([ys])


def forward_ (config, model, z, width, height, xs):

    ones = np.ones([xs.shape[0], 1])

    xs   = np.concatenate([xs, ones * z], axis=1)
    ys   = model.predict(xs)
    ys   = np.reshape(ys, (width, height, config.colours))

    return ys

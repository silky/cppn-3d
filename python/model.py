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
                    , "input_size"
                    , "latent_dim"
                    , "activations"
                    , "colours"
                    , "norms"
                    ])

Model = namedtuple("Model",
                    [ "z"
                    , "xs" # Input
                    , "ys" # Output
                    ])


def build_model (config):

    init       = tf.random_normal_initializer(mean=0, stddev=1)
    coord_dims = config.input_size - config.latent_dim
    xs         = tf.placeholder(tf.float32, shape = [ None, coord_dims ])

    # By default z will take on some random normal value.
    z_val = np.random.normal(0, 1, size=config.latent_dim)
    z     = tf.Variable(z_val, dtype=tf.float32)

    ones = tf.ones([tf.shape(xs)[0], 1])
    h    = tf.concat([xs, ones * z], axis = 1)

    for func in config.activations:
        h = tf.layers.dense( h
                           , config.net_size
                           , activation         = func
                           , kernel_initializer = init
                           , bias_initializer   = init
                           )

    ys = tf.layers.dense(h, config.colours, activation = tf.nn.sigmoid)

    model = Model( xs = xs
                 , ys = ys
                 , z  = z
                 )

    return model


def get_input_data (config, width, height):
    # Note: Changing the numbers here can have interesting results
    x = np.linspace(-1, 1, num = width)
    y = np.linspace(-1, 1, num = height)
    return get_input_data_(config, x, y)


def get_input_data_ (config, x, y):
    xx, yy = np.meshgrid(x, y)
    zz     = [ f([xx, yy]).ravel() for f in config.norms ]
    r      = np.array([ xx.ravel(), yy.ravel()] + zz )
    return np.transpose(r)


def stitch_together (yss):
    """ Given that we had to compute the things separately, let's stich them
        together.

        We know that our loop builds things like so:

        yss = [a, b, c, d]

        image (512x512)
             = 
                 a | b
                 -----
                 c | d

        and that everything will be a square, so e.g.

        yss = [a, b, c, d, e, f, g, h, i, j]
        image (768x768)
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


def forward (sess, config, model, z, width, height):
    # For convenience, let's make quite harsh restrictions.
    assert width == height

    max_size = 256
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
                ys = forward_(sess, config, model, z, max_size, max_size, xs)
                ys = np.reshape(ys, (max_size, max_size, config.colours))

                results.append(ys)

        return np.array(results)
    else:
        xs = get_input_data(config, width, height)
        ys = forward_(sess, config, model, z, width, height, xs)
        ys = np.reshape(ys, (max_size, max_size, config.colours))
        return np.array([ys])


def forward_ (sess, config, model, z, width, height, xs):
    ys = sess.run( model.ys, feed_dict = { model.z: z, model.xs: xs } )
    return ys

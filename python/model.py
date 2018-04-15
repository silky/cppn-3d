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
                      # = ["top"] or ["top", "bottom"]
                    , "border_matchings"
                    ])

Model = namedtuple("Model",
                    [ "z"
                    , "matching_loss"
                    , "matching_pixels"
                    , "xs" # Input
                    , "ys" # Output
                    ])


def build_model (config, width, height):
    init = tf.random_normal_initializer(mean=0, stddev=1)

    coord_dims = config.input_size - config.latent_dim

    xs              = tf.placeholder(tf.float32, shape = [ None, coord_dims ])

    # Everything will be arranged as:
    #
    #  [ x0 , w0
    #  , x1 , w1
    #  , x2 , w2
    #  , .    .
    #  , .    .
    #  , .    .
    #  ]
    
    matching_pixels = tf.placeholder(tf.float32, shape = [ None, None, coord_dims ])

    # By default z will take on some random normal value.
    z_val = np.random.normal(0, 1, size=config.latent_dim)
    z     = tf.Variable(z_val, dtype=tf.float32)

    ones = tf.ones([tf.shape(xs)[0], 1])
    h    = tf.concat([xs, ones * z], axis = 1)

    for k in range(config.num_dense):
        h = tf.layers.dense( h
                           , config.net_size
                           , activation         = config.activation_function
                           , kernel_initializer = init
                           , bias_initializer   = init
                           )

    ys = tf.layers.dense(h, config.colours, activation = tf.nn.sigmoid)
    ys = tf.reshape(ys, (width, height, config.colours))

    side_indexes = { "top":    (0, 1)
                   , "left":   (0, 1)
                   , "bottom": (height - 1, height)
                   , "right":  (width - 1, width)
                   }

    pixels = []

    for match_side in config.border_matchings:
        a, b = side_indexes[match_side]

        if match_side in ["top", "bottom"]:
            our_pixels = ys[a:b, :, :]
        else:
            our_pixels = tf.transpose(ys[:, a:b, :], perm=[1, 0, 2])

        pixels.append( our_pixels )


    if len(config.border_matchings) > 0:
        pixels        = tf.concat( pixels, axis = 0 )
        matching_loss = tf.norm(our_pixels - matching_pixels, ord=2)
    else:
        matching_loss = None

    model = Model( xs = xs
                 , ys = ys
                 , z  = z
                 , matching_loss   = matching_loss
                 , matching_pixels = matching_pixels
                 )

    return model


def get_input_data (config, width, height):
    # Note: Changing the numbers here can have interesting results
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


def forward (sess, config, model, z, width, height):
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
                ys = forward_(sess, config, model, z, max_size, max_size, xs)
                results.append(ys)

        return np.array(results)
    else:
        xs = get_input_data(config, width, height)
        ys = forward_(sess, config, model, z, width, height, xs)
        return np.array([ys])


def find_matching (config, model, matching_image):
    #
    #
    pass


def forward_ (sess, config, model, z, width, height, xs):
    ys = sess.run( model.ys, feed_dict = { model.z: z, model.xs: xs } )
    return ys

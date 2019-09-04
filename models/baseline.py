# External
import tensorflow as tf

import initializers


def downscale2d(x, n):
    """Box downscaling.

    Args:
    x: 4D tensor in NHWC format.
    n: integer scale.

    Returns:
    4D tensor down scaled by a factor n.
    """
    if n <= 1:
        return x
    if n % 2 == 0:
        x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")
        return downscale2d(x, n // 2)
    return tf.nn.avg_pool(x, [1, n, n, 1], [1, n, n, 1], "VALID")


def build_encoder(
    input, scales, depth, latent_dim=64, kernel_size=3, activation=tf.nn.leaky_relu
):
    model = tf.keras.layers.Conv2D(
        depth,
        kernel_size=1,
        padding="same",
        kernel_initializer=initializers.HeModifiedNormalInitializer(0.2),
    )(input)
    for scale in range(scales):
        model = tf.keras.layers.Conv2D(
            depth << scale,  # Double the number of channels for every added block
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer=initializers.HeModifiedNormalInitializer(0.2),
            activation=activation,
        )(model)

        model = tf.keras.layers.Conv2D(
            depth << scale,  # Double the number of channels for every added block
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer=initializers.HeModifiedNormalInitializer(0.2),
            activation=activation,
        )(model)

        model = tf.keras.layers.Conv2D(
            depth << scales,  # Double the number of channels for every added block
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer=initializers.HeModifiedNormalInitializer(0.2),
            activation=activation,
        )(model)

        model = tf.keras.layers.Conv2D(
            latent_dim,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer=initializers.HeModifiedNormalInitializer(0.2),
        )(model)
    return model

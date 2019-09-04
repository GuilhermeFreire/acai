# External
import tensorflow as tf


class HeModifiedNormalInitializer(tf.keras.initializers.RandomNormal):
    def __init__(self, slope):
        self.slope = slope

    def get_config(self):
        return dict(slope=self.slope)

    def __call__(self, shape, dtype=None, partition_info=None):
        del partition_info
        dtype = dtype or tf.float32
        std = tf.rsqrt(
            (1.0 + self.slope ** 2) * tf.cast(tf.reduce_prod(shape[:-1]), tf.float32)
        )
        return tf.random_normal(shape, stddev=std, dtype=dtype)

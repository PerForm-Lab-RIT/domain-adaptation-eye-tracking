import tensorflow as tf


class Swish(tf.keras.layers.Layer):
  def call(self, inputs):
    return inputs * tf.nn.sigmoid(inputs)
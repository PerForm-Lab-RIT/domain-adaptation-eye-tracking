import tensorflow as tf
# import tensorflow_addons as tfa
import tensorflow.keras.layers as tfkl
from .activation import Swish

  
class ReflectionPad2D(tf.keras.layers.Layer):
  # https://stackoverflow.com/a/53349976
  def __init__(self, padding=(1, 1)):
    super().__init__()
    self.padding = tuple(padding)

  def compute_output_shape(self, s):
    """ If you are using "channels_last" configuration"""
    return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

  def call(self, x):
    w_pad, h_pad = self.padding
    return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0] ], 'REFLECT')


class SimpleResidualBlock(tf.keras.Model):
  def __init__(self, in_features):
    super().__init__()

    self.block = tf.keras.Sequential([
      ReflectionPad2D((1, 1)),
      tfkl.Conv2D(in_features, 3, activation="relu"),
      ReflectionPad2D((1, 1)),
      tfkl.Conv2D(in_features, 3, activation="relu")
    ])

  def call(self, x):
    return x + self.block(x)
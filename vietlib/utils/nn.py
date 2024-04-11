from typing import List, Dict, Union, Tuple
import tensorflow as tf

def build_tf_model(model: tf.keras.Model, input_shape: Union[Tuple, List[Tuple]]) -> tf.keras.Model:
  """_summary_

  Args:
    model (tf.keras.Model): _description_
    input_shape (Union[Tuple, List[Tuple]]): _description_

  Returns:
    tf.keras.Model: _description_
  """
  batch_size = 2
  if isinstance(input_shape, tuple):
    _ = model(tf.random.normal([batch_size, *input_shape])) # (B, 8)
  elif isinstance(input_shape, list):
    list_var = []
    for _shape in input_shape:
      list_var.append(tf.random.normal([batch_size, *_shape]))
    _ = model(*list_var)
  else:
    raise NotImplementedError("build should take in the shape tuple or the list of tuple shape, otherwise, build yourself.")
  
  return model
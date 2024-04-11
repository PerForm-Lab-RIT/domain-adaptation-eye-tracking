import numpy as np
from collections import OrderedDict

class OUActionNoise(object):
  def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None) -> None:
    self.mu = mu
    self.theta = theta
    self.sigma = sigma
    self.dt = dt
    self.x0 = x0

    self.reset()

  def __call__(self):
    x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
      self.sigma * np.sqrt(self.dt) * np.random.normal(self.mu.shape)
    self.x_prev = x
    return x

  def reset(self):
    self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class GaussianNoise(object):
  def __init__(self, action_size: int, mu=0, sigma=0.1) -> None:
    # https://spinningup.openai.com/en/latest/algorithms/td3.html#documentation-pytorch-version. sigma = 0.1
    # mu is the mean,
    # sigma is the variance
    self.action_size = action_size # single integer
    self.mu = mu
    self.sigma = sigma

  def __call__(self):
    return np.random.normal(loc=self.mu, scale=self.sigma, size=(1, self.action_size))

NOISE_POOL = OrderedDict((
  ("ou", OUActionNoise),
  ("gaussian", GaussianNoise),
))

def get_noise_cls(noise_str: str):
  assert noise_str in NOISE_POOL, f"Should have config called `action_noise` that\
specify which noise to use, list of noise supported: {NOISE_POOL.keys()}"
  return NOISE_POOL[noise_str]



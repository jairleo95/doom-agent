import tensorflow as tf

class DDDQN(tf.keras.Model):

  def __init__(self, n_actions):
    super(DDDQN, self).__init__()
    self.d1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=4, activation='relu', padding="same")
    self.d2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu', padding="same")
    self.d3 = tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, activation='relu', padding="same")
    self.f = tf.keras.layers.Flatten()
    self.v = tf.keras.layers.Dense(1, activation=None)
    self.a = tf.keras.layers.Dense(n_actions, activation=None)

  def call(self, input_data):
    x = self.d1(input_data)
    x = self.d2(x)
    x = self.d3(x)
    x = self.f(x)
    v = self.v(x)
    a = self.a(x)
    Q = v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))
    return Q

  def advantage(self, state):
    x = self.d1(state)
    x = self.d2(x)
    x = self.d3(x)
    x = self.f(x)
    a = self.a(x)
    return a

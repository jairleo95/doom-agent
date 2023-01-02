import tensorflow as tf

#https://pylessons.com/CartPole-DDDQN
#https://github.com/pythonlessons/Reinforcement_Learning/blob/master/03_CartPole-reinforcement-learning_Dueling_DDQN/Cartpole_Double_DDQN_TF2.py


class DDDQN(tf.keras.Model):

  def __init__(self, n_actions):
    super(DDDQN, self).__init__()
    self.c1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=4, activation='relu', padding="same")
    self.c2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu', padding="same")
    self.c3 = tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, activation='relu', padding="same")
    self.f = tf.keras.layers.Flatten()
    self.v = tf.keras.layers.Dense(1, activation=None)
    self.a = tf.keras.layers.Dense(n_actions, activation=None)

  def call(self, input_data):
    x = self.c1(input_data)
    x = self.c2(x)
    x = self.c3(x)
    x = self.f(x)
    v = self.v(x)
    a = self.a(x)
    q_vals = v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))
    return q_vals

  def advantage(self, state):
    x = self.c1(state)
    x = self.c2(x)
    x = self.c3(x)
    x = self.f(x)
    a = self.a(x)
    return a

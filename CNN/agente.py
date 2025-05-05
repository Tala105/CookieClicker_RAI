
from typing_extensions import Self
import tensorflow as tf
import numpy as np
import random
from collections import deque
from queue import Queue

plot_queue = Queue()
checkpoint_queue = Queue()

class Agent:
    def __init__(self, state_size, action_size, graph=None, training=True, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, buffer_size=10000):
        self.graph = graph or tf.Graph()
        with self.graph.as_default():
            self.state_size = state_size
            self.action_size = action_size
            self.training = training
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay
            self.learning_rate = learning_rate
            self.replay_buffer = deque(maxlen=buffer_size)
            self.iteration_history = []
            self.score_history = []

            self._build_model()
            self.sess = tf.compat.v1.Session(graph=self.graph)
            self.sess.run(tf.compat.v1.global_variables_initializer())

    def _build_model(self):
        self.inputs = tf.compat.v1.placeholder(tf.float32, shape=(None, self.state_size), name="inputs")
        self.targets = tf.compat.v1.placeholder(tf.float32, shape=(None, self.action_size), name="targets")

        layer1 = tf.keras.layers.Dense(256, activation='relu')(self.inputs)
        layer2 = tf.keras.layers.Dense(128, activation='relu')(layer1)
        layer3 = tf.keras.layers.Dense(64, activation='relu')(layer2)
        layer4 = tf.keras.layers.Dense(32, activation='relu')(layer3)
        self.q_values = tf.keras.layers.Dense(self.action_size)(layer4)

        self.loss = tf.compat.v1.losses.mean_squared_error(self.targets, self.q_values)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def predict(self, state):
        return self.sess.run(self.q_values, feed_dict={self.inputs: state})

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.predict(state))
    
    def epsilon_reset(self):
        self.epsilon = 1.0

    def replay(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        batch = random.sample(self.replay_buffer, batch_size)
        states = np.array([x[0][0] for x in batch])
        targets = self.sess.run(self.q_values, feed_dict={self.inputs: states})

        for i, (state, action, reward, next_state, done) in enumerate(batch):
            if done:
                targets[i][action] = reward
            else:
                next_q = np.max(self.predict(next_state))
                targets[i][action] = reward + self.gamma * next_q

        self.sess.run(self.optimizer, feed_dict={self.inputs: states, self.targets: targets})

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        with self.graph.as_default():
            saver = tf.compat.v1.train.Saver()
            saver.save(self.sess, name)

    def load(self, name):
        with self.graph.as_default():
            saver = tf.compat.v1.train.Saver()
            saver.restore(self.sess, name)

    def load_from_agent(self, source_agent: Self):
        self.iteration_history = source_agent.iteration_history
        self.score_history = source_agent.score_history
        variables = tf.compat.v1.global_variables()
        values = source_agent.sess.run(variables)
        for var, value in zip(variables, values):
            self.sess.run(var.assign(value))
from keras import Model
from keras.layers import Input, Dense, Add, Subtract, Lambda
from keras.optimizers import Adam
from keras.models import load_model
import keras.backend as K

import numpy as np

from network import layers

import time

UPDATE_TARGET_EVERY = 10000

class DQN_Agent:
    def __init__(self, input_shape, num_actions, learning_rate, gamma, model_file=None):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = gamma
        if model_file is None:
            self.model = self.build_model()
        else:
            self.model = load_model(model_file)

        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        input_layer = Input(shape=self.input_shape)
        processed_layer = input_layer
        for layer in layers():
            processed_layer = layer(processed_layer)

        advantage = Dense(self.num_actions)(processed_layer)
        value = Dense(1)(processed_layer)
        output_layer = Lambda(lambda x: x[0] + x[1] - K.mean(x[1], keepdims=True))([value, advantage])

        model = Model(inputs=[input_layer], outputs=[output_layer])
        model.compile(loss='logcosh', optimizer=Adam(lr=self.lr), metrics=['accuracy'])
        return model

    def get_qs(self, state):
        return self.model.predict(np.expand_dims(state, axis=0))[0]

    def train(self, minibatch, step):

        minibatch_size = len(minibatch)
        current_states = np.array([exp[0]for exp in minibatch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([exp[3] for exp in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.gamma * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y), batch_size=minibatch_size,
                       verbose=0, shuffle=False)

        if step > 0 and (step % UPDATE_TARGET_EVERY == 0):
            self.target_model.set_weights(self.model.get_weights())

    def save(self, path):
        self.model.save(path)

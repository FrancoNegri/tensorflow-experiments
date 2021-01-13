import numpy as np
import random
from IPython.display import clear_output
from collections import deque

import keras

import plaidml.keras
import os
plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Sequential
from keras.layers import Input, Dense, Embedding, Reshape
from keras.optimizers import Adam
import json

print('Number of states: {}'.format(2))
print('Number of actions: {}'.format(1))


def binaryActivationFromTanh(x, threshold) :

    # convert [-inf,+inf] to [-1, 1]
    # you can skip this step if your threshold is actually within [-inf, +inf]

    activated_x = K.tanh(x)

    binary_activated_x = activated_x > threshold

    # cast the boolean array to float or int as necessary
    # you shall also cast it to Keras default
    # binary_activated_x = K.cast_to_floatx(binary_activated_x)

    return binary_activated_x

class Agent:
    def __init__(self, optimizer):
        
        # Initialize atributes
        self._state_size = 2
        # two actions:
        # 0 sell
        # 1 buy
        self._action_size = 2
        self._optimizer = optimizer
        
        self.expirience_replay = deque(maxlen=2000)
        
        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon = 0.2
        
        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        
        #self.q_network = keras.models.load_model('model')
        #self.target_network = keras.models.load_model('model')

        self.alighn_target_model()

    def store(self, state, action, reward, next_state):
        normalized_state = state / np.sqrt(np.sum(state**2))
        normalized_next_state = next_state / np.sqrt(np.sum(next_state**2))
        self.expirience_replay.append((state, action, reward, normalized_next_state))
    
    def _build_compile_model(self):
        model = Sequential()
        #model.add(Input(shape=(500,)))
        #model.add(Reshape((500,)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self._action_size, activation='sigmoid'))
        
        
        model.compile(loss='mse', optimizer=self._optimizer)
        model.build((None, 5))
        print(model)
        model.summary()
        return model

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, 1)
        normalized_state = state / np.sqrt(np.sum(state**2))
        #print(normalized_state)
        q_values = self.q_network.predict(normalized_state)
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch = random.sample(self.expirience_replay, batch_size)
        
        for state, action, reward, next_state in minibatch:
            target = self.q_network.predict(state)
            t = self.target_network.predict(next_state)
            target[0][action] = reward + self.gamma * np.amax(t)        
            self.q_network.fit(state, target, epochs=1, verbose=0)

    def save_model(self, i):
    	self.q_network.save("model_" + str(i))

class Kraken():
    def __init__(self, data, window_size):
        self.time = 0
        self.data = deque(data)
        self.window = deque()
        for i in range(0, window_size):
            frame = self.next_frame()
            self.window.append(frame)
        self.hodling = False

    def has_next(self):
        return len(self.data) > 1

    def trade(self,action):
        reward = 0
        if action == 1 and not self.hodling:
            self.buy_price = self.current_value[4]
            self.hodling = True
            print("Now holding at: " + str(self.buy_price))
        if action == 0 and self.hodling:
            self.hodling = False
            # fees of 0.4
            reward = (((self.current_value[4] * 100)) / self.buy_price - 100.4)*100
            print("Sold at: " + str(self.current_value[4]))
            print("Difference: " + str(reward) + " %") 
        return (reward, self.next())

    def next(self):
        self.window.pop()
        self.current_value = self.next_frame()
        print("Frame: " + str(self.current_value))
        self.window.append(self.current_value)
        return np.array(self.window)

    def next_frame(self):
    	frame = self.data.pop()
    	return np.array([frame["low"], frame["high"], frame["trades"], frame["open"], frame["close"]])

optimizer = Adam()
agent = Agent(optimizer)

batch_size = 32
num_of_episodes = 8
timesteps_per_episode = 20000
agent.q_network.summary()
window_size = 20

for e in range(0, num_of_episodes):
    # Reset the enviroment
    file_name = 'candles_5/tr_0' + str(e)
    print("Running: " + file_name)
    file = open(file_name, 'r')
    candles = file.readlines() 
    formated_candles = [ json.loads(candle) for candle in candles]
    kraken = Kraken(formated_candles, window_size)

    # Initialize variables
    reward = 0
    state = kraken.next()
    #state = np.reshape(state, [1, window_size])

    while kraken.has_next():
        # Run Action
        #enviroment.render()
        action = agent.act(state)
        # Take action
        print(action)
        reward, next_state = kraken.trade(action)

        #next_state = np.reshape(next_state, [1, window_size])
        agent.store(state, action, reward, next_state)
        state = next_state
        
        if len(agent.expirience_replay) > batch_size:
            agent.retrain(batch_size)
    print("Saving model...")
    agent.save_model(e)
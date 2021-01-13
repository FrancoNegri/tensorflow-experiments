import numpy as np
import random
from IPython.display import clear_output
from collections import deque
import progressbar

import gym

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
from collections import deque

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
        self.alighn_target_model()

    def store(self, state, action, reward, next_state):
        normalized_state = state / np.sqrt(np.sum(state**2))
        normalized_next_state = next_state / np.sqrt(np.sum(next_state**2))
        self.expirience_replay.append((state, action, reward, normalized_next_state))
    
    def _build_compile_model(self):
        model = Sequential()
        model.add(Input(shape=(500,)))
        model.add(Reshape((500,)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self._action_size, activation='sigmoid'))
        
        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, 1)
        normalized_state = state / np.sqrt(np.sum(state**2))
        q_values = self.q_network.predict(normalized_state)
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch = random.sample(self.expirience_replay, batch_size)
        
        for state, action, reward, next_state in minibatch:
            target = self.q_network.predict(state)
            t = self.target_network.predict(next_state)
            target[0][action] = reward + self.gamma * np.amax(t)        
            self.q_network.fit(state, target, epochs=1, verbose=0)

class Kraken():
    def __init__(self, data, window_size):
        self.time = 0
        self.data = deque(data)
        self.window = deque()
        for i in range(0, window_size):
            price = self.data.pop()
            self.window.append(price)
        self.hodling = False

    def has_next(self):
        return len(self.data) > 1

    def trade(self,action):
        reward = 0
        if action == 1 and not self.hodling:
            self.buy_price = self.current_value
            self.hodling = True
            print("Now holding at: " + str(self.buy_price))
        if action == 0 and self.hodling:
            self.hodling = False
            # implement fees
            reward = ((self.current_value * 100)) / self.buy_price - 100
            print("Sold at: " + str(self.current_value))
            print("Difference: " + str(reward) + " %") 
        return (reward, self.next())

    def next(self):
        self.window.pop()
        self.current_value = self.data.pop()
        print("Price is now: " + str(self.current_value))
        self.window.append(self.current_value)
        return self.window


optimizer = Adam(learning_rate=0.01)
agent = Agent(optimizer)

batch_size = 32
num_of_episodes = 60
timesteps_per_episode = 20000
agent.q_network.summary()
window_size = 500

for e in range(0, num_of_episodes):
    # Reset the enviroment

    file = open('trades/tr_' + str(e), 'r') 
    new_price_volume_pairs = file.readlines() 
    prices = [float(price_volume_pair.split(" ")[0]) for price_volume_pair in new_price_volume_pairs]
    kraken = Kraken(prices, window_size)

    # Initialize variables
    reward = 0
    state = kraken.next()
    state = np.reshape(state, [1, window_size])
    bar = progressbar.ProgressBar(maxval=timesteps_per_episode/10, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    while kraken.has_next():
        # Run Action
        #enviroment.render()
        action = agent.act(state)
        # Take action
        reward, next_state = kraken.trade(action)

        next_state = np.reshape(next_state, [1, window_size])
        agent.store(state, action, reward, next_state)
        state = next_state
        
        if len(agent.expirience_replay) > batch_size:
            agent.retrain(batch_size)
    
    bar.finish()
    if (e + 1) % 10 == 0:
        print("**********************************")
        print("Episode: {}".format(e + 1))
        enviroment.render()
        print("**********************************")
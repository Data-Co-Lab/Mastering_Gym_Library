import random
import gym
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from time import time

from colab_preview.video import wrap_env, show_video



Gamma = 0.95
LearningRate = 0.001
Memory = 1000000
BatchSize = 15
Exploration= 1.0
ExplorationLimit = 0.1
ExplorationDecay = 0.995

class Agent:

    def __init__(self, observation_space, action_space) :
        self.Exploration = Exploration
        self.action_space = action_space
        self.observation_space = observation_space
        self.memory = deque(maxlen=Memory)
        self.model = Sequential()

    def NeuralNet(self) :
        self.model.add(Dense(32, input_shape=(self.observation_space,), activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(self.action_space, activation="softmax"))
        self.model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=LearningRate))

    def CNN(self) :
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=(32, 32, 1)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.action_space, activation='softmax'))
        self.model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=LearningRate))

    def Save(self, state, action, reward, next_state, done) :
        self.memory.append((state, action, reward, next_state, done))

    def Action(self,state) :
        if np.random.rand() < self.Exploration:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def Update(self):
        if len(self.memory) < BatchSize:
            return
        batch = random.sample(self.memory, BatchSize)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + Gamma * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values,batch_size=None, verbose=0)
        self.Exploration *= ExplorationDecay
        self.Exploration = max(ExplorationLimit, self.Exploration)

def prepare(state) :
    output = tf.image.rgb_to_grayscale(state)
    output = tf.image.crop_to_bounding_box(output, 34, 0, 160, 160)
    output = tf.image.resize(output,[32, 32],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    output = tf.squeeze(output)
    array = K.eval(output)
    return np.reshape(array, (1, 32,32,1))

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


def Breakout():
    env = gym.make("Breakout-v0")
    env = wrap_env(env)
    observation_space = env.observation_space.shape#[0]
    action_space = env.action_space.n
    print('****************')
    print(observation_space)
    print(action_space)
    print('****************')
    agent = Agent(observation_space, action_space)
    agent.CNN()
    Episode = 0
    n = 50 #n is number of episodes
    while Episode < n :
        Episode += 1
        state = env.reset()
        #plt.imshow(state)
        state = np.reshape(state, (1, observation_space[0],observation_space[1],observation_space[2]))
        state = prepare(state)
        episode_reward = 0
        print("Epoch number {}".format(Episode))
        while True:
            #env.render()
            start = time()
            action = agent.Action(state)
            state_next, reward, terminal, info = env.step(action)
            
            episode_reward += reward
            state_next = np.reshape(state_next, (1, observation_space[0],observation_space[1],observation_space[2]))
            state_next=prepare(state_next)
            agent.Save(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print ("Episode: " + str(Episode) + ", exploration: " + str(agent.Exploration) + ", score: " + str(episode_reward))
                break
            agent.Update()
            print(f"While is taking {time()-start}s")
        show_video()

if __name__ == "__main__":
    Breakout()

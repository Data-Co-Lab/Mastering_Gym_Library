import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.optimizers import Adam

Gamma = 0.95
LearningRate = 0.001
Memory = 1000000
BatchSize = 15
Exploration= 1.0
ExplorationLimit = 0.1
ExplorationDecay = 0.9999

gym.envs.register(
    id='MountainCarContinuousCustom-v0',
    entry_point='gym.envs.classic_control:Continuous_MountainCarEnv',
    max_episode_steps=500,
    reward_threshold=-110.0,)

class Agent:

    def __init__(self, observation_space, action_low, action_high) :
        self.Exploration = Exploration
        self.action_low = action_low
        self.action_high = action_high
        self.observation_space = observation_space
        self.memory = deque(maxlen=Memory)
        self.model = Sequential()

    def NeuralNet(self) :
        self.model.add(Dense(24, input_shape=(self.observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(1, activation="tanh"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LearningRate))

    def Save(self, state, action, reward, next_state, done) :
        self.memory.append((state, action, reward, next_state, done))

    def Action(self,state) :
        if np.random.rand() < self.Exploration:
            return random.uniform(-1, 1)
        q_values = self.model.predict(state)
        return q_values

    def Update(self):
        if len(self.memory) < BatchSize:
            return
        batch = random.sample(self.memory, BatchSize)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + Gamma * self.model.predict(state))
            q_values = self.model.predict(state)
            q_values = q_update
            self.model.fit(state, [q_values], verbose=0)
        self.Exploration *= ExplorationDecay
        self.Exploration = max(ExplorationLimit, self.Exploration)

def MountainCarContinuous():
    env = gym.make("MountainCarContinuousCustom-v0")
    observation_space = env.observation_space.shape[0]
    action_low = env.action_space.low[0]
    action_high = env.action_space.high[0]
    agent = Agent(observation_space, action_low, action_high)
    agent.NeuralNet()
    Episode = 0
    n = 50 #n is number of episodes
    while Episode < n :
        Episode += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        Score = 500
        Step = 0
        while Step < 500:
            Score -= 1
            Step += 1
            env.render()
            action = agent.Action(state)
            state_next, reward, terminal, info = env.step([action])
            state_next = np.reshape(state_next, [1, observation_space])
            agent.Save(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print ("Episode: " + str(Episode) + ", exploration: " + str(agent.Exploration) + ", score: " + str(Score))
                break
            agent.Update()

if __name__ == "__main__":
    MountainCarContinuous()

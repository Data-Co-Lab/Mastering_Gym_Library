# Mastering the gym library  
  
**Disclaimer** : this is our first Reinforcement learning project and our main focus was the understanding of the Gym library, writing our own simplified code and adapting it to several classic games and Atari games.  
If this is your first look into RL I highly recommend [**DeepLizard's Reinforcement Learning - Goal Oriented Intelligence**](https://www.youtube.com/watch?v=nyjbcRQ-uQ8&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv).  
NOTE : This notebook assumes basic knowledge of RL and Deep learning.  
## Understanding OpenAI's Gym :  
The gym library is a collection of test problems — **environments** — that you can use to work out your reinforcement learning algorithms. These environments have a shared interface, allowing you to write general algorithms.  
Arguably the 2 most important variables in our training process are **Observation Space** and **Action Space** that are returned as either a **Discrete** or a **Box** (2 custom Gym classes).  
  
**The Box space** represents an n-dimensional box, so valid observations will be an array of n numbers.  
For example for MountainCar the Observation is Box(2) : position (-1.2;0.6) and velocity (-0.07;0.07) 
  
**The Discrete space** allows a fixed range of non-negative numbers.  
For example for MountainCar The Action space is Discrete(3) : (0,push left); (1,no push); (2,push right)  
In the other hand for **MountainCarContinuous** The Action space is Box(1) : Push car to the left (negative value) or to the right (positive value).  
  
## The Code :   
```Python
Gamma = 0.95              # [0,1] affects the output of the Bellman equation (Update function) the higher the value the more importance we give  to long term reward.
LearningRate = 0.001      # [0,1] instead of updating the weight with the full amount, it is scaled by the learning rate.
Exploration= 1.0          # The maximal value of the Exploration rate.
ExplorationLimit = 0.1    # The min value of the Exploration rate.
ExplorationDecay = 0.995  # The decay of the exploration rate over time.
```
These parameters can be tweaked accordingly to obtain improved results.  
  
The entire class **Agent** remains unchanged for **the classic games** except for **Mountain Car Continuous** where several changes were made the Neural Network, Action and Update functions since the Action space is no longer a Discrete like the other examples.  
The initialisation of the Agent now requires 3 inputs :  
```Python
agent = Agent(observation_space, action_low, action_high)
```  
to access the lowest and highest values possible of the Action space :  
```Python
action_low = env.action_space.low[0]
action_high = env.action_space.high[0]
```
The score displayed also changes depending on the rules of the game, if the game is about staying longer "alive" like cartpole agent scores increments every frame however if it's about reaching the goal as fast as possible then the score decreases over time.  
  
```Python
def Action(self,state) : # Taken from MountainCarCont.py 
    if np.random.rand() < self.Exploration:
        return random.uniform(-1, 1) #random value (float) between -1 and 1
    q_values = self.model.predict(state)
    return q_values
```  
The action function determines if the model is going to rely on exploration or the q-table (output of our neural net) a random number is generated if it's inferior to the exploration rate a random action is taken otherwise the action is predcited by our model.

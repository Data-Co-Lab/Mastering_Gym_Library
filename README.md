# Mastering the gym library  
  
**Disclaimer** : this is our first Reinforcement learning project and our main focus was the understanding of the Gym library, writing our own simplified code and adapting it to several classic games and Atari games.  
If this is your first look into RL I highly recommend [**DeepLizard's Reinforcement Learning - Goal Oriented Intelligence**](https://www.youtube.com/watch?v=nyjbcRQ-uQ8&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv).  
## Understanding OpenAI's Gym :  
The gym library is a collection of test problems — **environments** — that you can use to work out your reinforcement learning algorithms. These environments have a shared interface, allowing you to write general algorithms.  
Arguably the 2 most important variables in our training process are **Observation Space** and **Action Space** that are returned as either a **Discrete** or a **Box** (2 custom Gym classes).  
  
**The Box space** represents an n-dimensional box, so valid observations will be an array of n numbers.  
For example for MountainCar the Observation is Box(2) : position (-1.2;0.6) and velocity (-0.07;0.07) 
  
**The Discrete space** allows a fixed range of non-negative numbers.  
For example for MountainCar The Action space is Discrete(3) : (0,push left); (1,no push); (2,push right)  
In the other hand for **MountainCarContinuous** The Action space is Box(1) : Push car to the left (negative value) or to the right (positive value).  
  


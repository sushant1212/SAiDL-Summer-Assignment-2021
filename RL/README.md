# RL Optimization Problem

## Problem Statement
- Train an RL agent to optimize in a convex optimiztion environment. The coefficients of the objective function is visible to the agent. For simplicity, the environment is considered to be a general quadratic function.

## Environment Description
- Objective function is of the form ax^2 + by^2 + cxy + dx + ey + f
- To make the Hessian positive, (so that there is a global minima), c has been restricted to be between -2\*sqrt(ab) and 2\*sqrt(ab)
- The values of a,b,d,e,f are restricted to be between -10 and 10
- x and y denote the current coordinates of the environment

## State Description
- The state is given by the vector \[x,y,a,b,c,d,e,f\]

## Termination Condition : 
- If the agent reaches the global minima or if it has taken 200 steps in the environment, then the episode terminates.

## Approaches: 

### REINFORCE Algorithm
I implemented the REINFORCE algorithm in this environment and the details are as follows:
- The network inputs a state vector and outputs the mean and the std deviation for both x and y dimension.
- The action is then sampled using a Gaussian distribution. 

* I tested the REINFORCE algorithm on the cartpole environment to verify the algorithm. The diagram below shows the total rewards/episode for the agent.
![REINFORCE_cartpole](https://user-images.githubusercontent.com/57453637/129475208-1257eb26-b331-43a3-8dda-ffae43083fe9.jpeg)


#### Reward Functions tried :
1. Inverse distance : The reward per step was taken to be the inverse of the distance from the global minima. The results obtained is shown below :
![reward](https://user-images.githubusercontent.com/57453637/129475370-18714af9-1a25-4f58-8d38-b889902d49d8.png)

The agent could not learn much as we can see from the diagram.

2. Length of step taken towards the global minima : For each step, the reward that the agent gets is the length of the step that it takes in the direction of the global minima. This can be easily obtained using the dot product of the vector which joins the current state and next state with the vector joining the current state with the global minima. 

![reward_reinforce](https://user-images.githubusercontent.com/57453637/129476064-3466941c-bcf7-40d4-a4fe-4b8820bc8895.png)


Again, the agent found it diffcult to understand the environment.


While surfing the internet, I came across this [blog](https://ai.stackexchange.com/questions/23847/what-is-the-loss-for-policy-gradients-with-continuous-actions). Quoting a part of it "The continuous action space code should be correct but the agent will not learn because it is harder to learn the right actions with a continuous action space and our simple method isn't enough. Look into more advanced methods like PPO and DDPG." 
So I attempted using the DDPG algorithm.


### DDPG Algorithm
#### Implementation Details : 
- Actor network produces the action to be taken given an input state. The action has been kept between -0.5 and +0.5. Which means that in a step, the agent can increase or decrease each coordinate by a maximum of 0.5.
- The critic network takes in a state and an action and outputs the Q value of the state action pair.


Again, to check if my implementation was correct, I verified first on the Pendulum-v0 environment from gym and these were the results. : 
![reward_ddpg_pendulum](https://user-images.githubusercontent.com/57453637/129475958-6d877ff2-0e6a-4185-b76e-eec3f198c239.png)

As we can see, the agent has learnt properly which shows that the algorithm works fine.


### Reward functions used : 

- Inverse distance reward function :
These are the results which I obtained : 

![reward (1)](https://user-images.githubusercontent.com/57453637/129476131-216a6516-df26-465d-a6e3-5b023ead3f60.png)
The agent did not perform well using this reward function.

- Negative of the distance : 
![reward (2)](https://user-images.githubusercontent.com/57453637/129476196-af82626b-7db1-4265-b742-ebc8966001b6.png)
The agent could not perform well in this case as well. 

- Length of step taken towards the global minima : 
![reward (3)](https://user-images.githubusercontent.com/57453637/129476414-f94c6c9c-5355-42b4-98e2-7cb27b92dc75.png)

It was interesting to note the sudden increase in the total rewards for some time in the initial part. The increase in the peak was after the ```START STEPS``` number of steps. But later the agent did not learn properly. 



In order to investigate what was going on, I tried plotting the paths which the agent takes. Here are some of the samples. The black point shows the start of the episode, while the green point is the global minima. The red lines show the trajectory followed by the agent : 

![Screenshot from 2021-08-10 22-58-05](https://user-images.githubusercontent.com/57453637/129476535-31fd6011-7b6d-4ca3-9968-65b2312aa846.png)
![Screenshot from 2021-08-10 23-01-07](https://user-images.githubusercontent.com/57453637/129476541-d87fb17d-cb41-4e22-9600-3075c2bf3e93.png)
![Screenshot from 2021-08-10 23-07-54](https://user-images.githubusercontent.com/57453637/129476545-e1920507-e9de-4e6a-acfc-65f86f66ecb5.png)
![Screenshot from 2021-08-10 22-51-49](https://user-images.githubusercontent.com/57453637/129476670-9816964b-1659-4516-99b4-ecb9dfb4c91b.png)
![Screenshot from 2021-08-11 08-27-16](https://user-images.githubusercontent.com/57453637/129476673-e6588850-9f73-4292-bff4-a424efaed7db.png)
![Screenshot from 2021-08-10 22-23-33](https://user-images.githubusercontent.com/57453637/129476677-47119ae0-6940-4e2f-b151-7e146f229cac.png)
![Screenshot from 2021-08-10 23-47-21](https://user-images.githubusercontent.com/57453637/129476681-fa34e262-3c37-4272-a886-73137968265d.png)



What I observed was that the agent starts clustering in the end and thinks that the global minima is somewhere else. I was not able to work around this issue. There were some (very rare) cases in which the agent converged as well. 

- To try out some reward function other than these, I tried using a reward function which gave -1 if the agent is more than a threshold distance away from the global minima and inverse of the distance if it is inside the threshold, in the hope that it could try and learn to find the global miniam. But the agent was not able to perform well and we can see a lot of -200 rewards during most of the episodes:
![reward (4)](https://user-images.githubusercontent.com/57453637/129476951-b225a834-0e4e-4f72-89b9-bf28ff3f7e1e.png)


## Conclusions : 
- This environment is a bit difficult for the agent to learn. I tried plugging different hyperparameters for the algorithm but was not successful. 

# Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)
  
## Algorithm
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.  


The [DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html#id1) algorithm is an policy based method that implements and actor-critic archicteture.  

The pseudocode is [described](https://spinningup.openai.com/en/latest/algorithms/ddpg.html#pseudocode) as follows:  
 ![Pseudocode](/images/pseudocode.png)  

The [MADDPG](https://arxiv.org/pdf/1706.02275.pdf) algorithm is based on the DDPG considering some changes to overcome the challenges of non-stationary enviroments:  
_We adopt the framework of centralized training with decentralized execution, allowing the policies
to use extra information to ease training, so long as this information is not used at test time. It is
unnatural to do this with Q-learning without making additional assumptions about the structure of the
environment, as the Q function generally cannot contain different information at training and test
time. Thus, we propose a simple extension of actor-critic policy gradient methods where the critic is
augmented with extra information about the policies of other agents, while the actor only has access
to local information. After training is completed, only the local actors are used at execution phase,
acting in a decentralized manner and equally applicable in cooperative and competitive settings.
Since the centralized critic function explicitly uses the decision-making policies of other agents, we
additionally show that agents can learn approximate models of other agents online and effectively use
them in their own policy learning procedure. We also introduce a method to improve the stability of
multi-agent policies by training agents with an ensemble of policies, thus requiring robust interaction
with a variety of collaborator and competitor policies. We empirically show the success of our
approach compared to existing methods in cooperative as well as competitive scenarios, where agent
populations are able to discover complex physical and communicative coordination strategies._  

The MADDPG algorithm is available at the appendis of the [paper](https://arxiv.org/pdf/1706.02275.pdf):  
 ![MADDPG](/images/algorithm.PNG)  

The code is available ate the _ddpg_agent.py_ and _maddpg_agent.py_ files  
  
## Archicteture
Actor  
 - Input: (input, 128)   
 - BatchNorm(128) - ReLU  
 - Hidden: (128, 64) - ReLU  
 - Output: (64, action_size=4) - TanH  
   
Critic
 - Input: (input, 128) 
 - BatchNorm(128) - ReLU
 - Hidden: (128, 64) - ReLU
 - Output: (64, 1)  
   
 ## Hyperparameters  
BUFFER_SIZE = int(1e5)  # replay buffer size  
BATCH_SIZE = 128        # minibatch size  
GAMMA = 0.99            # discount factor  
TAU = 1e-3              # for soft update of target parameters  
LR_ACTOR = 2e-4         # learning rate of the actor   
LR_CRITIC = 2e-4        # learning rate of the critic  
WEIGHT_DECAY = 1.0        # L2 weight decay  

 
## Rewards
The resulting plot of the rewards during the trainings is as follows:  
![Reward](/images/reward.PNG)
  
  The enviroment was solved in in 1638 episodes with an Average Score of 0.50  
  
## Future work  
Trying to optimize the hyperparameters in a more controlled and systematic way (genetic algorithms, grid search, etc), implementing the the soccer game and allow multiple learnings at each step.
  
  References: [fdasilva59](https://github.com/fdasilva59/Udacity-DRL-Collaboration-and-Competition)

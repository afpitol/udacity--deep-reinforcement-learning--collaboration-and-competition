import numpy as np
import random
import copy
from collections import namedtuple, deque
from ddpg_agent import Agent, MAReplayBuffer

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 2e-4         # learning rate of the actor 
LR_CRITIC = 2e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
CLIP_CRITIC_GRADIENT = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def encode(sa):
    """
    Encode an Environment state or action list of array, which contain multiple agents action/state information, 
    by concatenating their information, thus removing (but not loosing) the agent dimension in the final output. 
    
    The ouput is a list intended to be inserted into a buffer memmory originally not designed to handle multiple 
    agents information, such as in the context of MADDPG)
    
    Params
    ======       
            sa (listr) : List of Environment states or actions array, corresponding to each agent
                
    """
    return np.array(sa).reshape(1,-1).squeeze()



def decode(size, num_agents, id_agent, sa, debug=False):
    """
    Decode a batch of Environment states or actions, which have been previously concatened to store 
    multiple agent information into a buffer memmory originally not designed to handle multiple 
    agents information(such as in the context of MADDPG)
    
    This returns a batch of Environment states or actions (torch.tensor) containing the data 
    of only the agent specified.
    
    Params
    ======
            size (int): size of the action space of state spaec to decode
            num_agents (int) : Number of agent in the environment (and for which info hasbeen concatenetaded)
            id_agent (int): index of the agent whose informationis going to be retrieved
            sa (torch.tensor) : Batch of Environment states or actions, each concatenating the info of several 
                                agents (This is sampled from the buffer memmory in the context of MADDPG)
            debug (boolean) : print debug information
    
    """
    
    list_indices  = torch.tensor([ idx for idx in range(id_agent * size, id_agent * size + size) ]).to(device)    
    out = sa.index_select(1, list_indices)
   
    if (debug):
        print("\nDebug decode:\n size=",size, " num_agents=", num_agents, " id_agent=", id_agent, "\n")
        print("input:\n", sa,"\n output:\n",out,"\n\n\n")
    return  out

class MADDPG():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize a MADDPG Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            random_seed (int): random seed
        """

        super(MADDPG, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        
        # Instantiate Multiple  Agent
        self.agents = [ Agent(state_size,action_size, random_seed, num_agents) for i in range(num_agents) ]
        
        # Instantiate Memory replay Buffer (shared between agents)
        self.memory = MAReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        #self.memory.add(state, action, reward, next_state, done)        
        self.memory.add(np.array(state).reshape(1,-1).squeeze(), 
                        np.array(action).reshape(1,-1).squeeze(), 
                        reward, 
                        np.array(next_state).reshape(1,-1).squeeze(), 
                        done)
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:            
            # Sample a batch of experience from the replay buffer 
            experiences = self.memory.sample()   
            # Update Agent #0
            self.MA_learn(experiences, own_idx=0, other_idx=1)
            # Sample another batch of experience from the replay buffer 
            experiences = self.memory.sample()   
            # Update Agent #1
            self.MA_learn(experiences, own_idx=1, other_idx=0)       
            
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        return [ agent.act(state, add_noise) for agent, state in zip(self.agents, state) ]

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def MA_learn(self, experiences, own_idx, other_idx, gamma=GAMMA):
        """
        Update the policy of the MADDPG "own" agent. The actors have only access to agent own 
        information, whereas the critics have access to all agents information.
        
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(states) -> action
            critic_target(all_states, all_actions) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            own_idx (int) : index of the own agent to update in self.agents
            other_idx (int) : index of the other agent to update in self.agents
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
               
        # Filter out the agent OWN states, actions and next_states batch
        own_states =  decode(self.state_size, self.num_agents, own_idx, states)
        own_actions = decode(self.action_size, self.num_agents, own_idx, actions)
        own_next_states = decode(self.state_size, self.num_agents, own_idx, next_states) 
                
        # Filter out the OTHER agent states, actions and next_states batch
        other_states =  decode(self.state_size, self.num_agents, other_idx, states)
        other_actions = decode(self.action_size, self.num_agents, other_idx, actions)
        other_next_states = decode(self.state_size, self.num_agents, other_idx, next_states)
        
        # Concatenate both agent information (own agent first, other agent in second position)
        all_states=torch.cat((own_states, other_states), dim=1).to(device)
        all_actions=torch.cat((own_actions, other_actions), dim=1).to(device)
        all_next_states=torch.cat((own_next_states, other_next_states), dim=1).to(device)
   
        agent = self.agents[own_idx]
        
            
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models        
        all_next_actions = torch.cat((agent.actor_target(own_states), agent.actor_target(other_states)),
                                     dim =1).to(device) 
        Q_targets_next = agent.critic_target(all_next_states, all_next_actions)
        
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = agent.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        if (CLIP_CRITIC_GRADIENT):
            torch.nn.utils.clip_grad_norm(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        all_actions_pred = torch.cat((agent.actor_local(own_states), agent.actor_local(other_states).detach()),
                                     dim = 1).to(device)      
        actor_loss = -agent.critic_local(all_states, all_actions_pred).mean()
        
        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()        
        agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        agent.soft_update(agent.critic_local, agent.critic_target, TAU)
        agent.soft_update(agent.actor_local, agent.actor_target, TAU)                   
  
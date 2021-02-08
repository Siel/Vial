#!/usr/bin/env python3
# coding: utf-8

# In[1]:


from npodEnv import NpodEnv
import numpy as np
env = NpodEnv()
env.ver()


# In[2]:


def argmaxQ(Q, s, n_actions):
    if s in Q.keys():
        return(np.argmax(Q[s]))
    else:
        return(np.random.randint(0, n_actions))


# In[3]:


def policy_greedy(Q, s, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return(argmaxQ(Q,s,n_actions))
    else:
        return(np.random.randint(0, n_actions))


# In[4]:


def SARSA(env, alpha, gamma, epsilon, Q, max_cylces):
    n_actions = len(env.actions())
    r = [None] * (2*max_cycles)
    s = [None] * (2*(max_cycles+1))
    a = [None] * (2*max_cycles)
    
    t=0
    s[t] = env.encoded_state()
    a[t] = policy_greedy(Q,s[t], epsilon, n_actions)
    obs = env.run(a[t])
    r[t] = obs['reward']
    s[t+1] = env.encoded_state() 
    if not s[t] in Q.keys():
        Q[s[t]] = [0]*n_actions
    try:
        while t < max_cycles:
            a[t+1] = policy_greedy(Q,s[t+1], epsilon, n_actions)
            obs = env.run(a[t])
            r[t+1] = obs['reward']
            s[t+2] = env.encoded_state()
            if not s[t+1] in Q.keys():
                Q[s[t+1]] = [0]*n_actions
            Q[s[t]][a[t]] = Q[s[t]][a[t]] + alpha * ( r[t] + (gamma * Q[s[t+1]][a[t+1]] ) - Q[s[t]][a[t]] )
            t += 1
            
    except KeyboardInterrupt:
        print("Press Ctrl-C to terminate while statement")
        pass
    return(s,a,r,Q)


# In[ ]:


Q = dict()
alpha = 1
gamma = 1
epsilon = 0.5
max_cycles = 250
sol = SARSA(env, alpha, gamma, epsilon, Q, max_cycles)


# In[ ]:


# Q=sol[3]
# env.reset()
# sol = SARSA(env, alpha, gamma, epsilon, Q, max_cycles)


# In[ ]:


#sol

file = open("sol.txt", "w")
str_sol = repr(sol)
file.write("sol = " + str_sol)

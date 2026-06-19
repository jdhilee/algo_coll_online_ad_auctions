############################## THE SETUP ##############################

import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import pandas as pd
import math
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# Defining the players' valuations and the reservation price
v1 = 4
v2 = 3
r = 0.5

# Setting the CTRs and the parameters
ctr_h = 5
ctr_l = 1
delta = 0.95 # This varies over the experiment
gamma = 0.5
alpha = 0.5
beta = 0.005 # Lower beta = higher exploration

# Initializing trials
z = 500
data_b1 = []
data_b2 = []
data_period = []
data_iteration = []
reps = range(1,5000)

# Defining a rounding function
def round_bid(number):
    return gamma * math.ceil(number/gamma)

############################## ACTIONS AND STATES ##############################

# Defining the action list
b1 = []
b2 = []

# Defining the action set
lowest_bid = r
if max(v1,v2) % gamma == 0:
    highest_bid = max(v1, v2) + gamma
elif max(v1,v2) % gamma != 0:
    highest_bid = round_bid(max(v1, v2))
b = []
_i = r
b.append(_i)

while _i <= highest_bid - gamma:
    _i = _i + gamma
    b.append(_i)

# Defining the state set
state_set = []
for pair in itertools.product(b, repeat = 2):
    state_set.append(pair)

# Randomizing the first bids
b1_current = random.choices(b)
b1.extend(b1_current)
b2_current = random.choices(b)
b2.extend(b2_current)

############################## Q-LEARNING ##############################

# Initializing Q1 and Q2 as empty matrices
Q1_inputs = state_set   # Inputs are 0.5, 1, ... max(v1,v2) + gamma
Q1_inputs = np.asarray(Q1_inputs).T
Q2_inputs = state_set   # Inputs are 0.5, 1, ... max(v1,v2) + gamma
Q2_inputs = np.asarray(Q2_inputs).T

Q1_outputs = np.array(np.zeros([len(state_set),len(b)]))
Q2_outputs = np.array(np.zeros([len(state_set),len(b)]))
Q1_outputs = Q1_outputs.astype(int)
Q2_outputs = Q2_outputs.astype(int)

Q1_agent = Sequential()
Q1_agent.add(Dense(round((len(b))*1.5*1.5), input_dim=1, activation='relu')) #b*1.5^2
Q1_agent.add(Dense(round((len(b))*1.5), activation='relu')) #b*1.5
Q1_agent.add(Dense(len(b), activation='relu'))
Q1_agent.compile(loss='binary_crossentropy', optimizer='adam')
Q1_agent.fit(Q1_inputs, Q1_outputs, epochs = 1)

Q2_agent = Sequential()
Q2_agent.add(Dense(round((len(b)**2)*1.5*1.5), input_dim=1, activation='relu'))
Q2_agent.add(Dense(round((len(b)**2)*1.5), activation='relu'))
Q2_agent.add(Dense(len(b)**2, activation='relu'))
Q2_agent.compile(loss='binary_crossentropy', optimizer='adam')
Q2_agent.fit(Q2_inputs, Q2_outputs, epochs = 1)

# Looping over each period
for _t in reps:
    e_t = np.exp(-beta * _t)
    
    # Choosing the actions
    rand = random.random()
    if rand <= e_t: # Explore
        b1_current = random.choices(b)[0]
        b2_current = random.choices(b)[0]
    else: # Exploit
        predictions_1 = Q1_agent.predict(np.asarray([b2_current]))
        b1_current = b[np.argmax(predictions_1)]

        predictions_2 = Q2_agent.predict(np.asarray([b1_current]))
        b2_current = b[np.argmax(predictions_2)]
    
    b1.append(b1_current)
    b2.append(b2_current)
    
    # Rewards from the actions
    if b1_current < b2_current:
        cs_1 = ctr_l * (v1 - r)
        cs_2 = ctr_h * (v2 - b1_current)
    elif b1_current == b2_current:
        draw = np.random.randint(1,3)
        if draw == 1:
            cs_1 = ctr_l * (v1 - r)
            cs_2 = ctr_h * (v2 - b1_current)
        elif draw == 2:
            cs_1 = ctr_h * (v1 - b2_current)
            cs_2 = ctr_l * (v2 - r)
    elif b1_current > b2_current:
        cs_1 = ctr_h * (v1 - b2_current)
        cs_2 = ctr_l * (v2 - r)
    
    # Updating the Q-values
    Q1_outputs[state_set.index((b1[-2], b2[-2])), b.index(b1[-1])] = (1 - alpha) * Q1_outputs[state_set.index((b1[-2], b2[-2])), b.index(b1[-1])] + alpha * (cs_1 + delta * np.max(Q1_outputs[state_set.index((b1[-1], b2[-1])), :]))
    Q1_agent.fit(Q1_inputs, Q1_outputs, epochs = 1)

    Q2_outputs[state_set.index((b1[-2], b2[-2])), b.index(b2[-1])] = (1 - alpha) * Q2_outputs[state_set.index((b1[-2], b2[-2])), b.index(b2[-1])] + alpha * (cs_2 + delta * np.max(Q2_outputs[state_set.index((b1[-1], b2[-1])), :]))
    Q2_agent.fit(Q2_inputs, Q2_outputs, epochs = 1)

period = list(range(len(b1)))

plt.plot(period, b1)
plt.show()
plt.plot(period, b2)
plt.show()
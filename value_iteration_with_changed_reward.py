#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

# MDP setup
states = ["Top", "Rolling", "Bottom"]
actions = ["Drive", "No Drive"]
discount_factor = 0.9  # Default discount factor
theta = 1e-6  # Convergence threshold

# Transition probabilities and rewards (Original)
transitions = {
    "Top": {
        "Drive": [("Top", 0.5, 2), ("Rolling", 0.5, 2)],
        "No Drive": [("Top", 0.5, 3), ("Rolling", 0.5, 1)],
    },
    "Rolling": {
        "Drive": [("Top", 0.3, 2), ("Rolling", 0.4, 1.5), ("Bottom", 0.3, 0.5)],
        "No Drive": [("Bottom", 1.0, 1)],
    },
    "Bottom": {
        "Drive": [("Top", 0.5, 2), ("Bottom", 0.5, 2)],
        "No Drive": [("Bottom", 1.0, 1)],
    },
}

# Function to run Value Iteration
def value_iteration(transitions, discount_factor):
    V = {s: 0 for s in states}  # Initialize values to 0

    iteration = 0
    while True:
        delta = 0
        new_V = V.copy()
        
        for s in states:
            max_value = max(
                sum(p * (r + discount_factor * V[s_next]) for s_next, p, r in transitions[s][a])
                for a in actions
            )
            delta = max(delta, abs(max_value - V[s]))
            new_V[s] = max_value
        
        V = new_V
        iteration += 1
        print(f"Iteration {iteration}: {V}")  # Print each iteration
        
        if delta < theta:
            break

    # Derive optimal policy
    policy = {}
    for s in states:
        policy[s] = max(actions, key=lambda a: sum(p * (r + discount_factor * V[s_next]) for s_next, p, r in transitions[s][a]))

    return V, policy

# Run value iteration on the original MDP
original_V, original_policy = value_iteration(transitions, discount_factor)
print("\nOriginal Optimal Values:", original_V)
print("Original Optimal Policy:", original_policy)


# In[6]:


# Import necessary library
import copy

# Create a deep copy of the original transition dictionary to avoid modifying the original data
modified_transitions = copy.deepcopy(transitions)

# Modify the reward for the "No Drive" action at the "Top" state
# - Originally, the rewards were 3 for staying at "Top" and 1 for transitioning to "Rolling"
# - Now, both rewards are set to 1, making the "No Drive" action less rewarding
modified_transitions["Top"]["No Drive"] = [("Top", 0.5, 1), ("Rolling", 0.5, 1)]  # Changed reward from +3 to +1

# Run the Value Iteration algorithm on the modified MDP
V_modified_reward, policy_modified_reward = value_iteration(modified_transitions, discount_factor)

# Display the results
print("\nModified Reward (Top, No Drive) - Optimal Values:", V_modified_reward)  # Shows updated state values
print("Modified Reward (Top, No Drive) - Optimal Policy:", policy_modified_reward)  # Displays new optimal policy


# In[ ]:





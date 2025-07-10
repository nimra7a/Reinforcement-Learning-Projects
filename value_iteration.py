#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# ===========================
# MDP Setup
# ===========================
states = ["Top", "Rolling", "Bottom"]  # List of states
actions = ["Drive", "No Drive"]  # List of possible actions
discount_factor = 0.9  # Discount factor for future rewards
theta = 1e-6  # Convergence threshold (small number to determine when to stop)

# Transition probabilities and rewards
# Format: state -> action -> [(next_state, probability, reward), ...]
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

# ===========================
# Value Iteration Algorithm
# ===========================

# Initialize the value function to 0 for all states
V = {s: 0 for s in states}
iteration = 0  # Track the number of iterations

print("Starting Value Iteration...\n")

# Iterate until convergence
while True:
    delta = 0  # Track the largest change in values for stopping condition
    new_V = V.copy()  # Create a copy to store new values
    iteration += 1
    print(f"Iteration {iteration}:")
    
    for s in states:
        # Compute the value of each action and select the best one
        max_value = max(
            sum(p * (r + discount_factor * V[s_next]) for s_next, p, r in transitions[s][a])
            for a in actions
        )
        
        # Update the max difference for convergence check
        delta = max(delta, abs(max_value - V[s]))
        
        # Update the state value
        new_V[s] = max_value
        print(f"  V({s}) = {max_value:.6f}")
    
    # Update the value function
    V = new_V
    print("-" * 40)

    # Stop when the change in value function is below the threshold
    if delta < theta:
        break

# ===========================
# Extract Optimal Policy
# ===========================

# Derive the optimal policy from the computed value function
policy = {}
for s in states:
    # Select the action that maximizes expected reward
    policy[s] = max(actions, key=lambda a: sum(p * (r + discount_factor * V[s_next]) for s_next, p, r in transitions[s][a]))

# ===========================
# Display Results
# ===========================

print("\nFinal Optimal Values:", V)
print("Final Optimal Policy:", policy)


# In[ ]:





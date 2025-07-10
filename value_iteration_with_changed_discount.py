#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# ===========================
# MDP Setup
# ===========================

states = ["Top", "Rolling", "Bottom"]  # Set of states
actions = ["Drive", "No Drive"]  # Available actions
discount_factor = 0.9  # Future rewards discount factor
theta = 1e-6  # Convergence threshold

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
# Value Iteration Function
# ===========================

def value_iteration(transitions, discount_factor):
    """
    Performs Value Iteration to find the optimal policy.

    Parameters:
    - transitions: MDP transition probabilities and rewards
    - discount_factor: Discount factor for future rewards

    Returns:
    - V: Optimal state values
    - policy: Optimal policy mapping states to actions
    """
    V = {s: 0 for s in states}  # Initialize state values to 0

    iteration = 0  # Track iterations
    while True:
        delta = 0  # Tracks the maximum change in value function
        new_V = V.copy()  # Copy the current values
        
        for s in states:
            # Compute the best possible value for this state
            max_value = max(
                sum(p * (r + discount_factor * V[s_next]) for s_next, p, r in transitions[s][a])
                for a in actions
            )
            delta = max(delta, abs(max_value - V[s]))  # Check max difference
            new_V[s] = max_value  # Update new value for state
        
        V = new_V  # Update values for the next iteration
        iteration += 1
        print(f"Iteration {iteration}: {V}")  # Print intermediate values
        
        if delta < theta:  # Stop if values have converged
            break

    # ===========================
    # Extract Optimal Policy
    # ===========================
    policy = {}
    for s in states:
        # Choose the action that maximizes expected value
        policy[s] = max(actions, key=lambda a: sum(p * (r + discount_factor * V[s_next]) for s_next, p, r in transitions[s][a]))

    return V, policy

# ===========================
# Run Value Iteration
# ===========================

original_V, original_policy = value_iteration(transitions, discount_factor)
print("\nOptimal Values:", original_V)
print("Optimal Policy:", original_policy)


# In[2]:


# Change the discount factor to 0.75
discount_factor_75 = 0.75

# Run value iteration with the modified discount factor
V_75, policy_75 = value_iteration(transitions, discount_factor_75)

# Display results
print("\nModified Discount Factor (0.75) - Optimal Values:", V_75)
print("Modified Discount Factor (0.75) - Optimal Policy:", policy_75)


# In[ ]:





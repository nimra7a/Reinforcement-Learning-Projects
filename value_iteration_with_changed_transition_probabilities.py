#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# Define MDP components
states = ["Top", "Rolling", "Bottom"]
actions = ["Drive", "No Drive"]
discount_factor = 0.9  # Discount factor (γ) for future rewards
theta = 1e-6  # Small threshold for convergence (stopping condition)

# Transition probabilities and rewards
# Format: {state: {action: [(next_state, probability, reward)]}}
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

# Function to perform Value Iteration
def value_iteration(transitions, discount_factor):
    """
    Performs Value Iteration algorithm to find the optimal policy and value function.

    Parameters:
    - transitions: Dictionary containing state transition probabilities and rewards.
    - discount_factor: Discount factor (γ) determining how future rewards are weighted.

    Returns:
    - V: Optimal state values.
    - policy: Optimal policy for each state.
    """
    
    # Step 1: Initialize value function for all states
    V = {s: 0 for s in states}  # Start with all values as 0

    iteration = 0  # Track iterations
    while True:
        delta = 0  # Track change in value function
        new_V = V.copy()  # Create a copy to update new values
        
        # Step 2: Update value function for each state
        for s in states:
            max_value = max(
                sum(p * (r + discount_factor * V[s_next]) for s_next, p, r in transitions[s][a])
                for a in actions
            )
            delta = max(delta, abs(max_value - V[s]))  # Track largest change
            new_V[s] = max_value  # Update value
        
        V = new_V  # Apply the updated values
        iteration += 1
        print(f"Iteration {iteration}: {V}")  # Print values at each iteration
        
        # Step 3: Check for convergence
        if delta < theta:
            break

    # Step 4: Derive the optimal policy from the optimal values
    policy = {}
    for s in states:
        policy[s] = max(actions, key=lambda a: sum(p * (r + discount_factor * V[s_next]) for s_next, p, r in transitions[s][a]))

    return V, policy

# Run Value Iteration on the defined MDP
optimal_V, optimal_policy = value_iteration(transitions, discount_factor)

# Display final results
print("\nOptimal State Values:", optimal_V)
print("Optimal Policy:", optimal_policy)


# In[2]:


import copy

# Create a deep copy of the transitions dictionary
modified_transitions = copy.deepcopy(transitions)

# Modify the "Rolling" state for the "Drive" action
modified_transitions["Rolling"]["Drive"] = [
    ("Top", 0.1, 2), ("Rolling", 0.4, 1.5), ("Bottom", 0.5, 0.5)  # Adjusted probabilities
]

# Run Value Iteration with the modified transition probabilities
V_modified_transition, policy_modified_transition = value_iteration(modified_transitions, discount_factor)

# Display results
print("\nModified Transition Probabilities - Optimal Values:", V_modified_transition)
print("Modified Transition Probabilities - Optimal Policy:", policy_modified_transition)


# In[ ]:





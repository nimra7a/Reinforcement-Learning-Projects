#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

# ===========================
# MDP Setup
# ===========================

states = ["Top", "Rolling", "Bottom"]  # List of states
actions = ["Drive", "No Drive"]  # List of possible actions
discount_factor = 0.9  # Discount factor for future rewards
theta = 1e-6  # Small threshold for convergence check

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
# Policy Evaluation Function
# ===========================

def policy_evaluation(policy):
    """
    Computes the value function for a given policy.
    
    Parameters:
    - policy: Dictionary mapping states to actions
    
    Returns:
    - V: Dictionary of state values under the given policy
    """
    V = {s: 0 for s in states}  # Initialize value function to zero
    while True:
        delta = 0  # Tracks the maximum change in value function
        new_V = V.copy()  # Create a copy to store new values
        
        for s in states:
            action = policy[s]  # Get the action for the current policy
            # Compute the expected value for this action
            new_V[s] = sum(p * (r + discount_factor * V[s_next]) for s_next, p, r in transitions[s][action])
            delta = max(delta, abs(new_V[s] - V[s]))  # Track the largest change
        
        V = new_V  # Update the value function
        
        if delta < theta:  # Stop if the value function has converged
            break

    return V

# ===========================
# Policy Improvement Function
# ===========================

def policy_improvement(V, policy):
    """
    Updates the policy using the computed value function.
    
    Parameters:
    - V: Dictionary of state values
    - policy: Current policy dictionary
    
    Returns:
    - new_policy: Improved policy
    - policy_stable: Boolean indicating if the policy has changed
    """
    policy_stable = True  # Assume policy is stable initially
    new_policy = {}

    for s in states:
        # Select the action that maximizes expected value
        best_action = max(actions, key=lambda a: sum(p * (r + discount_factor * V[s_next]) for s_next, p, r in transitions[s][a]))
        
        if policy[s] != best_action:  # Check if the policy has changed
            policy_stable = False
        
        new_policy[s] = best_action  # Store the improved action

    return new_policy, policy_stable

# ===========================
# Policy Iteration Function
# ===========================

def policy_iteration(policy):
    """
    Runs the Policy Iteration algorithm to find the optimal policy.
    
    Parameters:
    - policy: Initial policy dictionary
    
    Returns:
    - V: Optimal value function
    - policy: Optimal policy
    """
    iteration = 0  # Track number of iterations

    while True:
        iteration += 1
        print(f"Iteration {iteration}")
        
        V = policy_evaluation(policy)  # Compute value function for current policy
        print("  Values:", V)

        new_policy, policy_stable = policy_improvement(V, policy)  # Improve policy
        print("  Policy:", new_policy)

        if policy_stable:  # Stop if policy does not change
            break

        policy = new_policy  # Update policy for next iteration

    return V, policy

# ===========================
# Deterministic Initial Policy (Always No Drive)
# ===========================

# Start with a deterministic policy where all states choose "No Drive"
deterministic_policy = {s: "No Drive" for s in states}

print("\nStarting Policy Iteration with Deterministic (Never Drives) Policy...\n")
optimal_values, optimal_policy = policy_iteration(deterministic_policy)

# ===========================
# Display Results
# ===========================

print("\nFinal Optimal Values:", optimal_values)
print("Final Optimal Policy:", optimal_policy)


# In[ ]:





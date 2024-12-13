import numpy as np
import pandas as pd
import copy
import random
from dataclasses import dataclass
from typing import Tuple, Any
# import torch
# import torch.nn as nn
# import torch.optim as optim
from architectural_principles import State


def state_to_key(state) -> str:
    """Convert state to a hashable key."""
    return str(state.layout) + str(state.placed_rooms) + str(state.current_step)


def sample_initial_states(env):
    """Generate initial random states."""
    states = []
    env.reset()
    for room_name in env.required_rooms.keys():
        action = {
            "type": "add_room",
            "params": {
                "name": room_name,
                "room_type": env.required_rooms[room_name].room_type,
                "position": (
                    np.random.randint(0, env.grid_size[0] - 1),
                    np.random.randint(0, env.grid_size[1] - 1)
                ),
                "size": env.required_rooms[room_name].min_size,
            },
        }
        env.step(action)
        states.append(env._get_state())
    return states


@dataclass
class ValueIterationAgent:
    """
    Implementation of Value Iteration algorithm for architectural space planning.
    """
    action_space: list[dict]
    gamma: float = 0.2
    epsilon: float = 1e-6
    max_iterations: int = 5
    max_states: int = 100
    
    def __post_init__(self):
        self.V = {}
        self.policy = {}


    def expand_states(self, env, sampled_states: list[State], actions: list[dict]):
        # Expand states using valid actions
        expanded_states = []
        for state in sampled_states:
            env.set_state(state)
            for action in actions:
                next_state, reward, _, _ = env.step(action)
                # Add new state if it wasn't visited before
                key = state_to_key(next_state)
                if key not in self.policy and len(self.policy) < self.max_states:
                    self.policy[key] = (action, 0.0)
                    expanded_states.append(next_state)
        return expanded_states

    
    def train(self, env) -> dict:
        """
        Execute Value Iteration algorithm.
        
        Returns:
            dict containing training statistics
        """
        delta = 0
        sampled_states = sample_initial_states(env)
        for state in sampled_states:
            self.policy[state_to_key(state)] = (random.choice(self.action_space), 0.0)
        
        iteration = 0
        while iteration < self.max_iterations:
            print(f"Iteration: {iteration}")
            new_policy = copy.deepcopy(self.policy)
            expanded_states = self.expand_states(env, sampled_states, self.action_space)

            for state in sampled_states:
                max_next_value = 0
                next_action = random.choice(self.action_space)
                for action in self.action_space:
                    next_state, reward  = self._simulate_action(env, state, action)
                    key = state_to_key(next_state)
                    _, value = self.policy.get(key, (None, 0.0))
                    V_t = reward + self.gamma * value
                    max_next_value = max(max_next_value, V_t)
                    if max_next_value == V_t:
                        next_action = action
                
                    new_policy[state_to_key(state)] = (next_action, max_next_value)
            self.policy = new_policy
            sampled_states.extend(expanded_states)
            sampled_states = list({state_to_key(s): s for s in sampled_states}.values())

            # if delta < self.epsilon:
            #     break
            iteration += 1
        out = {
            'iterations': iteration,
            'final_delta': delta,
            'converged': delta < self.epsilon
        }
        return out
    

    def act(self, state: State) -> dict:
        """Return best action for given state based on learned policy."""
        if not self.action_space:
            raise ValueError("Action space is empty!")
        action, _ = self.policy.get(state_to_key(state), (random.choice(self.action_space), 0.0))
        return action
    

    def _simulate_action(self, env, state: State, action: dict) -> Tuple[State, float]:
        """Simulate action to get next state and reward."""
        # Create copy of environment to simulate action
        env_copy = env.copy()
        env_copy.set_state(state)
        next_state, reward, _, _ = env_copy.step(action)
        return next_state, reward



@dataclass
class PolicyIterationAgent:
    """
    Implementation of Policy Iteration algorithm for architectural space planning.
    """
    action_space: list[dict]
    gamma: float = 0.2
    epsilon: float = 1e-6
    max_iterations: int = 5
    max_states: int = 100
    
    def __post_init__(self):
        self.V = {}
        self.policy = {}

    def expand_states(self, env, sampled_states: list[State], actions: list[dict]):
        # Expand states using valid actions
        expanded_states = []
        for state in sampled_states:
            env.set_state(state)
            for action in actions:
                next_state, reward, _, _ = env.step(action)
                # Add new state if it wasn't visited before
                key = state_to_key(next_state)
                if key not in self.policy and len(self.policy) < self.max_states:
                    self.policy[key] = (action, 0.0)
                    expanded_states.append(next_state)
        return expanded_states
    
    def policy_evaluation(self, env, threshold=1e-6):
        """
        Perform Policy Evaluation: iteratively compute the value function for the current policy.
        """
        delta = float('inf')
        while delta > threshold:
            delta = 0
            for state_key, (action, _) in self.policy.items():
                state = state_from_key(state_key)
                next_state, reward = self._simulate_action(env, state, action)
                _, value = self.policy.get(state_to_key(next_state), (None, 0.0))
                V_t = reward + self.gamma * value
                self.V[state_key] = V_t
                delta = max(delta, abs(self.V[state_key] - V_t))
        return delta

    def policy_improvement(self, env):
        """
        Perform Policy Improvement: Update the policy based on the value function.
        """
        policy_stable = True
        new_policy = copy.deepcopy(self.policy)

        for state_key, (old_action, _) in self.policy.items():
            state = state_from_key(state_key)
            max_value = float('-inf')
            best_action = old_action

            # Find best action based on the current value function
            for action in self.action_space:
                next_state, reward = self._simulate_action(env, state, action)
                key = state_to_key(next_state)
                V_t = reward + self.gamma * self.V.get(key, 0.0)
                
                if V_t > max_value:
                    max_value = V_t
                    best_action = action
            
            # Update policy if action changed
            if best_action != old_action:
                policy_stable = False
            new_policy[state_key] = (best_action, max_value)

        self.policy = new_policy
        return policy_stable

    def train(self, env) -> dict:
        """
        Execute Policy Iteration algorithm.
        
        Returns:
            dict containing training statistics
        """
        delta = 0
        sampled_states = sample_initial_states(env)

        # Initialize policy with random actions
        for state in sampled_states:
            self.policy[state_to_key(state)] = (random.choice(self.action_space), 0.0)
        
        iteration = 0
        while iteration < self.max_iterations:
            print(f"Iteration: {iteration}")
            delta = self.policy_evaluation(env)
            
            if delta < self.epsilon:
                print("Policy Evaluation converged.")
                break
            
            policy_stable = self.policy_improvement(env)
            
            if policy_stable:
                print("Policy Iteration converged.")
                break

            iteration += 1

        out = {
            'iterations': iteration,
            'final_delta': delta,
            'converged': delta < self.epsilon
        }
        return out
    
    def act(self, state: State) -> dict:
        """Return best action for given state based on learned policy."""
        if not self.action_space:
            raise ValueError("Action space is empty!")
        action, _ = self.policy.get(state_to_key(state), (random.choice(self.action_space), 0.0))
        return action
    
    def _simulate_action(self, env, state: State, action: dict) -> Tuple[State, float]:
        """Simulate action to get next state and reward."""
        # Create copy of environment to simulate action
        env_copy = env.copy()
        env_copy.set_state(state)
        next_state, reward, _, _ = env_copy.step(action)
        return next_state, reward
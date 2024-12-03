import numpy as np
from dataclasses import dataclass
from typing import Tuple, Any
import torch
import torch.nn as nn
import torch.optim as optim
from architectural_principles import State

@dataclass
class ValueIterationAgent:
    """
    Implementation of Value Iteration algorithm for architectural space planning.
    """
    state_space_size: Tuple[int, ...]
    action_space: list[dict]
    gamma: float = 0.95
    theta: float = 1e-6
    max_iterations: int = 1000
    
    def __post_init__(self):
        self.V = {}
        self.policy = {}
    
    def train(self, env) -> dict:
        """
        Execute Value Iteration algorithm.
        
        Returns:
            dict containing training statistics
        """
        iteration = 0
        while iteration < self.max_iterations:
            delta = 0
            # Get initial state
            state = env.reset()
            state_key = self._get_state_key(state)
            
            if state_key not in self.V:
                self.V[state_key] = 0.0
            
            v = self.V[state_key]
            # Find maximum value over all actions
            max_value = float('-inf')
            best_action = None
            
            for action in self.action_space:
                # Simulate action to get next state and reward
                next_state, reward = self._simulate_action(env, state, action)
                next_state_key = self._get_state_key(next_state)
                
                if next_state_key not in self.V:
                    self.V[next_state_key] = 0.0
                
                value = reward + self.V[next_state_key]
                
                if value > max_value:
                    max_value = value
                    best_action = action
            
            self.V[state_key] = max_value
            self.policy[state_key] = best_action
            delta = max(delta, abs(v - self.V[state_key]))
            
            if delta < self.theta:
                break
            iteration += 1
        
        return {
            'iterations': iteration,
            'final_delta': delta,
            'converged': delta < self.theta
        }
    
    def act(self, state: State) -> dict:
        """Return best action for given state based on learned policy."""
        state_key = self._get_state_key(state)
        return self.policy.get(state_key, self.action_space[0])
    
    def _get_state_key(self, state: State) -> str:
        """Convert state array to hashable key."""
        return np.array(state).tobytes()
    
    def _simulate_action(self, env, state: State, action: dict) -> Tuple[State, float]:
        """Simulate action to get next state and reward."""
        # Create copy of environment to simulate action
        env_copy = env.copy()
        env_copy.set_state(state)
        next_state, reward, _, _ = env_copy.step(action)
        return next_state, reward

class PolicyIterationAgent:
    """
    Implementation of Policy Iteration algorithm for architectural space planning.
    """
    def __init__(self,
                 state_space_size: Tuple[int, ...],
                 action_space: list[dict],
                 gamma: float = 0.95,
                 theta: float = 1e-6,
                 max_iterations: int = 1000):
        self.state_space_size = state_space_size
        self.action_space = action_space
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        
        # Initialize value function and policy
        self.V = {}
        self.policy = {}
    
    def train(self, env) -> dict:
        """
        Execute Policy Iteration algorithm.
        
        Returns:
            Dict containing training statistics
        """
        iteration = 0
        policy_stable = False
        
        # Get initial state
        state = env.reset()
        state_key = self._get_state_key(state)
        
        # Initialize random policy
        if state_key not in self.policy:
            self.policy[state_key] = np.random.choice(self.action_space)
        
        while not policy_stable and iteration < self.max_iterations:
            # Policy Evaluation
            self._policy_evaluation(env)
            
            # Policy Improvement
            policy_stable = self._policy_improvement(env)
            iteration += 1
        
        return {
            'iterations': iteration,
            'policy_stable': policy_stable,
            'converged': policy_stable
        }
    
    def _policy_evaluation(self, env):
        """Evaluate current policy."""
        while True:
            delta = 0
            state = env.reset()
            state_key = self._get_state_key(state)
            
            if state_key not in self.V:
                self.V[state_key] = 0.0
            
            v = self.V[state_key]
            action = self.policy[state_key]
            
            # Simulate action to get next state and reward
            next_state, reward = self._simulate_action(env, state, action)
            next_state_key = self._get_state_key(next_state)
            
            if next_state_key not in self.V:
                self.V[next_state_key] = 0.0
            
            self.V[state_key] = reward + self.gamma * self.V[next_state_key]
            
            delta = max(delta, abs(v - self.V[state_key]))
            
            if delta < self.theta:
                break
    
    def _policy_improvement(self, env) -> bool:
        """
        Improve policy based on current value function.
        
        Returns:
            bool indicating whether policy is stable
        """
        policy_stable = True
        state = env.reset()
        state_key = self._get_state_key(state)
        
        old_action = self.policy[state_key]
        
        # Find best action based on current value function
        max_value = float('-inf')
        best_action = None
        
        for action in self.action_space:
            next_state, reward = self._simulate_action(env, state, action)
            next_state_key = self._get_state_key(next_state)
            
            if next_state_key not in self.V:
                self.V[next_state_key] = 0.0
            
            value = reward + self.gamma * self.V[next_state_key]
            
            if value > max_value:
                max_value = value
                best_action = action
        
        self.policy[state_key] = best_action
        
        if old_action != best_action:
            policy_stable = False
        
        return policy_stable
    
    def act(self, state) -> dict:
        """Return best action for given state based on learned policy."""
        state_key = self._get_state_key(state)
        return self.policy.get(state_key, self.action_space[0])
    
    def _get_state_key(self, state: np.ndarray) -> str:
        """Convert state array to hashable key."""
        return state.tobytes()
    
    def _simulate_action(self, env, state: np.ndarray, action: dict) -> Tuple[np.ndarray, float]:
        """Simulate action to get next state and reward."""
        env_copy = env.copy()
        env_copy.set_state(state)
        next_state, reward, _, _ = env_copy.step(action)
        return next_state, reward

class DeepRLAgent(nn.Module):
    """
    Deep RL agent for continuous state/action spaces.
    Uses actor-critic architecture with policy gradients.
    """
    def __init__(self,
                 state_dim: Tuple[int, int],  # Grid dimensions
                 action_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.state_dim = state_dim
        self.flattened_dim = state_dim[0] * state_dim[1]  # Total number of grid cells
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(self.flattened_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(self.flattened_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters())
        
    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Preprocess state grid into flattened tensor."""
        # Ensure state is the correct shape
        if state.shape != self.state_dim:
            raise ValueError(f"State shape {state.shape} does not match expected shape {self.state_dim}")
        
        # Flatten the grid
        flattened = state.reshape(-1)
        return torch.FloatTensor(flattened)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both actor and critic networks.
        
        Args:
            state: Current state tensor (already flattened)
            
        Returns:
            action: Continuous action values
            value: Estimated state value
        """
        action = self.actor(state)
        value = self.critic(state)
        return action, value
    
    def act(self, state: np.ndarray) -> np.ndarray:
        """Select action based on current policy."""
        # Preprocess state
        state_tensor = self._preprocess_state(state).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            action, _ = self.forward(state_tensor)
        return action.numpy()[0]
    
    def train_step(self, state: np.ndarray, action: np.ndarray, 
                  reward: float, next_state: np.ndarray, done: bool) -> dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            Dictionary containing training metrics
        """
        # Convert to tensors and preprocess
        state_tensor = self._preprocess_state(state).unsqueeze(0)
        next_state_tensor = self._preprocess_state(next_state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward])
        done_tensor = torch.FloatTensor([float(done)])
        
        # Get current action probabilities and value
        action_probs, value = self.forward(state_tensor)
        
        # Get next state value (for TD error)
        with torch.no_grad():
            _, next_value = self.forward(next_state_tensor)
            next_value = next_value * (1 - done_tensor)
        
        # Calculate TD error
        td_error = reward_tensor + 0.99 * next_value - value
        
        # Calculate losses
        actor_loss = -(action_probs * td_error.detach()).mean()
        critic_loss = td_error.pow(2).mean()
        
        # Combined loss
        total_loss = actor_loss + 0.5 * critic_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'total_loss': total_loss.item(),
            'td_error': td_error.mean().item()
        }

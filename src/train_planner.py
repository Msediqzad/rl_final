"""
Training script for the architectural space planning agent.
Uses curriculum learning to gradually increase layout complexity.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from environment import ArchitecturalEnvironment
from architectural_principles import (
    RoomType,
    Orientation,
    ArchitecturalConstraints,
    RoomRequirements
)
from visualization_principles import ArchitecturalVisualization

class DeepArchitecturalAgent(nn.Module):
    """Deep learning agent for architectural space planning."""
    
    def __init__(self,
                 grid_size: tuple,
                 n_room_types: int,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.grid_size = grid_size
        self.input_dim = grid_size[0] * grid_size[1]  # Flattened grid
        self.n_room_types = n_room_types
        
        # Neural network architecture
        self.shared = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Room type prediction
        self.room_type = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_room_types)
        )
        
        # Position prediction
        self.position = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # (x, y) coordinates
        )
        
        # Size prediction
        self.size = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # (width, height)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)  # Increased learning rate
        self.placed_rooms = set()  # Track which room types have been placed
        
    def forward(self, state: torch.Tensor):
        """Forward pass through the network."""
        x = state.view(-1, self.input_dim)
        shared_features = self.shared(x)
        
        room_type_logits = self.room_type(shared_features)
        position = torch.sigmoid(self.position(shared_features))  # Normalize to [0, 1]
        size = torch.sigmoid(self.size(shared_features))  # Normalize to [0, 1]
        
        return room_type_logits, position, size
    
    def act(self, state: np.ndarray, requirements: dict):
        """Select action based on current state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            room_type_logits, position, size = self(state_tensor)
            
            # Filter available room types to those not yet placed
            available_types = [rt for rt in requirements.keys() if rt not in self.placed_rooms]
            if not available_types:
                self.placed_rooms.clear()  # Reset if all rooms have been placed
                available_types = list(requirements.keys())
            
            # Sample room type from available types
            room_type = random.choice(available_types)
            self.placed_rooms.add(room_type)
            
            # Get position and size
            x = int(position[0, 0].item() * (self.grid_size[0] - 1))
            y = int(position[0, 1].item() * (self.grid_size[1] - 1))
            
            req = requirements[room_type]
            w = int(size[0, 0].item() * (req.max_size[0] - req.min_size[0] + 1)) + req.min_size[0]
            h = int(size[0, 1].item() * (req.max_size[1] - req.min_size[1] + 1)) + req.min_size[1]
            
            # Ensure room fits within grid
            x = min(x, self.grid_size[0] - w)
            y = min(y, self.grid_size[1] - h)
            
            return {
                'type': 'add_room',
                'params': {
                    'room_type': room_type,
                    'position': (x, y),
                    'size': (w, h)
                }
            }
    
    def reset(self):
        """Reset agent's internal state."""
        self.placed_rooms.clear()

def train_agent(epochs: int = 1500,  # Reduced total epochs
                batch_size: int = 32,
                curriculum_stages: int = 5,
                save_dir: str = 'models'):
    """
    Train the architectural planning agent using curriculum learning.
    
    Args:
        epochs: Number of training epochs (300 per stage)
        batch_size: Training batch size
        curriculum_stages: Number of curriculum difficulty stages
        save_dir: Directory to save models and training results
    """
    print("\nInitializing Training...")
    print(f"Total Epochs: {epochs}")
    print(f"Epochs per Stage: {epochs // curriculum_stages}")
    print(f"Batch Size: {batch_size}")
    print(f"Curriculum Stages: {curriculum_stages}")
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Initialize environment and agent
    grid_size = (12, 12)
    room_types = [
        RoomType.ENTRY,
        RoomType.LIVING,
        RoomType.DINING,
        RoomType.KITCHEN,
        RoomType.BEDROOM,
        RoomType.BATHROOM
    ]
    
    env = ArchitecturalEnvironment(
        grid_size=grid_size,
        building_orientation=Orientation.SOUTH,
        requirements=ArchitecturalConstraints.default_room_requirements()
    )
    
    agent = DeepArchitecturalAgent(
        grid_size=grid_size,
        n_room_types=len(room_types)
    )
    
    # Training metrics
    metrics = {
        'rewards': [],
        'space_efficiency': [],
        'adjacency_scores': [],
        'natural_light_scores': [],
        'privacy_scores': [],
        'room_counts': []
    }
    
    # Experience replay buffer
    replay_buffer = deque(maxlen=10000)
    
    # Curriculum learning
    for stage in range(curriculum_stages):
        min_rooms = 1 + stage
        max_rooms = 2 + stage
        print(f"\nCurriculum Stage {stage + 1}/{curriculum_stages}")
        print(f"Room Range: {min_rooms}-{max_rooms} rooms")
        
        for epoch in tqdm(range(epochs // curriculum_stages)):
            # Reset environment and agent
            state = env.reset()
            agent.reset()
            done = False
            episode_reward = 0
            
            # Collect experience
            episode_buffer = []
            while not done:
                # Select action
                action = agent.act(state, env.requirements)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                
                # Store transition
                episode_buffer.append((state, action, reward, next_state, done))
                
                if done or len(env.room_info) >= max_rooms:
                    break
                    
                state = next_state
            
            # Add episode to replay buffer
            replay_buffer.extend(episode_buffer)
            
            # Train on mini-batches
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                
                # Unpack batch
                states, actions, rewards, next_states, dones = zip(*batch)
                
                # Convert to tensors
                state_tensor = torch.FloatTensor(np.array(states))
                reward_tensor = torch.FloatTensor(rewards)
                
                # Update agent
                agent.optimizer.zero_grad()
                room_type_logits, position, size = agent(state_tensor)
                
                # Calculate loss (simplified)
                loss = -torch.mean(reward_tensor)  # Maximize reward
                loss.backward()
                agent.optimizer.step()
            
            # Record metrics
            metrics['rewards'].append(episode_reward)
            metrics['space_efficiency'].append(info['space_efficiency'])
            metrics['adjacency_scores'].append(info['adjacency_score'])
            metrics['natural_light_scores'].append(info['natural_light_score'])
            metrics['privacy_scores'].append(info['privacy_score'])
            metrics['room_counts'].append(len(env.room_info))
            
            # Save intermediate visualizations
            if epoch % 50 == 0:  # Increased visualization frequency
                visualizer = ArchitecturalVisualization()
                fig = visualizer.create_floor_plan(
                    env.grid,
                    env.room_info,
                    title=f"Stage {stage + 1}, Epoch {epoch}",
                    show_metrics=True
                )
                plt.savefig(save_dir / f'layout_stage{stage + 1}_epoch{epoch}.png')
                plt.close()
        
        # Save stage model
        torch.save(agent.state_dict(), save_dir / f'agent_stage{stage + 1}.pt')
        
        # Print stage metrics
        print(f"\nStage {stage + 1} Results:")
        print(f"Average Reward: {np.mean(metrics['rewards'][-100:]):.2f}")
        print(f"Average Space Efficiency: {np.mean(metrics['space_efficiency'][-100:]):.2f}")
        print(f"Average Adjacency Score: {np.mean(metrics['adjacency_scores'][-100:]):.2f}")
        print(f"Average Natural Light Score: {np.mean(metrics['natural_light_scores'][-100:]):.2f}")
        print(f"Average Privacy Score: {np.mean(metrics['privacy_scores'][-100:]):.2f}")
        print(f"Average Room Count: {np.mean(metrics['room_counts'][-100:]):.1f}")
    
    # Save final model and metrics
    torch.save(agent.state_dict(), save_dir / 'agent_final.pt')
    with open(save_dir / 'training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot training curves
    plot_training_curves(metrics, save_dir)
    
    print("\nTraining completed!")
    print(f"Model and metrics saved in: {save_dir}")
    return agent

def plot_training_curves(metrics: dict, save_dir: Path):
    """Plot and save training metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Metrics')
    
    # Plot rewards
    axes[0, 0].plot(metrics['rewards'])
    axes[0, 0].set_title('Rewards')
    axes[0, 0].set_xlabel('Episode')
    
    # Plot space efficiency
    axes[0, 1].plot(metrics['space_efficiency'])
    axes[0, 1].set_title('Space Efficiency')
    axes[0, 1].set_xlabel('Episode')
    
    # Plot adjacency scores
    axes[0, 2].plot(metrics['adjacency_scores'])
    axes[0, 2].set_title('Adjacency Scores')
    axes[0, 2].set_xlabel('Episode')
    
    # Plot natural light scores
    axes[1, 0].plot(metrics['natural_light_scores'])
    axes[1, 0].set_title('Natural Light Scores')
    axes[1, 0].set_xlabel('Episode')
    
    # Plot privacy scores
    axes[1, 1].plot(metrics['privacy_scores'])
    axes[1, 1].set_title('Privacy Scores')
    axes[1, 1].set_xlabel('Episode')
    
    # Plot room counts
    axes[1, 2].plot(metrics['room_counts'])
    axes[1, 2].set_title('Room Counts')
    axes[1, 2].set_xlabel('Episode')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png')
    plt.close()

if __name__ == "__main__":
    # Train the agent with reduced epochs
    trained_agent = train_agent(
        epochs=1500,  # Reduced from 5000 to 1500
        batch_size=32,
        curriculum_stages=5,
        save_dir='models'
    )

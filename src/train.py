import numpy as np
import matplotlib.pyplot as plt
from environment import ArchitecturalEnvironment
from agent import ValueIterationAgent, PolicyIterationAgent, DeepRLAgent
from architectural_principles import ArchitecturalConstraints
# import torch
from typing import List, Dict, Any
import json
from pathlib import Path
import time
from tqdm import tqdm

class ExperimentManager:
    """Manages training experiments and result tracking."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env = ArchitecturalEnvironment(
            grid_size=config['env']['grid_size'],
            max_steps=config['env']['max_steps'],
            required_rooms=config['env']['required_rooms']
        )
        
        # Initialize agents based on config
        self.agents = self._initialize_agents()
        
        # Create results directory
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize agents specified in config."""
        agents = {}
        
        if 'value_iteration' in self.config['algorithms']:
            agents['value_iteration'] = ValueIterationAgent(
                state_space_size=self.env.grid_size,
                action_space=self._get_action_space(),
                gamma=self.config['algorithms']['value_iteration']['gamma'],
                theta=self.config['algorithms']['value_iteration']['theta']
            )
            
        if 'policy_iteration' in self.config['algorithms']:
            agents['policy_iteration'] = PolicyIterationAgent(
                state_space_size=self.config['env']['grid_size'],
                action_space=self._get_action_space(),
                gamma=self.config['algorithms']['policy_iteration']['gamma'],
                theta=self.config['algorithms']['policy_iteration']['theta']
            )
            
        if 'deep_rl' in self.config['algorithms']:
            # For DeepRL, we need to determine the action dimension
            action_space = self._get_action_space()
            action_dim = len(action_space)  # Number of possible actions
            
            agents['deep_rl'] = DeepRLAgent(
                state_dim=self.config['env']['grid_size'],  # Pass grid dimensions directly
                action_dim=action_dim,
                hidden_dim=self.config['algorithms']['deep_rl']['hidden_dim']
            )
            
        return agents
    
    def _get_action_space(self) -> list[dict]:
        """Define the action space for the environment."""
        actions = []
        for room, requirements in self.env.required_rooms.items():
            actions.append({'type': 'remove_room', 'params': {'name': room}})
            for w in range(requirements.min_size[0], requirements.max_size[0] + 1):
                for h in range(requirements.min_size[1], requirements.max_size[1] + 1):
                    actions.append({'type': 'modify_room', 'params': {'name': room,'size': (w, h)}})
                    for x in range(self.env.grid_size[0] - w + 1):
                        for y in range(self.env.grid_size[1] - h + 1):
                            actions.append(
                                {
                                    'type': 'add_room',
                                    'params': {
                                        'name': room,
                                        'room_type': requirements.room_type,
                                        'position': (x, y),
                                        'size': (w, h)
                                    }
                                }
                            )
        
        return actions
    
    def run_experiments(self):
        """Run experiments for all configured agents."""
        results = {}
        
        for agent_name, agent in self.agents.items():
            print(f"\nTraining {agent_name}...")
            start_time = time.time()
            
            if isinstance(agent, (ValueIterationAgent, PolicyIterationAgent)):
                training_stats = self._train_planning_agent(agent)
            else:  # DeepRLAgent
                training_stats = self._train_deep_rl_agent(agent)
            
            training_time = time.time() - start_time
            
            results[agent_name] = {
                'training_stats': training_stats,
                'training_time': training_time,
                'final_evaluation': self._evaluate_agent(agent)
            }
            
            print(f"Training completed in {training_time:.2f} seconds")
            
        self._save_results(results)
        self._plot_results(results)
    
    def _train_planning_agent(self, agent) -> Dict:
        """Train Value Iteration or Policy Iteration agent."""
        return agent.train(self.env)
    
    def _train_deep_rl_agent(self, agent) -> Dict:
        """Train Deep RL agent."""
        config = self.config['algorithms']['deep_rl']
        episodes = config['episodes']
        max_steps = config['max_steps']
        
        rewards_history = []
        losses_history = []
        
        for episode in tqdm(range(episodes), desc="Training episodes"):
            state = self.env.reset()
            episode_reward = 0
            episode_losses = []
            
            for step in range(max_steps):
                # Select action
                action = agent.act(state)
                
                # Convert continuous action to discrete action space index
                action_idx = int((action[0] + 1) * len(self._get_action_space()) / 2)
                action_idx = max(0, min(action_idx, len(self._get_action_space()) - 1))
                discrete_action = self._get_action_space()[action_idx]
                
                # Execute action
                next_state, reward, done, _ = self.env.step(discrete_action)
                
                # Train agent
                losses = agent.train_step(state, action, reward, next_state, done)
                episode_losses.append(losses)
                episode_reward += reward
                
                if done:
                    break
                    
                state = next_state
            
            rewards_history.append(episode_reward)
            losses_history.append({
                k: np.mean([loss[k] for loss in episode_losses])
                for k in episode_losses[0].keys()
            })
            
            if episode % 10 == 0:
                mean_reward = np.mean(rewards_history[-10:])
                print(f"Episode {episode}/{episodes}, "
                      f"Average Reward: {mean_reward:.2f}")
        
        return {
            'rewards_history': rewards_history,
            'losses_history': losses_history,
            'final_average_reward': np.mean(rewards_history[-100:])
        }
    
    def _evaluate_agent(self, agent, num_episodes: int = 10) -> Dict:
        """Evaluate trained agent's performance."""
        rewards = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                if isinstance(agent, DeepRLAgent):
                    # Handle continuous actions for DeepRL
                    action = agent.act(state)
                    action_idx = int((action[0] + 1) * len(self._get_action_space()) / 2)
                    action_idx = max(0, min(action_idx, len(self._get_action_space()) - 1))
                    discrete_action = self._get_action_space()[action_idx]
                else:
                    # Handle discrete actions for planning agents
                    discrete_action = agent.act(state)
                
                state, reward, done, _ = self.env.step(discrete_action)
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards)
        }
    
    def _save_results(self, results: Dict):
        """Save experiment results to file."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_file = self.results_dir / f"results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to basic Python types."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def _plot_results(self, results: Dict):
        """Generate and save plots of experiment results."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Training time comparison
        plt.figure(figsize=(10, 6))
        agents = list(results.keys())
        times = [results[agent]['training_time'] for agent in agents]
        plt.bar(agents, times)
        plt.title('Training Time Comparison')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / f"training_times_{timestamp}.png")
        plt.close()
        
        # Performance comparison
        plt.figure(figsize=(10, 6))
        mean_rewards = [results[agent]['final_evaluation']['mean_reward'] 
                       for agent in agents]
        std_rewards = [results[agent]['final_evaluation']['std_reward'] 
                      for agent in agents]
        plt.bar(agents, mean_rewards, yerr=std_rewards, capsize=5)
        plt.title('Final Performance Comparison')
        plt.ylabel('Mean Reward')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / f"performance_{timestamp}.png")
        plt.close()
        
        # Learning curves for Deep RL
        if 'deep_rl' in results:
            plt.figure(figsize=(10, 6))
            rewards = results['deep_rl']['training_stats']['rewards_history']
            plt.plot(rewards)
            plt.title('Deep RL Learning Curve')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.tight_layout()
            plt.savefig(self.results_dir / f"learning_curve_{timestamp}.png")
            plt.close()

def main():
    """Main entry point for training."""
    config = {
        'env': {
            'grid_size': (50, 50),
            'max_steps': 1,
            'required_rooms': ArchitecturalConstraints.default_rooms()
        },
        'algorithms': {
            'value_iteration': {
                'gamma': 0.95,
                'theta': 1e-6
            },
            # 'policy_iteration': {
            #     'gamma': 0.95,
            #     'theta': 1e-6
            # },
            # 'deep_rl': {
            #     'hidden_dim': 256,
            #     'episodes': 1000,
            #     'max_steps': 100
            # }
        }
    }
    
    experiment_manager = ExperimentManager(config)
    experiment_manager.run_experiments()

if __name__ == "__main__":
    main()

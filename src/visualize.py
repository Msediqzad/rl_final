import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Tuple
import json
from pathlib import Path
import networkx as nx

class LayoutVisualizer:
    """Visualizes architectural layouts and training results."""
    
    def __init__(self):
        # Set style for consistent visualization
        plt.style.use('seaborn')
        self.colors = sns.color_palette("husl", 20)
    
    def plot_layout(self, 
                   grid: np.ndarray,
                   room_info: Dict,
                   title: str = "Building Layout",
                   save_path: str = None):
        """
        Plot the current building layout.
        
        Args:
            grid: 2D numpy array representing the layout
            room_info: Dictionary containing room metadata
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot grid
        ax.set_xlim(-0.5, grid.shape[1] - 0.5)
        ax.set_ylim(-0.5, grid.shape[0] - 0.5)
        
        # Draw rooms
        for room_id, info in room_info.items():
            x, y = info['position']
            w, h = info['size']
            
            color = self.colors[room_id % len(self.colors)]
            rect = Rectangle((y, x), h, w, 
                           facecolor=color, 
                           alpha=0.5,
                           edgecolor='black',
                           linewidth=2)
            ax.add_patch(rect)
            
            # Add room label
            ax.text(y + h/2, x + w/2, f"Room {room_id}",
                   horizontalalignment='center',
                   verticalalignment='center')
        
        # Customize plot
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel("Y coordinate")
        ax.set_ylabel("X coordinate")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def plot_adjacency_graph(self,
                           grid: np.ndarray,
                           room_info: Dict,
                           title: str = "Room Adjacency Graph",
                           save_path: str = None):
        """
        Plot graph showing room adjacencies.
        
        Args:
            grid: 2D numpy array representing the layout
            room_info: Dictionary containing room metadata
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        # Create adjacency graph
        G = nx.Graph()
        
        # Add nodes for each room
        for room_id in room_info.keys():
            G.add_node(f"Room {room_id}")
        
        # Add edges for adjacent rooms
        for room1_id in room_info.keys():
            for room2_id in room_info.keys():
                if room1_id < room2_id:  # Avoid duplicate edges
                    if self._are_rooms_adjacent(grid, room1_id, room2_id):
                        G.add_edge(f"Room {room1_id}", f"Room {room2_id}")
        
        # Plot graph
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(G)
        
        nx.draw(G, pos,
               with_labels=True,
               node_color='lightblue',
               node_size=2000,
               font_size=10,
               font_weight='bold',
               edge_color='gray',
               width=2)
        
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def plot_training_progress(self,
                             results_file: str,
                             save_dir: str = None):
        """
        Plot training progress from results file.
        
        Args:
            results_file: Path to JSON results file
            save_dir: Directory to save plots (optional)
        """
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        save_dir = Path(save_dir) if save_dir else None
        
        # Plot training times
        self._plot_training_times(results, save_dir)
        
        # Plot final performance
        self._plot_final_performance(results, save_dir)
        
        # Plot learning curves for Deep RL
        if 'deep_rl' in results:
            self._plot_learning_curves(results, save_dir)
    
    def _are_rooms_adjacent(self,
                          grid: np.ndarray,
                          room1_id: int,
                          room2_id: int) -> bool:
        """Check if two rooms are adjacent."""
        room1_mask = (grid == room1_id)
        room2_mask = (grid == room2_id)
        
        # Dilate room1 mask
        dilated = np.zeros_like(grid)
        dilated[:-1, :] |= room1_mask[1:, :]  # Up
        dilated[1:, :] |= room1_mask[:-1, :]  # Down
        dilated[:, :-1] |= room1_mask[:, 1:]  # Left
        dilated[:, 1:] |= room1_mask[:, :-1]  # Right
        
        # Check if dilated mask overlaps with room2
        return np.any(dilated & room2_mask)
    
    def _plot_training_times(self,
                           results: Dict,
                           save_dir: Path = None):
        """Plot training times comparison."""
        plt.figure(figsize=(10, 6))
        
        algorithms = list(results.keys())
        times = [results[alg]['training_time'] for alg in algorithms]
        
        plt.bar(algorithms, times)
        plt.title('Training Time Comparison')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        
        if save_dir:
            plt.savefig(save_dir / 'training_times.png',
                       bbox_inches='tight',
                       dpi=300)
            plt.close()
        else:
            plt.show()
    
    def _plot_final_performance(self,
                              results: Dict,
                              save_dir: Path = None):
        """Plot final performance comparison."""
        plt.figure(figsize=(10, 6))
        
        algorithms = list(results.keys())
        performance = [results[alg]['final_evaluation']['mean_reward']
                      for alg in algorithms]
        std = [results[alg]['final_evaluation']['std_reward']
               for alg in algorithms]
        
        plt.bar(algorithms, performance, yerr=std, capsize=5)
        plt.title('Final Performance Comparison')
        plt.ylabel('Mean Reward')
        plt.xticks(rotation=45)
        
        if save_dir:
            plt.savefig(save_dir / 'final_performance.png',
                       bbox_inches='tight',
                       dpi=300)
            plt.close()
        else:
            plt.show()
    
    def _plot_learning_curves(self,
                            results: Dict,
                            save_dir: Path = None):
        """Plot learning curves for Deep RL."""
        plt.figure(figsize=(10, 6))
        
        rewards = results['deep_rl']['training_stats']['rewards_history']
        episodes = range(1, len(rewards) + 1)
        
        plt.plot(episodes, rewards)
        plt.title('Deep RL Learning Curve')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        if save_dir:
            plt.savefig(save_dir / 'learning_curve.png',
                       bbox_inches='tight',
                       dpi=300)
            plt.close()
        else:
            plt.show()
    
    def plot_space_utilization(self,
                             grid: np.ndarray,
                             title: str = "Space Utilization Heatmap",
                             save_path: str = None):
        """
        Plot heatmap showing space utilization.
        
        Args:
            grid: 2D numpy array representing the layout
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(10, 10))
        
        # Create utilization heatmap
        utilization = (grid > 0).astype(float)
        
        sns.heatmap(utilization,
                   cmap='YlOrRd',
                   square=True,
                   cbar_kws={'label': 'Utilization'})
        
        plt.title(title)
        plt.xlabel("Y coordinate")
        plt.ylabel("X coordinate")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

def main():
    """Example usage of visualization tools."""
    # Create sample layout
    grid = np.zeros((10, 10))
    room_info = {
        1: {'position': (1, 1), 'size': (3, 4)},
        2: {'position': (5, 1), 'size': (2, 3)},
        3: {'position': (1, 6), 'size': (4, 3)}
    }
    
    # Fill grid based on room info
    for room_id, info in room_info.items():
        x, y = info['position']
        w, h = info['size']
        grid[x:x+w, y:y+h] = room_id
    
    # Create visualizer
    viz = LayoutVisualizer()
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    viz.plot_layout(grid, room_info,
                   save_path=str(results_dir / 'sample_layout.png'))
    viz.plot_adjacency_graph(grid, room_info,
                           save_path=str(results_dir / 'adjacency_graph.png'))
    viz.plot_space_utilization(grid,
                             save_path=str(results_dir / 'utilization.png'))

if __name__ == "__main__":
    main()

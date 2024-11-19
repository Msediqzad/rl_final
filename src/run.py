import numpy as np
from environment import ArchitecturalEnvironment
from agent import PolicyIterationAgent
import matplotlib.pyplot as plt
from visualize import LayoutVisualizer

def run_space_planning(requirements):
    """
    Run space planning with given requirements.
    
    Args:
        requirements: Dict containing:
            - min_rooms: Minimum number of rooms required
            - max_rooms: Maximum number of rooms allowed
            - room_sizes: List of preferred room sizes [(width, height), ...]
    """
    # Initialize environment
    env = ArchitecturalEnvironment(
        grid_size=(10, 10),
        max_steps=100,
        min_room_size=2,
        max_room_size=5
    )
    
    # Initialize agent
    action_space = []
    # Add room actions
    for w in range(env.min_room_size, env.max_room_size + 1):
        for h in range(env.min_room_size, env.max_room_size + 1):
            for x in range(env.grid_size[0] - w + 1):
                for y in range(env.grid_size[1] - h + 1):
                    action_space.append({
                        'type': 'add_room',
                        'params': {
                            'position': (x, y),
                            'size': (w, h)
                        }
                    })
    
    agent = PolicyIterationAgent(
        state_space_size=env.grid_size,
        action_space=action_space,
        gamma=0.95,
        theta=1e-6
    )
    
    # Train agent with requirements
    print("Training agent...")
    agent.train(env)
    
    # Generate layout
    print("\nGenerating layout...")
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        
        # Stop if we've met the room requirements
        if len(env.room_info) >= requirements['min_rooms']:
            break
    
    # Visualize result
    print("\nFinal layout:")
    visualizer = LayoutVisualizer()
    visualizer.plot_layout(env.grid, env.room_info)
    plt.show()
    
    print(f"\nTotal rooms: {len(env.room_info)}")
    print(f"Total reward: {total_reward:.2f}")
    
    # Print room details
    print("\nRoom details:")
    for room_id, info in env.room_info.items():
        print(f"Room {room_id}:")
        print(f"  Position: {info['position']}")
        print(f"  Size: {info['size']}")

if __name__ == "__main__":
    # Example requirements
    requirements = {
        'min_rooms': 3,
        'max_rooms': 5,
        'room_sizes': [
            (3, 4),  # Living room
            (3, 3),  # Bedroom
            (2, 2),  # Bathroom
        ]
    }
    
    run_space_planning(requirements)

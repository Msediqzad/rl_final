"""
Space planning tool that uses the trained model to generate architectural layouts.
"""

import torch
import numpy as np
from architectural_principles import (
    RoomType,
    Orientation,
    ArchitecturalConstraints
)
from visualization_principles import ArchitecturalVisualization
from environment import ArchitecturalEnvironment
from train_planner import DeepArchitecturalAgent
import matplotlib.pyplot as plt

def generate_layout(
    building_size: tuple = (12, 12),
    orientation: Orientation = Orientation.SOUTH,
    room_types: list = None,
    model_path: str = 'models/agent_final.pt'
) -> tuple:
    """
    Generate an architectural layout using the trained model.
    
    Args:
        building_size: (width, height) of the building
        orientation: Building orientation for natural light
        room_types: List of RoomType to include
        model_path: Path to the trained model
    
    Returns:
        (layout_grid, room_info, metrics)
    """
    # Initialize default room types if none provided
    if room_types is None:
        room_types = [
            RoomType.ENTRY,
            RoomType.LIVING,
            RoomType.DINING,
            RoomType.KITCHEN,
            RoomType.BEDROOM,
            RoomType.BATHROOM
        ]
    
    print("\nGenerating Architectural Layout...")
    print(f"Building Size: {building_size}")
    print(f"Orientation: {orientation.name}")
    print(f"Requested Rooms: {[rt.value for rt in room_types]}\n")
    
    # Initialize environment
    env = ArchitecturalEnvironment(
        grid_size=building_size,
        building_orientation=orientation,
        requirements=ArchitecturalConstraints.default_room_requirements()
    )
    
    # Load trained model
    agent = DeepArchitecturalAgent(
        grid_size=building_size,
        n_room_types=len(room_types)
    )
    agent.load_state_dict(torch.load(model_path))
    agent.eval()
    
    # Generate layout
    state = env.reset()
    done = False
    total_reward = 0
    visualizer = ArchitecturalVisualization()
    
    print("Placing rooms...")
    while not done:
        # Select action
        with torch.no_grad():
            action = agent.act(state, env.requirements)
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Show progress
            print(f"\nPlaced {len(env.room_info)} rooms:")
            print(f"Space Efficiency: {info['space_efficiency']:.2f}")
            print(f"Adjacency Score: {info['adjacency_score']:.2f}")
            print(f"Natural Light Score: {info['natural_light_score']:.2f}")
            print(f"Privacy Score: {info['privacy_score']:.2f}")
            
            # Visualize current state
            fig = visualizer.create_floor_plan(
                env.grid,
                env.room_info,
                title=f"Layout Progress - {len(env.room_info)} rooms",
                show_metrics=True
            )
            plt.show()
            
            if len(env.room_info) >= len(room_types):
                break
    
    print("\nFinal Layout Analysis:")
    
    # Show final visualizations
    print("\n1. Floor Plan and Metrics:")
    fig = visualizer.create_floor_plan(
        env.grid,
        env.room_info,
        title="Final Architectural Layout",
        show_metrics=True
    )
    plt.show()
    
    print("\n2. Architectural Relationships:")
    fig = visualizer.create_relationship_diagram(
        env.grid,
        env.room_info
    )
    plt.show()
    
    print("\n3. Analysis Views:")
    fig = visualizer.create_analysis_views(
        env.grid,
        env.room_info
    )
    plt.show()
    
    # Print final metrics
    print("\nFinal Metrics:")
    print(f"Total Rooms: {len(env.room_info)}")
    print(f"Space Efficiency: {info['space_efficiency']:.2f}")
    print(f"Adjacency Score: {info['adjacency_score']:.2f}")
    print(f"Natural Light Score: {info['natural_light_score']:.2f}")
    print(f"Privacy Score: {info['privacy_score']:.2f}")
    print(f"Total Reward: {total_reward:.2f}")
    
    # Print room details
    print("\nRoom Details:")
    for room_id, info in env.room_info.items():
        print(f"\nRoom {room_id} ({info['type'].value}):")
        print(f"  Position: {info['position']}")
        print(f"  Size: {info['size']}")
        
        # Get adjacent rooms
        adjacent_rooms = []
        for other_id, other_info in env.room_info.items():
            if other_id != room_id and ArchitecturalConstraints._are_rooms_adjacent(env.grid, room_id, other_id):
                adjacent_rooms.append(f"{other_info['type'].value} (Room {other_id})")
        if adjacent_rooms:
            print(f"  Adjacent to: {', '.join(adjacent_rooms)}")
    
    return env.grid, env.room_info, info

if __name__ == "__main__":
    # Generate a layout with default settings
    generate_layout()

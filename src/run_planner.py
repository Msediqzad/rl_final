"""
Interactive space planning tool that uses the trained model to generate architectural layouts.
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

def get_user_input():
    """Get layout requirements from user."""
    print("\n=== Architectural Space Planning System ===")
    print("This system will help you design an optimal layout based on your requirements.")
    
    # Get building size
    print("\n--- Step 1: Building Dimensions ---")
    print("Specify the size of your building (12x12 recommended)")
    while True:
        try:
            width = int(input("\nEnter building width (12 recommended): "))
            height = int(input("Enter building height (12 recommended): "))
            if width == 12 and height == 12:
                break
            confirm = input("\nWarning: Using non-standard dimensions may require retraining.\nContinue? (yes/no): ").lower()
            if confirm.startswith('y'):
                break
            print("Please use recommended dimensions (12x12) for best results.")
        except ValueError:
            print("! Please enter valid numbers.")
    
    # Get orientation
    print("\n--- Step 2: Building Orientation ---")
    print("Choose the main facade orientation:")
    print("1. North (best for southern hemisphere)")
    print("2. East (morning sun)")
    print("3. South (best for northern hemisphere)")
    print("4. West (evening sun)")
    while True:
        try:
            orientation_choice = int(input("\nEnter orientation number (1-4): "))
            if 1 <= orientation_choice <= 4:
                orientation = [
                    Orientation.NORTH,
                    Orientation.EAST,
                    Orientation.SOUTH,
                    Orientation.WEST
                ][orientation_choice - 1]
                break
            print("! Please choose a number between 1 and 4.")
        except ValueError:
            print("! Please enter a valid number.")
    
    # Get room types
    print("\n--- Step 3: Room Selection ---")
    print("Let's select the rooms for your layout.")
    
    available_rooms = {
        1: ("Entry", RoomType.ENTRY, "Main entrance and reception area"),
        2: ("Living Room", RoomType.LIVING, "Primary social and gathering space"),
        3: ("Dining Room", RoomType.DINING, "Formal eating area"),
        4: ("Kitchen", RoomType.KITCHEN, "Food preparation area"),
        5: ("Bedroom", RoomType.BEDROOM, "Private sleeping quarters"),
        6: ("Bathroom", RoomType.BATHROOM, "Sanitary facilities")
    }
    
    print("\nAvailable Rooms:")
    for num, (name, _, desc) in available_rooms.items():
        print(f"{num}. {name:<12} - {desc}")
    
    print("\nPre-selected Essential Rooms:")
    print("✓ Entry (Main entrance)")
    print("✓ Living Room (Primary social space)")
    
    print("\nPlease select additional rooms (recommended: 4 total rooms):")
    print("Examples of good combinations:")
    print("- Kitchen + Dining: For a complete living space")
    print("- Bedroom + Bathroom: For residential layout")
    print("- Kitchen + Bedroom: For a studio apartment")
    
    # Start with required rooms
    room_types = [RoomType.ENTRY, RoomType.LIVING]
    selected_rooms = {1, 2}  # Entry and Living are pre-selected
    
    while True:
        try:
            print("\nCurrently selected rooms:")
            for rt in room_types:
                print(f"✓ {rt.value}")
            
            print("\nAvailable choices:")
            for num, (name, _, _) in available_rooms.items():
                if num not in selected_rooms:
                    print(f"{num}. {name}")
            
            choice = int(input("\nSelect room number (or 0 to finish): "))
            
            if choice == 0:
                if len(room_types) >= 4:  # Recommended minimum
                    confirm = input("\nFinished selecting rooms? (yes/no): ").lower()
                    if confirm.startswith('y'):
                        break
                else:
                    print("! Please select at least 2 more rooms (4 total recommended).")
            elif choice in available_rooms:
                if choice in selected_rooms:
                    print(f"! {available_rooms[choice][0]} is already selected.")
                else:
                    room_types.append(available_rooms[choice][1])
                    selected_rooms.add(choice)
                    print(f"✓ Added {available_rooms[choice][0]}")
            else:
                print("! Invalid room number. Please choose from the available rooms.")
        except ValueError:
            print("! Please enter a valid number.")
    
    # Get specific requirements
    print("\n--- Step 4: Design Priority ---")
    print("Choose your main design priority:")
    priorities = [
        ("Maximize Natural Light", "Prioritize rooms with window access"),
        ("Prioritize Privacy", "Separate public and private spaces"),
        ("Optimize Space Efficiency", "Minimize wasted space"),
        ("Balance All Factors", "Equal consideration to all aspects")
    ]
    
    for i, (priority, desc) in enumerate(priorities, 1):
        print(f"{i}. {priority:<20} - {desc}")
    
    while True:
        try:
            requirement_choice = int(input("\nEnter priority number (1-4): "))
            if 1 <= requirement_choice <= 4:
                break
            print("! Please choose a number between 1 and 4.")
        except ValueError:
            print("! Please enter a valid number.")
    
    # Create requirements dictionary
    requirements = {
        'building_size': (width, height),
        'orientation': orientation,
        'room_types': room_types,
        'priority': requirement_choice
    }
    
    # Show summary
    print("\n=== Layout Requirements Summary ===")
    print(f"\nBuilding Dimensions: {width}m x {height}m")
    print(f"Main Facade Orientation: {orientation.name}")
    print("\nSelected Rooms:")
    for rt in room_types:
        print(f"- {rt.value}")
    print(f"\nDesign Priority: {priorities[requirement_choice-1][0]}")
    
    confirm = input("\nProceed with these requirements? (yes/no): ").lower()
    if not confirm.startswith('y'):
        print("\nRestarting room selection...")
        return get_user_input()
    
    return requirements

def generate_layout(requirements: dict = None) -> tuple:
    """
    Generate an architectural layout using the trained model.
    
    Args:
        requirements: Dictionary containing user requirements.
                     If None, will prompt for user input.
    
    Returns:
        (layout_grid, room_info, metrics)
    """
    if requirements is None:
        requirements = get_user_input()
    
    building_size = requirements['building_size']
    orientation = requirements['orientation']
    room_types = requirements['room_types']
    priority = requirements['priority']
    
    print("\n=== Generating Architectural Layout ===")
    print("Starting the layout generation process...")
    print(f"\nBuilding Size: {building_size[0]}m x {building_size[1]}m")
    print(f"Orientation: {orientation.name}")
    print("Rooms to place:", ", ".join(rt.value for rt in room_types))
    
    # Initialize environment with user requirements
    env = ArchitecturalEnvironment(
        grid_size=building_size,
        building_orientation=orientation,
        requirements={rt: ArchitecturalConstraints.default_room_requirements()[rt] 
                     for rt in room_types}
    )
    
    # Adjust rewards based on priority
    if priority == 1:  # Maximize natural light
        env.light_weight = 2.0
    elif priority == 2:  # Prioritize privacy
        env.privacy_weight = 2.0
    elif priority == 3:  # Optimize for space efficiency
        env.efficiency_weight = 2.0
    # priority 4 keeps default balanced weights
    
    # Load trained model
    print("\nLoading AI model...")
    try:
        # Load model metadata and state dict
        checkpoint = torch.load('models/agent_final.pt')
        trained_grid_size = checkpoint['grid_size']
        trained_n_room_types = checkpoint['n_room_types']
        
        if building_size != trained_grid_size:
            print(f"\nWarning: Using different grid size than training ({building_size} vs {trained_grid_size})")
            print("Creating new model with correct dimensions...")
            agent = DeepArchitecturalAgent.create_model_for_requirements(requirements)
        else:
            # Create model with same architecture as trained model
            agent = DeepArchitecturalAgent(
                grid_size=trained_grid_size,
                n_room_types=trained_n_room_types
            )
            # Load weights where possible
            agent.load_state_dict(checkpoint['state_dict'], strict=False)
            
    except Exception as e:
        print(f"\nError loading trained model: {e}")
        print("Creating new model...")
        agent = DeepArchitecturalAgent.create_model_for_requirements(requirements)
    
    agent.eval()
    
    # Generate layout
    state = env.reset()
    done = False
    total_reward = 0
    visualizer = ArchitecturalVisualization()
    
    print("\nPlacing rooms...")
    while not done:
        # Select action
        with torch.no_grad():
            action = agent.act(state, env.requirements)
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Show progress
            print(f"\nProgress: {len(env.room_info)}/{len(room_types)} rooms placed")
            print("\nCurrent Metrics:")
            print(f"Space Efficiency: {info['space_efficiency']:.2f}")
            print(f"Adjacency Score: {info['adjacency_score']:.2f}")
            print(f"Natural Light Score: {info['natural_light_score']:.2f}")
            print(f"Privacy Score: {info['privacy_score']:.2f}")
            
            # Visualize current state
            fig = visualizer.create_floor_plan(
                env.grid,
                env.room_info,
                title=f"Layout Progress - {len(env.room_info)}/{len(room_types)} rooms",
                show_metrics=True
            )
            plt.show()
            
            if len(env.room_info) >= len(room_types):
                break
    
    print("\n=== Final Layout Analysis ===")
    
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
    print("\n=== Final Metrics ===")
    print(f"Total Rooms Placed: {len(env.room_info)}")
    print(f"Space Efficiency: {info['space_efficiency']:.2f}")
    print(f"Adjacency Score: {info['adjacency_score']:.2f}")
    print(f"Natural Light Score: {info['natural_light_score']:.2f}")
    print(f"Privacy Score: {info['privacy_score']:.2f}")
    print(f"Overall Layout Score: {total_reward:.2f}")
    
    # Print room details
    print("\n=== Room Details ===")
    for room_id, info in env.room_info.items():
        print(f"\nRoom {room_id}: {info['type'].value}")
        print(f"  Size: {info['size'][0]}m x {info['size'][1]}m")
        print(f"  Position: ({info['position'][0]}, {info['position'][1]})")
        
        # Get adjacent rooms
        adjacent_rooms = []
        for other_id, other_info in env.room_info.items():
            if other_id != room_id and ArchitecturalConstraints._are_rooms_adjacent(env.grid, room_id, other_id):
                adjacent_rooms.append(f"{other_info['type'].value} (Room {other_id})")
        if adjacent_rooms:
            print(f"  Connected to: {', '.join(adjacent_rooms)}")
    
    print("\nLayout generation complete!")
    return env.grid, env.room_info, info

if __name__ == "__main__":
    # Generate a layout with user input
    generate_layout()

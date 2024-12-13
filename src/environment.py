from dataclasses import dataclass, field
from typing import Tuple, Any, NamedTuple
import copy
import json
import numpy as np
from architectural_principles import (
    ArchitecturalConstraints,
    RoomRequirements,
    State
)



@dataclass
class ArchitecturalEnvironment:
    """
    Environment for architectural space planning using RL.
    Implements a custom gym-like interface with architectural constraints.
    """
    grid_size: Tuple[int, int] = (10, 10),
    max_steps: int = 100,
    required_rooms: dict[str, RoomRequirements] = field(default_factory=ArchitecturalConstraints.default_rooms())

    def __post_init__(self):        
        # Initialize grid representation
        self.grid = np.zeros(self.grid_size).astype(int)
        self.current_step = 0
        self.placed_rooms = {}  # Store room metadata
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.grid = np.zeros(self.grid_size).astype(int)
        self.current_step = 0
        self.placed_rooms = {}
        return self._get_state()
    
    def step(self, action: str) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action in environment.
        
        Args:
            action: Dictionary containing action parameters
                   {
                    
                       'type': 'add_room'/'modify_room'/'remove_room',
                       'params': {
                           'name': room_name
                           'room_type': RoomType,
                           'position': (x, y),
                           'size': (width, height)
                       }
                   }
        
        Returns:
            state: Current state observation
            reward: Reward for current action
            done: Whether episode is finished
            metrics: Evaluation of design quality
        """
        self.current_step += 1
        if action['type'] == 'add_room':
            reward = self._add_room(action['params'])
        elif action['type'] == 'modify_room':
            reward = self._modify_room(action['params'])
        elif action['type'] == 'remove_room':
            reward = self._remove_room(action['params'])
        else:
            raise ValueError(f"Unknown action type: {action['type']}")
        
        # Check if episode is done
        done = self._is_done()
        state = self._get_state()
        # Get additional info
        metrics = {
            'space_efficiency': ArchitecturalConstraints.evaluate_space_efficiency(self.grid),
            'adjacency_score': ArchitecturalConstraints.evaluate_adjacency(state),
            'natural_light_score': ArchitecturalConstraints.evaluate_natural_light(state),
            'privacy_score': ArchitecturalConstraints.evaluate_privacy(state)
        }
        
        return state, reward, done, metrics
    
    def _get_state(self) -> State:
        """Return current state observation."""
        return State(self.grid.copy(), self.placed_rooms, self.current_step, self.required_rooms)
    
    def _add_room(self, params: dict) -> float:
        """Add a new room to the layout."""
        name = params['name']
        room_type = params['room_type']
        x, y = params['position']
        width, height = params['size']
        
        # Check if room placement is valid
        if not self._is_valid_room_placement(name, x, y, width, height):
            return -1.0
        
        # Add room to grid
        room_id = len(self.placed_rooms) + 1
        self.grid[x:x+width, y:y+height] = room_id
        self.placed_rooms[name] = {
            'id': room_id,
            'type': room_type,
            'position': (x, y),
            'size': (width, height)
        }
        
        return self._calculate_reward()
    
    def _modify_room(self, params: dict) -> float:
        """Modify existing room dimensions."""
        name = params['name']

        if name not in self.placed_rooms.keys():
            return -1.0
        
        room_id = self.placed_rooms[name]['id']
        new_width, new_height = params['size']
        x, y = self.placed_rooms[name]['position']
        
        # Check if new dimensions are valid
        if not self._is_valid_room_placement(name, x, y, new_width, new_height, exclude_room=room_id):
            return -1.0
            
        # Clear old room
        old_width, old_height = self.placed_rooms[name]['size']
        self.grid[x:x+old_width, y:y+old_height] = 0
        
        # Add modified room
        self.grid[x:x+new_width, y:y+new_height] = room_id
        self.placed_rooms[name]['size'] = (new_width, new_height)
        
        return self._calculate_reward() - 0.5 # penalty for modification
    
    def _remove_room(self, params: dict) -> float:
        """Remove existing room."""
        name = params['name']

        if name not in self.placed_rooms.keys():
            return -1.0
            
        room_id = self.placed_rooms[name]['id']
        x, y = self.placed_rooms[name]['position']
        width, height = self.placed_rooms[name]['size']
        room_type = self.placed_rooms[name]['type']
        
        # Clear room from grid
        self.grid[x:x+width, y:y+height] = 0
        del self.placed_rooms[name]
        
        return self._calculate_reward() - 0.25 # penalty for demolition
    
    def _is_valid_room_placement(
        self, 
        room_name: str,
        x: int, 
        y: int, 
        width: int, 
        height: int,
        exclude_room: int = None
    ) -> bool:
        """Check if room placement is valid."""
        # Get room requirements
        req = self.required_rooms[room_name]
        
        # Check boundaries
        if (x < 0 or y < 0 or 
            x + width > self.grid_size[0] or 
            y + height > self.grid_size[1]):
            return False
            
        # Check size constraints
        if (width < req.min_size[0] or height < req.min_size[1] or
            width > req.max_size[0] or height > req.max_size[1]):
            return False
            
        # Check overlap with existing rooms
        room_area = self.grid[x:x+width, y:y+height]
        if exclude_room is not None:
            room_area = np.where(room_area == exclude_room, 0, room_area)
        if np.any(room_area != 0):
            return False
        
        return True
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on architectural principles.
        
        Combines multiple evaluation metrics:
        1. Overall layout quality (from ArchitecturalConstraints)
        2. Progress toward placing all required rooms
        3. Penalties for invalid configurations
        """
        # Calculate base reward from architectural principles
        base_reward = ArchitecturalConstraints.evaluate_overall(
            State(layout=self.grid.copy(), placed_rooms=self.placed_rooms, current_step=self.current_step, required_rooms=self.required_rooms)
        )
        
        # Add bonus for completing required rooms
        completion_bonus = len(self.placed_rooms) / len(self.required_rooms)

        # Penalize time spent
        time_penalty = self.current_step / self.max_steps
        
        # Combine rewards
        return base_reward + completion_bonus - time_penalty
    
    def _is_done(self) -> bool:
        """Check if episode should end."""
        # End if maximum steps reached
        if self.current_step >= self.max_steps:
            return True
        
        # End if all required rooms are placed
        if self.placed_rooms == self.required_rooms:
            return True
        
        return False
    
    def copy(self):
        """Create a deep copy of the environment."""
        env_copy = ArchitecturalEnvironment(
            grid_size=self.grid_size,
            max_steps=self.max_steps,
            required_rooms=self.required_rooms
        )
        env_copy.grid = self.grid.copy()
        env_copy.current_step = self.current_step
        env_copy.placed_rooms = copy.deepcopy(self.placed_rooms)
        env_copy.placed_rooms = self.placed_rooms.copy()
        return env_copy
    
    def set_state(self, state: State):
        """Set the environment state."""
        self.grid = state.layout.copy()
        self.placed_rooms = state.placed_rooms
        self.current_step = state.current_step
        self.required_rooms = state.required_rooms
        self.placed_rooms = state.placed_rooms

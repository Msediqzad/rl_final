import numpy as np
from typing import Tuple, Dict, Any
import copy
from architectural_principles import (
    ArchitecturalConstraints,
    RoomType,
    Orientation,
    RoomRequirements
)

class ArchitecturalEnvironment:
    """
    Environment for architectural space planning using RL.
    Implements a custom gym-like interface with architectural constraints.
    """
    def __init__(self, 
                 grid_size: Tuple[int, int] = (10, 10),
                 max_steps: int = 100,
                 building_orientation: Orientation = Orientation.NORTH,
                 requirements: Dict[RoomType, RoomRequirements] = None):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.building_orientation = building_orientation
        self.requirements = requirements or ArchitecturalConstraints.default_room_requirements()
        
        # Initialize grid representation
        self.grid = np.zeros(grid_size)
        self.current_step = 0
        self.room_info = {}  # Store room metadata
        
        # Track required rooms
        self.required_rooms = set(self.requirements.keys())
        self.placed_rooms = set()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.grid = np.zeros(self.grid_size)
        self.current_step = 0
        self.room_info = {}
        self.placed_rooms = set()
        return self._get_state()
    
    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action in environment.
        
        Args:
            action: Dictionary containing action parameters
                   {
                       'type': 'add_room'/'modify_room'/'remove_room',
                       'params': {
                           'room_type': RoomType,
                           'position': (x, y),
                           'size': (width, height)
                       }
                   }
        
        Returns:
            state: Current state observation
            reward: Reward for current action
            done: Whether episode is finished
            info: Additional information
        """
        self.current_step += 1
        
        # Execute action based on type
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
        
        # Get additional info
        info = {
            'space_efficiency': ArchitecturalConstraints.evaluate_space_efficiency(self.grid),
            'adjacency_score': ArchitecturalConstraints.evaluate_adjacency(
                self.grid, self.room_info, self.requirements
            ),
            'natural_light_score': ArchitecturalConstraints.evaluate_natural_light(
                self.grid, self.room_info, self.requirements, self.building_orientation
            ),
            'privacy_score': ArchitecturalConstraints.evaluate_privacy(
                self.grid, self.room_info, self.requirements
            )
        }
        
        return self._get_state(), reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """Return current state observation."""
        return self.grid.copy()
    
    def _add_room(self, params: Dict) -> float:
        """Add a new room to the layout."""
        room_type = params['room_type']
        x, y = params['position']
        width, height = params['size']
        
        # Check if room placement is valid
        if not self._is_valid_room_placement(room_type, x, y, width, height):
            return -1.0
        
        # Add room to grid
        room_id = len(self.room_info) + 1
        self.grid[x:x+width, y:y+height] = room_id
        self.room_info[room_id] = {
            'type': room_type,
            'position': (x, y),
            'size': (width, height)
        }
        
        # Update placed rooms
        self.placed_rooms.add(room_type)
        
        return self._calculate_reward()
    
    def _modify_room(self, params: Dict) -> float:
        """Modify existing room dimensions."""
        room_id = params['room_id']
        new_width, new_height = params['size']
        
        if room_id not in self.room_info:
            return -1.0
            
        x, y = self.room_info[room_id]['position']
        room_type = self.room_info[room_id]['type']
        
        # Check if new dimensions are valid
        if not self._is_valid_room_placement(room_type, x, y, new_width, new_height, exclude_room=room_id):
            return -1.0
            
        # Clear old room
        old_width, old_height = self.room_info[room_id]['size']
        self.grid[x:x+old_width, y:y+old_height] = 0
        
        # Add modified room
        self.grid[x:x+new_width, y:y+new_height] = room_id
        self.room_info[room_id]['size'] = (new_width, new_height)
        
        return self._calculate_reward()
    
    def _remove_room(self, params: Dict) -> float:
        """Remove existing room."""
        room_id = params['room_id']
        
        if room_id not in self.room_info:
            return -1.0
            
        x, y = self.room_info[room_id]['position']
        width, height = self.room_info[room_id]['size']
        room_type = self.room_info[room_id]['type']
        
        # Clear room from grid
        self.grid[x:x+width, y:y+height] = 0
        self.placed_rooms.remove(room_type)
        del self.room_info[room_id]
        
        return self._calculate_reward()
    
    def _is_valid_room_placement(self, 
                               room_type: RoomType,
                               x: int, 
                               y: int, 
                               width: int, 
                               height: int,
                               exclude_room: int = None) -> bool:
        """Check if room placement is valid."""
        # Get room requirements
        req = self.requirements[room_type]
        
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
            self.grid,
            self.room_info,
            self.requirements,
            self.building_orientation
        )
        
        # Add bonus for completing required rooms
        completion_bonus = len(self.placed_rooms) / len(self.required_rooms)
        
        # Combine rewards
        return base_reward + completion_bonus
    
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
            building_orientation=self.building_orientation,
            requirements=self.requirements
        )
        env_copy.grid = self.grid.copy()
        env_copy.current_step = self.current_step
        env_copy.room_info = copy.deepcopy(self.room_info)
        env_copy.placed_rooms = self.placed_rooms.copy()
        return env_copy
    
    def set_state(self, state: np.ndarray):
        """Set the environment state."""
        if not isinstance(state, np.ndarray):
            raise ValueError("State must be a numpy array")
            
        self.grid = state.copy()
        # Reconstruct room_info from grid
        self.room_info = {}
        self.placed_rooms = set()
        unique_ids = np.unique(self.grid)
        unique_ids = unique_ids[unique_ids != 0]  # Exclude empty space
        
        for room_id in unique_ids:
            room_mask = (self.grid == room_id)
            x_coords, y_coords = np.where(room_mask)
            x, y = np.min(x_coords), np.min(y_coords)
            width = np.max(x_coords) - x + 1
            height = np.max(y_coords) - y + 1
            
            # Infer room type based on size
            room_type = None
            for rt, req in self.requirements.items():
                if (req.min_size[0] <= width <= req.max_size[0] and
                    req.min_size[1] <= height <= req.max_size[1]):
                    room_type = rt
                    break
            
            if room_type:
                self.room_info[int(room_id)] = {
                    'type': room_type,
                    'position': (x, y),
                    'size': (width, height)
                }
                self.placed_rooms.add(room_type)

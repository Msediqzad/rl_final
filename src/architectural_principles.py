"""
Architectural principles and constraints for space planning.
"""

from enum import Enum
from typing import Dict, List, Tuple, Set
import numpy as np

class RoomType(Enum):
    """Types of rooms with their specific requirements."""
    LIVING = "living"          # Needs natural light, central location
    BEDROOM = "bedroom"        # Needs natural light, privacy
    KITCHEN = "kitchen"        # Needs ventilation, adjacency to dining
    DINING = "dining"          # Needs natural light, adjacency to kitchen
    BATHROOM = "bathroom"      # Needs ventilation, privacy
    ENTRY = "entry"           # Needs central location, accessibility
    CORRIDOR = "corridor"      # Circulation space

class PrivacyLevel(Enum):
    """Privacy levels for different spaces."""
    PUBLIC = 1    # Entry, Living, Dining
    SEMI = 2      # Kitchen, Corridor
    PRIVATE = 3   # Bedroom, Bathroom

class Orientation(Enum):
    """Possible orientations for natural light."""
    NORTH = 0
    EAST = 90
    SOUTH = 180
    WEST = 270

class RoomRequirements:
    """Requirements and constraints for a specific room."""
    
    def __init__(self, 
                 room_type: RoomType,
                 min_size: Tuple[int, int],
                 max_size: Tuple[int, int],
                 privacy_level: PrivacyLevel,
                 needs_natural_light: bool = False,
                 needs_ventilation: bool = False,
                 adjacent_to: Set[RoomType] = None,
                 min_distance_from: Set[RoomType] = None):
        self.room_type = room_type
        self.min_size = min_size
        self.max_size = max_size
        self.privacy_level = privacy_level
        self.needs_natural_light = needs_natural_light
        self.needs_ventilation = needs_ventilation
        self.adjacent_to = adjacent_to or set()
        self.min_distance_from = min_distance_from or set()

class ArchitecturalConstraints:
    """Defines architectural constraints and evaluation metrics."""
    
    @staticmethod
    def default_room_requirements() -> Dict[RoomType, RoomRequirements]:
        """Default requirements for each room type."""
        return {
            RoomType.LIVING: RoomRequirements(
                room_type=RoomType.LIVING,
                min_size=(4, 4),
                max_size=(6, 6),
                privacy_level=PrivacyLevel.PUBLIC,
                needs_natural_light=True,
                adjacent_to={RoomType.DINING, RoomType.ENTRY}
            ),
            RoomType.BEDROOM: RoomRequirements(
                room_type=RoomType.BEDROOM,
                min_size=(3, 3),
                max_size=(4, 4),
                privacy_level=PrivacyLevel.PRIVATE,
                needs_natural_light=True,
                min_distance_from={RoomType.ENTRY, RoomType.LIVING}
            ),
            RoomType.KITCHEN: RoomRequirements(
                room_type=RoomType.KITCHEN,
                min_size=(3, 3),
                max_size=(4, 4),
                privacy_level=PrivacyLevel.SEMI,
                needs_ventilation=True,
                adjacent_to={RoomType.DINING}
            ),
            RoomType.DINING: RoomRequirements(
                room_type=RoomType.DINING,
                min_size=(3, 3),
                max_size=(4, 5),
                privacy_level=PrivacyLevel.PUBLIC,
                needs_natural_light=True,
                adjacent_to={RoomType.KITCHEN, RoomType.LIVING}
            ),
            RoomType.BATHROOM: RoomRequirements(
                room_type=RoomType.BATHROOM,
                min_size=(2, 2),
                max_size=(3, 3),
                privacy_level=PrivacyLevel.PRIVATE,
                needs_ventilation=True,
                min_distance_from={RoomType.KITCHEN, RoomType.DINING}
            ),
            RoomType.ENTRY: RoomRequirements(
                room_type=RoomType.ENTRY,
                min_size=(2, 2),
                max_size=(3, 3),
                privacy_level=PrivacyLevel.PUBLIC,
                adjacent_to={RoomType.LIVING}
            ),
            RoomType.CORRIDOR: RoomRequirements(
                room_type=RoomType.CORRIDOR,
                min_size=(1, 3),
                max_size=(2, 6),
                privacy_level=PrivacyLevel.SEMI
            )
        }

    @staticmethod
    def evaluate_space_efficiency(layout: np.ndarray) -> float:
        """
        Evaluate space efficiency of the layout.
        
        Considers:
        1. Ratio of usable space to total area
        2. Compactness of room arrangements
        3. Minimization of circulation space
        """
        total_area = layout.size
        used_area = np.count_nonzero(layout)
        return used_area / total_area if total_area > 0 else 0.0

    @staticmethod
    def evaluate_adjacency(layout: np.ndarray, 
                         room_info: Dict[int, Dict],
                         requirements: Dict[RoomType, RoomRequirements]) -> float:
        """
        Evaluate how well the layout satisfies adjacency requirements.
        """
        if not room_info:  # Handle empty layout
            return 0.0
            
        score = 0.0
        total_requirements = 0
        
        # Check each room's adjacency requirements
        for room_id, info in room_info.items():
            room_type = info['type']
            req = requirements[room_type]
            
            # Check required adjacencies
            for adj_type in req.adjacent_to:
                total_requirements += 1
                # Find if any room of the required type is adjacent
                for other_id, other_info in room_info.items():
                    if other_id != room_id and other_info['type'] == adj_type:
                        if ArchitecturalConstraints._are_rooms_adjacent(layout, room_id, other_id):
                            score += 1
                            break
            
            # Check minimum distance requirements
            for dist_type in req.min_distance_from:
                total_requirements += 1
                min_distance_met = True
                for other_id, other_info in room_info.items():
                    if other_id != room_id and other_info['type'] == dist_type:
                        if ArchitecturalConstraints._get_room_distance(info, other_info) < 2:
                            min_distance_met = False
                            break
                if min_distance_met:
                    score += 1
        
        return score / total_requirements if total_requirements > 0 else 0.0

    @staticmethod
    def evaluate_natural_light(layout: np.ndarray,
                             room_info: Dict[int, Dict],
                             requirements: Dict[RoomType, RoomRequirements],
                             building_orientation: Orientation) -> float:
        """
        Evaluate natural light access for rooms that require it.
        """
        if not room_info:  # Handle empty layout
            return 0.0
            
        score = 0.0
        rooms_needing_light = 0
        
        for room_id, info in room_info.items():
            room_type = info['type']
            req = requirements[room_type]
            
            if req.needs_natural_light:
                rooms_needing_light += 1
                # Check if room has access to exterior wall
                if ArchitecturalConstraints._has_exterior_access(layout, room_id):
                    score += 1
        
        return score / rooms_needing_light if rooms_needing_light > 0 else 1.0

    @staticmethod
    def evaluate_privacy(layout: np.ndarray,
                        room_info: Dict[int, Dict],
                        requirements: Dict[RoomType, RoomRequirements]) -> float:
        """
        Evaluate privacy zoning of the layout.
        """
        if not room_info:  # Handle empty layout
            return 0.0
            
        total_rooms = len(room_info)
        if total_rooms < 2:  # Need at least 2 rooms to evaluate privacy
            return 1.0
            
        score = 0.0
        total_pairs = total_rooms * (total_rooms - 1)  # Number of room pairs to evaluate
        
        for room1_id, info1 in room_info.items():
            room1_type = info1['type']
            req1 = requirements[room1_type]
            
            for room2_id, info2 in room_info.items():
                if room1_id != room2_id:
                    room2_type = info2['type']
                    req2 = requirements[room2_type]
                    
                    # Check if privacy levels are compatible with room placement
                    privacy_diff = abs(req1.privacy_level.value - req2.privacy_level.value)
                    distance = ArchitecturalConstraints._get_room_distance(info1, info2)
                    
                    if privacy_diff == 2:  # Public next to private
                        score += 1 if distance >= 2 else 0
                    elif privacy_diff == 1:  # Semi-private buffer
                        score += 1 if distance >= 1 else 0
                    else:  # Same privacy level
                        score += 1
        
        return score / total_pairs

    @staticmethod
    def _are_rooms_adjacent(layout: np.ndarray, room1_id: int, room2_id: int) -> bool:
        """Check if two rooms share a wall."""
        room1_mask = (layout == room1_id)
        room2_mask = (layout == room2_id)
        
        # Dilate room1 mask
        dilated = np.zeros_like(layout)
        dilated[:-1, :] |= room1_mask[1:, :]  # Up
        dilated[1:, :] |= room1_mask[:-1, :]  # Down
        dilated[:, :-1] |= room1_mask[:, 1:]  # Left
        dilated[:, 1:] |= room1_mask[:, :-1]  # Right
        
        return np.any(dilated & room2_mask)

    @staticmethod
    def _get_room_distance(room1_info: Dict, room2_info: Dict) -> float:
        """Calculate Manhattan distance between room centers."""
        x1, y1 = room1_info['position']
        w1, h1 = room1_info['size']
        x2, y2 = room2_info['position']
        w2, h2 = room2_info['size']
        
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        
        return abs(center1[0] - center2[0]) + abs(center1[1] - center2[1])

    @staticmethod
    def _has_exterior_access(layout: np.ndarray, room_id: int) -> bool:
        """Check if room has access to exterior wall."""
        room_mask = (layout == room_id)
        return (np.any(room_mask[0, :]) or  # Top wall
                np.any(room_mask[-1, :]) or  # Bottom wall
                np.any(room_mask[:, 0]) or   # Left wall
                np.any(room_mask[:, -1]))    # Right wall

    @staticmethod
    def evaluate_overall(layout: np.ndarray,
                        room_info: Dict[int, Dict],
                        requirements: Dict[RoomType, RoomRequirements],
                        building_orientation: Orientation) -> float:
        """
        Calculate overall layout score combining all metrics.
        
        Weights different aspects of the design based on importance:
        - Space efficiency: 30%
        - Adjacency requirements: 25%
        - Natural light: 20%
        - Privacy zoning: 25%
        """
        space_score = ArchitecturalConstraints.evaluate_space_efficiency(layout)
        adjacency_score = ArchitecturalConstraints.evaluate_adjacency(layout, room_info, requirements)
        light_score = ArchitecturalConstraints.evaluate_natural_light(layout, room_info, requirements, building_orientation)
        privacy_score = ArchitecturalConstraints.evaluate_privacy(layout, room_info, requirements)
        
        return (0.30 * space_score +
                0.25 * adjacency_score +
                0.20 * light_score +
                0.25 * privacy_score)

"""
Architectural principles and constraints for space planning.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple
import numpy as np


class RoomType(Enum):
    """Types of spaces."""
    LIVING = "living"
    BEDROOM = "bedroom"
    KITCHEN = "kitchen"
    DINING = "dining"
    BATHROOM = "bathroom"
    ENTRY = "entry"
    CORRIDOR = "corridor"


class PrivacyLevel(Enum):
    """Privacy levels for different spaces."""
    PUBLIC = 1
    SEMI = 2
    PRIVATE = 3


@dataclass
class RoomRequirements:
    """Requirements and constraints for a specific room."""
    room_type: RoomType
    min_size: Tuple[int, int]
    max_size: Tuple[int, int]
    privacy_level: PrivacyLevel
    needs_natural_light: bool = False
    needs_ventilation: bool = False
    adjacent_to: set[RoomType] = None
    min_distance_from: set[RoomType] = None


@dataclass
class State:
    layout: np.ndarray
    placed_rooms: dict[str, dict]
    current_step: int
    required_rooms: dict[str, RoomRequirements]


class ArchitecturalConstraints:
    """Architectural constraints and evaluation metrics."""
    
    @staticmethod
    def default_rooms() -> Dict[str, RoomRequirements]:
        """Default rooms."""
        return {
            "living": RoomRequirements(
                room_type=RoomType.LIVING,
                min_size=(4, 4),
                max_size=(6, 6),
                privacy_level=PrivacyLevel.PUBLIC,
                needs_natural_light=True,
                adjacent_to={RoomType.DINING, RoomType.ENTRY}
            ),
            "bedroom": RoomRequirements(
                room_type=RoomType.BEDROOM,
                min_size=(3, 3),
                max_size=(4, 4),
                privacy_level=PrivacyLevel.PRIVATE,
                needs_natural_light=True,
                min_distance_from={RoomType.ENTRY, RoomType.LIVING}
            ),
            "kitchen": RoomRequirements(
                room_type=RoomType.KITCHEN,
                min_size=(3, 3),
                max_size=(4, 4),
                privacy_level=PrivacyLevel.SEMI,
                needs_ventilation=True,
                adjacent_to={RoomType.DINING}
            ),
            "dining": RoomRequirements(
                room_type=RoomType.DINING,
                min_size=(3, 3),
                max_size=(4, 5),
                privacy_level=PrivacyLevel.PUBLIC,
                needs_natural_light=True,
                adjacent_to={RoomType.KITCHEN, RoomType.LIVING}
            ),
            "bathroom": RoomRequirements(
                room_type=RoomType.BATHROOM,
                min_size=(2, 2),
                max_size=(3, 3),
                privacy_level=PrivacyLevel.PRIVATE,
                needs_ventilation=True,
                min_distance_from={RoomType.KITCHEN, RoomType.DINING}
            ),
            "entry": RoomRequirements(
                room_type=RoomType.ENTRY,
                min_size=(2, 2),
                max_size=(3, 3),
                privacy_level=PrivacyLevel.PUBLIC,
                adjacent_to={RoomType.LIVING}
            ),
            "corridor": RoomRequirements(
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
    def evaluate_adjacency(state: State) -> float:
        """
        Evaluate how well the layout satisfies adjacency requirements.
        """
        if not state.placed_rooms:  # Handle empty layout
            return 0.0
            
        score = 0.0
        total_requirements = 0
        
        # Check each room's adjacency requirements
        for room, info in state.placed_rooms.items():
            room_id = info['id']
            req = state.required_rooms[room]
            
            # Check required adjacencies
            if req.adjacent_to is not None:
                for adj_type in req.adjacent_to:
                    total_requirements += 1
                    # Find if any room of the required type is adjacent
                    for other_id, other_info in state.placed_rooms.items():
                        if other_id != room_id and other_info['type'] == adj_type:
                            if ArchitecturalConstraints._are_rooms_adjacent(state.layout, room_id, other_id):
                                score += 1
                                break
            
            # Check minimum distance requirements
            if req.min_distance_from is not None:
                for dist_type in req.min_distance_from:
                    total_requirements += 1
                    min_distance_met = True
                    for other_id, other_info in state.placed_rooms.items():
                        if other_id != room_id and other_info['type'] == dist_type:
                            if ArchitecturalConstraints._get_room_distance(info, other_info) < 2:
                                min_distance_met = False
                                break
                    if min_distance_met:
                        score += 1
        
        return score / total_requirements if total_requirements > 0 else 0.0

    @staticmethod
    def evaluate_natural_light(state: State) -> float:
        """
        Evaluate natural light access for rooms that require it.
        """
        if not state.placed_rooms:  # Handle empty layout
            return 0.0
            
        score = 0.0
        rooms_needing_light = 0
        
        for room, info in state.placed_rooms.items():
            room_id = info['id']
            req = state.required_rooms[room]
            
            if req.needs_natural_light:
                rooms_needing_light += 1
                # Check if room has access to exterior wall
                if ArchitecturalConstraints._has_exterior_access(state.layout, room_id):
                    score += 1
        
        return score / rooms_needing_light if rooms_needing_light > 0 else 1.0

    @staticmethod
    def evaluate_privacy(state: State) -> float:
        """
        Evaluate privacy zoning of the layout.
        """
        if not state.placed_rooms:  # Handle empty layout
            return 0.0
            
        total_rooms = len(state.placed_rooms)
        if total_rooms < 2:  # Need at least 2 rooms to evaluate privacy
            return 1.0
            
        score = 0.0
        total_pairs = total_rooms * (total_rooms - 1)  # Number of room pairs to evaluate
        
        for room1, info1 in state.placed_rooms.items():
            room1_id = info1['id']
            req1 = state.required_rooms[room1]
            
            for room2, info2 in state.placed_rooms.items():
                if room1 != room2:
                    room2 = info2['id']
                    req2 = state.required_rooms[room2]
                    
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
    def evaluate_overall(state: State) -> float:
        """
        Calculate overall layout score combining all metrics.
        
        Weights different aspects of the design based on importance:
        - Space efficiency: 30%
        - Adjacency requirements: 25%
        - Natural light: 20%
        - Privacy zoning: 25%
        """
        space_score = ArchitecturalConstraints.evaluate_space_efficiency(state.layout)
        adjacency_score = ArchitecturalConstraints.evaluate_adjacency(state)
        light_score = ArchitecturalConstraints.evaluate_natural_light(state)
        privacy_score = ArchitecturalConstraints.evaluate_privacy(state)
        
        return (0.30 * space_score +
                0.25 * adjacency_score +
                0.20 * light_score +
                0.25 * privacy_score)

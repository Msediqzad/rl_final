"""
Architectural visualization principles and implementation.

This module defines the core visualization components for architectural space planning:
1. Floor Plan Visualization
2. Relationship Diagrams
3. Performance Metrics Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow, Circle
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional
import networkx as nx

class ArchitecturalVisualization:
    """
    Comprehensive visualization system for architectural layouts.
    Implements multiple visualization techniques based on architectural principles.
    """
    
    def __init__(self):
        """Initialize visualization settings and color schemes."""
        # Color schemes for different room types
        self.functional_colors = {
            'living': '#1f77b4',     # Blue for social spaces
            'bedroom': '#2ca02c',    # Green for private spaces
            'kitchen': '#ff7f0e',    # Orange for service spaces
            'dining': '#9467bd',     # Purple for social spaces
            'bathroom': '#8c564b',   # Brown for service spaces
            'entry': '#e377c2',      # Pink for circulation
            'corridor': '#7f7f7f'    # Gray for circulation
        }
        
        # Color schemes for metrics
        self.metric_colors = {
            'efficiency': 'YlOrRd',
            'privacy': 'RdYlBu',
            'natural_light': 'YlOrBr',
            'adjacency': 'PuBuGn'
        }
        
        # Set default style
        plt.style.use('default')
    
    def create_floor_plan(self,
                         grid: np.ndarray,
                         room_info: Dict,
                         title: str = "Architectural Floor Plan",
                         show_metrics: bool = True,
                         show_dimensions: bool = True) -> plt.Figure:
        """
        Create detailed floor plan visualization.
        
        Args:
            grid: Layout grid
            room_info: Room information dictionary
            title: Plot title
            show_metrics: Whether to show performance metrics
            show_dimensions: Whether to show room dimensions
        
        Returns:
            Matplotlib figure object
        """
        # Create main figure
        if show_metrics:
            fig = plt.figure(figsize=(20, 10))
            gs = fig.add_gridspec(2, 2)
            ax_main = fig.add_subplot(gs[:, 0])
            ax_metrics = [fig.add_subplot(gs[0, 1]),
                         fig.add_subplot(gs[1, 1])]
        else:
            fig = plt.figure(figsize=(10, 10))
            gs = fig.add_gridspec(1, 1)
            ax_main = fig.add_subplot(gs[0])
        
        # Plot main floor plan
        self._plot_floor_plan(ax_main, grid, room_info, show_dimensions)
        
        # Add metrics if requested
        if show_metrics:
            self._plot_metrics(ax_metrics, grid, room_info)
        
        # Customize layout
        plt.suptitle(title, fontsize=16, y=0.95)
        fig.tight_layout()
        
        return fig
    
    def create_relationship_diagram(self,
                                  grid: np.ndarray,
                                  room_info: Dict) -> plt.Figure:
        """
        Create diagram showing room relationships.
        
        Args:
            grid: Layout grid
            room_info: Room information dictionary
        
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(1, 3)
        
        # Create adjacency graph
        ax_adj = fig.add_subplot(gs[0])
        self._plot_adjacency_graph(ax_adj, grid, room_info)
        
        # Create privacy zones diagram
        ax_priv = fig.add_subplot(gs[1])
        self._plot_privacy_zones(ax_priv, grid, room_info)
        
        # Create circulation diagram
        ax_circ = fig.add_subplot(gs[2])
        self._plot_circulation(ax_circ, grid, room_info)
        
        fig.suptitle("Architectural Relationships", fontsize=16)
        fig.tight_layout()
        
        return fig
    
    def create_analysis_views(self,
                            grid: np.ndarray,
                            room_info: Dict) -> plt.Figure:
        """
        Create analytical views of the layout.
        
        Args:
            grid: Layout grid
            room_info: Room information dictionary
        
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(1, 3)
        
        # Space utilization heatmap
        ax_util = fig.add_subplot(gs[0])
        self._plot_space_utilization(ax_util, grid)
        
        # Natural light analysis
        ax_light = fig.add_subplot(gs[1])
        self._plot_natural_light(ax_light, grid, room_info)
        
        # Functional zoning
        ax_func = fig.add_subplot(gs[2])
        self._plot_functional_zones(ax_func, grid, room_info)
        
        fig.suptitle("Architectural Analysis", fontsize=16)
        fig.tight_layout()
        
        return fig
    
    def _plot_floor_plan(self,
                        ax: plt.Axes,
                        grid: np.ndarray,
                        room_info: Dict,
                        show_dimensions: bool):
        """Plot detailed floor plan."""
        # Set up axes
        ax.set_xlim(-0.5, grid.shape[1] - 0.5)
        ax.set_ylim(-0.5, grid.shape[0] - 0.5)
        
        # Draw rooms
        for room_id, info in room_info.items():
            x, y = info['position']
            w, h = info['size']
            room_type = info['type'].value if 'type' in info else f'Room {room_id}'
            
            # Create room rectangle
            color = self.functional_colors.get(room_type, '#7f7f7f')
            rect = Rectangle((y, x), h, w,
                           facecolor=color,
                           alpha=0.5,
                           edgecolor='black',
                           linewidth=2)
            ax.add_patch(rect)
            
            # Add room label and dimensions
            label = f"{room_type}"
            if show_dimensions:
                label += f"\n({w}x{h})"
            ax.text(y + h/2, x + w/2, label,
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=10,
                   fontweight='bold')
        
        # Customize plot
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlabel("Y coordinate")
        ax.set_ylabel("X coordinate")
    
    def _plot_metrics(self,
                     axes: List[plt.Axes],
                     grid: np.ndarray,
                     room_info: Dict):
        """Plot performance metrics."""
        # Space efficiency
        utilization = (grid > 0).astype(float)
        im = axes[0].imshow(utilization, cmap=self.metric_colors['efficiency'])
        axes[0].set_title("Space Utilization")
        plt.colorbar(im, ax=axes[0])
        
        # Privacy zones
        privacy_map = np.zeros_like(grid, dtype=float)
        for room_id, info in room_info.items():
            x, y = info['position']
            w, h = info['size']
            room_type = info['type'].value
            if room_type in ['bedroom', 'bathroom']:
                privacy_map[x:x+w, y:y+h] = 1.0
            elif room_type in ['kitchen', 'corridor']:
                privacy_map[x:x+w, y:y+h] = 0.5
        
        im = axes[1].imshow(privacy_map, cmap=self.metric_colors['privacy'])
        axes[1].set_title("Privacy Zones")
        plt.colorbar(im, ax=axes[1])
    
    def _plot_adjacency_graph(self,
                            ax: plt.Axes,
                            grid: np.ndarray,
                            room_info: Dict):
        """Plot room adjacency graph."""
        G = nx.Graph()
        
        # Add nodes
        for room_id, info in room_info.items():
            room_type = info['type'].value
            G.add_node(room_type)
        
        # Add edges for adjacent rooms
        for room1_id, info1 in room_info.items():
            room1_type = info1['type'].value
            for room2_id, info2 in room_info.items():
                if room1_id < room2_id:
                    room2_type = info2['type'].value
                    if self._are_rooms_adjacent(grid, room1_id, room2_id):
                        G.add_edge(room1_type, room2_type)
        
        # Draw graph
        pos = nx.spring_layout(G)
        nx.draw(G, pos,
               ax=ax,
               with_labels=True,
               node_color=[self.functional_colors.get(node, '#7f7f7f') for node in G.nodes()],
               node_size=2000,
               font_size=8,
               font_weight='bold',
               edge_color='gray',
               width=2)
        
        ax.set_title("Room Adjacencies")
    
    def _plot_privacy_zones(self,
                          ax: plt.Axes,
                          grid: np.ndarray,
                          room_info: Dict):
        """Plot privacy zone diagram."""
        ax.set_xlim(-0.5, grid.shape[1] - 0.5)
        ax.set_ylim(-0.5, grid.shape[0] - 0.5)
        
        # Draw zones
        for room_id, info in room_info.items():
            x, y = info['position']
            w, h = info['size']
            room_type = info['type'].value
            
            # Determine privacy level
            if room_type in ['bedroom', 'bathroom']:
                color = 'red'
                alpha = 0.3
                label = 'Private'
            elif room_type in ['kitchen', 'corridor']:
                color = 'yellow'
                alpha = 0.3
                label = 'Semi-private'
            else:
                color = 'green'
                alpha = 0.3
                label = 'Public'
            
            rect = Rectangle((y, x), h, w,
                           facecolor=color,
                           alpha=alpha,
                           edgecolor='black',
                           linewidth=1)
            ax.add_patch(rect)
            ax.text(y + h/2, x + w/2, label,
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=8)
        
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title("Privacy Zones")
    
    def _plot_circulation(self,
                         ax: plt.Axes,
                         grid: np.ndarray,
                         room_info: Dict):
        """Plot circulation diagram."""
        ax.set_xlim(-0.5, grid.shape[1] - 0.5)
        ax.set_ylim(-0.5, grid.shape[0] - 0.5)
        
        # Draw rooms
        for room_id, info in room_info.items():
            x, y = info['position']
            w, h = info['size']
            room_type = info['type'].value
            
            rect = Rectangle((y, x), h, w,
                           facecolor='lightgray',
                           alpha=0.3,
                           edgecolor='black',
                           linewidth=1)
            ax.add_patch(rect)
            
            # Draw circulation paths
            if room_type == 'entry':
                # Draw arrows from entry to adjacent rooms
                for other_id, other_info in room_info.items():
                    if other_id != room_id and self._are_rooms_adjacent(grid, room_id, other_id):
                        self._draw_circulation_arrow(ax, info, other_info)
        
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title("Circulation Paths")
    
    def _plot_space_utilization(self,
                              ax: plt.Axes,
                              grid: np.ndarray):
        """Plot space utilization heatmap."""
        utilization = (grid > 0).astype(float)
        im = ax.imshow(utilization, cmap=self.metric_colors['efficiency'])
        ax.set_title("Space Utilization")
        plt.colorbar(im, ax=ax)
    
    def _plot_natural_light(self,
                           ax: plt.Axes,
                           grid: np.ndarray,
                           room_info: Dict):
        """Plot natural light analysis."""
        light_map = np.zeros_like(grid, dtype=float)
        
        # Calculate light levels based on distance from exterior walls
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                dist_to_edge = min(i, j, grid.shape[0]-1-i, grid.shape[1]-1-j)
                light_map[i, j] = 1 - (dist_to_edge / max(grid.shape))
        
        im = ax.imshow(light_map, cmap=self.metric_colors['natural_light'])
        ax.set_title("Natural Light Analysis")
        plt.colorbar(im, ax=ax)
    
    def _plot_functional_zones(self,
                             ax: plt.Axes,
                             grid: np.ndarray,
                             room_info: Dict):
        """Plot functional zoning diagram."""
        ax.set_xlim(-0.5, grid.shape[1] - 0.5)
        ax.set_ylim(-0.5, grid.shape[0] - 0.5)
        
        # Group rooms by function
        social_spaces = []
        private_spaces = []
        service_spaces = []
        
        for room_id, info in room_info.items():
            room_type = info['type'].value
            if room_type in ['living', 'dining']:
                social_spaces.append(info)
            elif room_type in ['bedroom', 'bathroom']:
                private_spaces.append(info)
            else:
                service_spaces.append(info)
        
        # Draw functional zones
        self._draw_zone(ax, social_spaces, 'Social', 'blue', 0.2)
        self._draw_zone(ax, private_spaces, 'Private', 'red', 0.2)
        self._draw_zone(ax, service_spaces, 'Service', 'green', 0.2)
        
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title("Functional Zones")
    
    def _are_rooms_adjacent(self,
                           grid: np.ndarray,
                           room1_id: int,
                           room2_id: int) -> bool:
        """Check if two rooms share a wall."""
        room1_mask = (grid == room1_id)
        room2_mask = (grid == room2_id)
        
        # Dilate room1 mask
        dilated = np.zeros_like(grid)
        dilated[:-1, :] |= room1_mask[1:, :]  # Up
        dilated[1:, :] |= room1_mask[:-1, :]  # Down
        dilated[:, :-1] |= room1_mask[:, 1:]  # Left
        dilated[:, 1:] |= room1_mask[:, :-1]  # Right
        
        return np.any(dilated & room2_mask)
    
    def _draw_circulation_arrow(self,
                              ax: plt.Axes,
                              from_room: Dict,
                              to_room: Dict):
        """Draw circulation arrow between rooms."""
        x1, y1 = from_room['position']
        w1, h1 = from_room['size']
        x2, y2 = to_room['position']
        w2, h2 = to_room['size']
        
        # Calculate center points
        start = (y1 + h1/2, x1 + w1/2)
        end = (y2 + h2/2, x2 + w2/2)
        
        # Draw arrow
        ax.arrow(start[0], start[1],
                end[0] - start[0], end[1] - start[1],
                head_width=0.3,
                head_length=0.5,
                fc='black',
                ec='black',
                alpha=0.5)
    
    def _draw_zone(self,
                   ax: plt.Axes,
                   rooms: List[Dict],
                   label: str,
                   color: str,
                   alpha: float):
        """Draw functional zone encompassing multiple rooms."""
        if not rooms:
            return
            
        # Find zone boundaries
        x_coords = []
        y_coords = []
        for room in rooms:
            x, y = room['position']
            w, h = room['size']
            x_coords.extend([x, x+w])
            y_coords.extend([y, y+h])
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Draw zone
        rect = Rectangle((min_y-0.5, min_x-0.5),
                        max_y-min_y+1, max_x-min_x+1,
                        facecolor=color,
                        alpha=alpha,
                        edgecolor=color,
                        linewidth=2,
                        linestyle='--')
        ax.add_patch(rect)
        
        # Add label
        ax.text(min_y + (max_y-min_y)/2, min_x + (max_x-min_x)/2,
                label,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=12,
                fontweight='bold',
                color=color)

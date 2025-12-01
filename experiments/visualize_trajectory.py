#!/usr/bin/env python3
"""
Trajectory Visualizer for Experiment Results
- Shows warehouse map with shelves and walls
- Plots robot trajectories with collision points
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path


# Warehouse map definition
WAREHOUSE = {
    'walls': [
        # (x, y, width, height) - outer walls
        {'pos': (-12, -12), 'size': (0.2, 24)},   # Left wall
        {'pos': (12, -12), 'size': (0.2, 24)},    # Right wall
        {'pos': (-12, 12), 'size': (24, 0.2)},    # Top wall
        {'pos': (-12, -12), 'size': (24, 0.2)},   # Bottom wall
    ],
    'shelves': [
        # (center_x, center_y, width, height) - shelves
        {'center': (-11, -0.5), 'size': (2, 9)},  # Shelf 1
        {'center': (-6, -0.5), 'size': (2, 9)},   # Shelf 2
        {'center': (-1, -0.5), 'size': (2, 9)},   # Shelf 3
        {'center': (4, -0.5), 'size': (2, 9)},    # Shelf 4
        {'center': (11, -0.5), 'size': (2, 9)},   # Shelf 5
    ]
}


def draw_warehouse(ax):
    """Draw warehouse map with walls and shelves"""
    # Draw ground (light gray)
    ground = patches.Rectangle((-12, -12), 24, 24,
                               facecolor='#f5f5f5', edgecolor='none', zorder=0)
    ax.add_patch(ground)

    # Draw walls (dark gray)
    for wall in WAREHOUSE['walls']:
        rect = patches.Rectangle(wall['pos'], wall['size'][0], wall['size'][1],
                                facecolor='#404040', edgecolor='#303030',
                                linewidth=1, zorder=1)
        ax.add_patch(rect)

    # Draw shelves (brown)
    for shelf in WAREHOUSE['shelves']:
        cx, cy = shelf['center']
        w, h = shelf['size']
        rect = patches.Rectangle((cx - w/2, cy - h/2), w, h,
                                 facecolor='#8B4513', edgecolor='#5D2E0C',
                                 linewidth=1.5, zorder=2, alpha=0.9)
        ax.add_patch(rect)

    # Draw aisle labels
    aisles = [
        {'x': -8.5, 'label': 'Aisle 1'},
        {'x': -3.5, 'label': 'Aisle 2'},
        {'x': 1.5, 'label': 'Aisle 3'},
        {'x': 7.5, 'label': 'Aisle 4'},
    ]
    for aisle in aisles:
        ax.text(aisle['x'], 0, aisle['label'], fontsize=7,
                ha='center', va='center', alpha=0.4, rotation=90)


def load_episode(filepath):
    """Load episode data from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_trajectory(episode_data, ax, label=None, color='blue', show_humans=True):
    """Plot single trajectory"""
    traj = np.array(episode_data['trajectory'])
    if len(traj) == 0:
        return

    # Plot trajectory
    ax.plot(traj[:, 0], traj[:, 1], '-', color=color, alpha=0.8, linewidth=2.5, label=label, zorder=10)

    # Start point (green circle)
    ax.scatter(traj[0, 0], traj[0, 1], c='lime', s=120, marker='o', zorder=15,
               edgecolors='darkgreen', linewidths=2)

    # End point (blue square if success, red if failed)
    end_color = '#00BFFF' if episode_data.get('success', False) else 'red'
    ax.scatter(traj[-1, 0], traj[-1, 1], c=end_color, s=100, marker='s', zorder=15,
               edgecolors='black', linewidths=1.5)

    # Goal point (gold star)
    if 'goal' in episode_data:
        goal = episode_data['goal']
        ax.scatter(goal[0], goal[1], c='gold', s=250, marker='*', zorder=16,
                   edgecolors='darkorange', linewidths=1.5)

    # Plot human positions (small gray/orange dots with time-based alpha)
    if show_humans and 'human_positions' in episode_data and episode_data['human_positions']:
        human_positions = episode_data['human_positions']
        total_frames = len(human_positions)

        # Sample frames to avoid overcrowding (every 5th frame)
        sample_rate = max(1, total_frames // 50)

        for frame_idx in range(0, total_frames, sample_rate):
            if frame_idx < len(human_positions):
                humans = human_positions[frame_idx]
                # Alpha based on time progression (start=0.3, end=0.8)
                alpha = 0.2 + 0.5 * (frame_idx / total_frames)
                for h in humans:
                    ax.scatter(h[0], h[1], c='orange', s=25, marker='o',
                              alpha=alpha, zorder=8, edgecolors='none')

        # Plot final human positions more prominently
        if len(human_positions) > 0:
            final_humans = human_positions[-1]
            for h in final_humans:
                ax.scatter(h[0], h[1], c='darkorange', s=80, marker='o',
                          zorder=9, edgecolors='red', linewidths=1, alpha=0.9)

    # Collision points (red X marks)
    if 'collisions' in episode_data and episode_data['collisions']:
        collision_positions = set()  # Deduplicate nearby collisions
        for collision in episode_data['collisions']:
            robot_pos = tuple(collision['robot_pos'])
            # Round to avoid plotting too many overlapping points
            rounded = (round(robot_pos[0], 1), round(robot_pos[1], 1))
            if rounded not in collision_positions:
                collision_positions.add(rounded)
                ax.scatter(robot_pos[0], robot_pos[1], c='red', s=200, marker='X',
                          zorder=20, edgecolors='darkred', linewidths=2)


def visualize_experiment(results_dir):
    """Visualize all episodes in experiment"""
    results_dir = Path(results_dir)
    episodes_dir = results_dir / 'episodes'

    if not episodes_dir.exists():
        print(f"No episodes directory found in {results_dir}")
        return

    # Load all episodes
    episode_files = sorted(episodes_dir.glob('episode_*.json'))
    if not episode_files:
        print(f"No episode files found in {episodes_dir}")
        return

    print(f"Found {len(episode_files)} episodes")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))

    # Draw warehouse map first
    draw_warehouse(ax)

    # Color map for different episodes
    colors = plt.cm.tab10(np.linspace(0, 1, len(episode_files)))

    for i, ep_file in enumerate(episode_files):
        ep_data = load_episode(ep_file)
        status = "SUCCESS" if ep_data.get('success', False) else "FAIL"
        label = f"Ep{i+1} ({status})"
        plot_trajectory(ep_data, ax, label=label, color=colors[i])

    # Draw spawn positions
    top_positions = [(-10, 10), (-5, 10), (0, 10), (5, 10), (10, 10)]
    bottom_positions = [(-10, -10), (-5, -10), (0, -10), (5, -10), (10, -10)]

    for pos in top_positions:
        ax.scatter(pos[0], pos[1], c='lightgreen', s=80, marker='^', alpha=0.6, zorder=5)
    for pos in bottom_positions:
        ax.scatter(pos[0], pos[1], c='lightblue', s=80, marker='v', alpha=0.6, zorder=5)

    # Load config
    config_file = results_dir / 'config.json'
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        title = f"Module: {config.get('module', 'unknown')} | Scenario: {config.get('scenario', 'unknown')}"
    else:
        title = f"Experiment: {results_dir.name}"

    # Load summary and build title
    summary_file = results_dir / 'summary.json'
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        title += f"\nSR: {summary.get('success_rate', 0):.1f}%"
        title += f" | AvgTime: {summary.get('avg_time', 0):.1f}s"
        title += f" | AvgVel: {summary.get('avg_velocity', 0):.2f}m/s"
        if 'total_collisions' in summary:
            title += f" | Collisions: {summary.get('total_collisions', 0)}"
        # Add new metrics
        if 'avg_angular_velocity' in summary:
            title += f"\nAngVel: {summary.get('avg_angular_velocity', 0):.2f}rad/s"
        if 'avg_path_length' in summary:
            title += f" | PathLen: {summary.get('avg_path_length', 0):.1f}m"
        if 'avg_itr' in summary:
            title += f" | ITR: {summary.get('avg_itr', 0):.3f}"
        # Planning time (for fair comparison)
        if 'avg_planning_time' in summary:
            title += f" | PlanTime: {summary.get('avg_planning_time', 0):.2f}s"
        min_dist = summary.get('min_human_dist', float('inf'))
        if min_dist != float('inf'):
            title += f" | MinHumanDist: {min_dist:.2f}m"

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_xlim(-14, 14)
    ax.set_ylim(-14, 14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # Save figure
    output_path = results_dir / 'trajectory_plot.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")

    plt.close()


def main():
    if len(sys.argv) < 2:
        # Find latest experiment
        results_base = Path('/home/pyongjoo/Desktop/newstart/environment/experiments/results')
        if not results_base.exists():
            print("No results directory found")
            return

        exp_dirs = sorted([d for d in results_base.iterdir() if d.is_dir()],
                         key=lambda x: x.stat().st_mtime, reverse=True)
        if not exp_dirs:
            print("No experiments found")
            return

        results_dir = exp_dirs[0]
        print(f"Using latest experiment: {results_dir}")
    else:
        results_dir = Path(sys.argv[1])

    visualize_experiment(results_dir)


if __name__ == '__main__':
    main()

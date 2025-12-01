#!/usr/bin/env python3
"""
Comparison Visualizer - DWA vs CIGP 비교 시각화
"""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_episodes(module_dir):
    """에피소드 데이터 로드"""
    episodes = []
    ep_dir = Path(module_dir) / 'episodes'
    if ep_dir.exists():
        for ep_file in sorted(ep_dir.glob('episode_*.json')):
            with open(ep_file) as f:
                episodes.append(json.load(f))
    return episodes


def plot_trajectories(ax, episodes, color, label):
    """궤적 플롯"""
    for i, ep in enumerate(episodes):
        traj = np.array(ep['trajectory'])
        if len(traj) == 0:
            continue
        
        ep_label = f'{label} Ep{i+1}' if i == 0 else None
        ax.plot(traj[:, 0], traj[:, 1], '-', color=color, alpha=0.7, linewidth=2, label=ep_label)
        
        # 시작점
        ax.scatter(traj[0, 0], traj[0, 1], c='green', s=80, marker='o', zorder=5, edgecolors='black')
        
        # 끝점
        end_color = 'blue' if ep.get('success') else 'red'
        ax.scatter(traj[-1, 0], traj[-1, 1], c=end_color, s=80, marker='s', zorder=5, edgecolors='black')


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_comparison.py <results_dir>")
        return

    results_dir = Path(sys.argv[1])
    
    # 데이터 로드
    dwa_dir = results_dir / 'dwa'
    cigp_dir = results_dir / 'cigp'
    
    dwa_episodes = load_episodes(dwa_dir)
    cigp_episodes = load_episodes(cigp_dir)
    
    # Summary 로드
    with open(dwa_dir / 'summary.json') as f:
        dwa_summary = json.load(f)
    with open(cigp_dir / 'summary.json') as f:
        cigp_summary = json.load(f)
    
    # Figure 생성
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # DWA Trajectories
    ax1 = axes[0]
    plot_trajectories(ax1, dwa_episodes, 'blue', 'DWA')
    ax1.set_title(f'DWA\nSR: {dwa_summary["success_rate"]:.1f}% | Time: {dwa_summary["avg_time"]:.1f}s | Vel: {dwa_summary["avg_velocity"]:.2f}m/s')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_xlim(-15, 15)
    ax1.set_ylim(-15, 15)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=10, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=-10, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=10, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=-10, color='gray', linestyle='--', alpha=0.5)
    
    # CIGP Trajectories
    ax2 = axes[1]
    plot_trajectories(ax2, cigp_episodes, 'red', 'CIGP')
    ax2.set_title(f'CIGP\nSR: {cigp_summary["success_rate"]:.1f}% | Time: {cigp_summary["avg_time"]:.1f}s | Vel: {cigp_summary["avg_velocity"]:.2f}m/s')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_xlim(-15, 15)
    ax2.set_ylim(-15, 15)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=10, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=-10, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=10, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=-10, color='gray', linestyle='--', alpha=0.5)
    
    # Goal 표시
    ax1.scatter(-10, 10, c='gold', s=200, marker='*', zorder=6, edgecolors='black', label='Goal')
    ax2.scatter(-10, 10, c='gold', s=200, marker='*', zorder=6, edgecolors='black', label='Goal')
    
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    
    plt.suptitle('DWA vs CIGP Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 저장
    output_path = results_dir / 'comparison_plot.png'
    plt.savefig(output_path, dpi=150)
    print(f'Saved: {output_path}')
    
    # Bar chart comparison
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Success Rate (%)', 'Avg Time (s)', 'Avg Velocity (m/s)']
    dwa_values = [dwa_summary['success_rate'], dwa_summary['avg_time'], dwa_summary['avg_velocity']]
    cigp_values = [cigp_summary['success_rate'], cigp_summary['avg_time'], cigp_summary['avg_velocity']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, dwa_values, width, label='DWA', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, cigp_values, width, label='CIGP', color='red', alpha=0.7)
    
    ax.set_ylabel('Value')
    ax.set_title('DWA vs CIGP Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # 값 표시
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    
    output_path2 = results_dir / 'metrics_comparison.png'
    plt.savefig(output_path2, dpi=150)
    print(f'Saved: {output_path2}')
    
    plt.close('all')


if __name__ == '__main__':
    main()

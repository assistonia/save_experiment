#!/usr/bin/env python3
"""Collect pedestrian trajectory data from pedsim and save to file"""
import rospy
import json
import time
from pedsim_msgs.msg import AgentStates

class TrajectoryCollector:
    def __init__(self, duration=30):
        self.trajectories = {}  # {agent_id: [(timestamp, x, y), ...]}
        self.duration = duration
        self.start_time = None

    def callback(self, msg):
        if self.start_time is None:
            self.start_time = time.time()

        current_time = time.time() - self.start_time

        for agent in msg.agent_states:
            aid = agent.id
            x = agent.pose.position.x
            y = agent.pose.position.y

            if aid not in self.trajectories:
                self.trajectories[aid] = []
            self.trajectories[aid].append((current_time, x, y))

    def run(self):
        rospy.init_node('trajectory_collector', anonymous=True)
        rospy.Subscriber('/pedsim_simulator/simulated_agents', AgentStates, self.callback)

        print(f"Collecting trajectories for {self.duration} seconds...")
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.start_time and (time.time() - self.start_time) > self.duration:
                break
            rate.sleep()

        # Save to file
        output = {
            'metadata': {
                'duration': self.duration,
                'num_agents': len(self.trajectories),
                'map_bounds': {'x': [-12, 12], 'y': [-12, 12]}
            },
            'trajectories': {str(k): v for k, v in self.trajectories.items()}
        }

        with open('/environment/trajectory_data.json', 'w') as f:
            json.dump(output, f)

        print(f"Saved {len(self.trajectories)} agent trajectories to /environment/trajectory_data.json")

        # Also save in ETH/UCY format for SingularTrajectory
        with open('/environment/trajectory_eth_format.txt', 'w') as f:
            for aid, traj in self.trajectories.items():
                for t, x, y in traj:
                    frame = int(t * 10)  # 10fps -> frame number
                    f.write(f"{frame}\t{aid}\t{x:.4f}\t{y:.4f}\n")
        print("Saved ETH format to /environment/trajectory_eth_format.txt")

if __name__ == '__main__':
    import sys
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    collector = TrajectoryCollector(duration=duration)
    collector.run()

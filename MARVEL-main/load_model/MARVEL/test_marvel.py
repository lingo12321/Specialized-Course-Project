import os
import sys
import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
os.chdir(project_root)

from utils.model import PolicyNet
from utils.env_test import Env
from utils.agent import Agent
from utils.node_manager import NodeManager
from utils.utils import *
from test_parameter import *

def test_single_episode(map_index=0, n_agents=4, fov=120, sensor_range=10, greedy=True):
    device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    print(f"Using device: {device}")
    
    global_network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN).to(device)
    
    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoint.pth')
    if device.type == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    global_network.load_state_dict(checkpoint['policy_model'])
    global_network.eval()
    print(f"Successfully loaded model from {checkpoint_path}")
    
    env = Env(map_index, fov, n_agents, sensor_range, plot=False)
    utility_range = 0.9 * sensor_range
    node_manager = NodeManager(fov, sensor_range, utility_range, plot=False)
    
    robot_list = []
    for i in range(n_agents):
        agent = Agent(i, global_network, fov, env.angles[i], sensor_range, node_manager, None, device, plot=False)
        robot_list.append(agent)
    
    for robot in robot_list:
        robot.update_graph(env.belief_info, env.robot_locations[robot.id].copy())
    for robot in robot_list:
        robot.update_planning_state(env.robot_locations)
    
    done = False
    max_travel_dist = 0
    trajectory_length = 0
    reach_checkpoint = False
    
    for i in range(MAX_EPISODE_STEP):
        selected_locations = []
        dist_list = []
        next_node_index_list = []
        next_heading_index_list = []
        
        for robot in robot_list:
            observation = robot.get_observation(pad=False)
            next_location, next_node_index, _, next_heading_index = robot.select_next_waypoint(observation, greedy=greedy)
            selected_locations.append(next_location)
            dist_list.append(np.linalg.norm(next_location - robot.location))
            next_node_index_list.append(next_node_index)
            next_heading_index_list.append(next_heading_index)
        
        selected_locations = np.array(selected_locations).reshape(-1, 2)
        arriving_sequence = np.argsort(np.array(dist_list))
        selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]
        
        for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
            solved_locations = selected_locations_in_arriving_sequence[:j]
            while selected_location[0] + selected_location[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                id = arriving_sequence[j]
                nearby_nodes = robot_list[id].node_manager.nodes_dict.nearest_neighbors(
                    selected_location.tolist(), 25)
                for node in nearby_nodes:
                    coords = node.data.coords
                    if coords[0] + coords[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                        continue
                    selected_location = coords
                    break
                selected_locations_in_arriving_sequence[j] = selected_location
                selected_locations[id] = selected_location
        
        for robot, next_location, next_node_index in zip(robot_list, selected_locations, next_node_index_list):
            env.final_sim_step(next_location, robot.id)
            robot.update_graph(env.belief_info, env.robot_locations[robot.id].copy())
        
        for robot in robot_list:
            robot.update_planning_state(env.robot_locations)
        
        max_travel_dist += np.max(dist_list)
        
        if env.explored_rate > INITIAL_EXPLORED_RATE and not reach_checkpoint:
            trajectory_length = max([robot.travel_dist for robot in robot_list])
            reach_checkpoint = True
        
        if env.explored_rate > 0.99:
            done = True
        
        if done:
            break
    
    results = {
        'travel_dist': max([robot.travel_dist for robot in robot_list]),
        'explored_rate': env.explored_rate,
        'success_rate': done,
        'dist_to_0_90': trajectory_length if trajectory_length > 0 else None,
        'steps_taken': i + 1
    }
    
    return results

def run_batch_test(num_tests=10, n_agents=4, fov=120, sensor_range=10):
    print(f"\n{'='*60}")
    print(f"MARVEL Model Testing")
    print(f"{'='*60}")
    print(f"Number of tests: {num_tests}")
    print(f"Number of agents: {n_agents}")
    print(f"FOV: {fov} degrees")
    print(f"Sensor range: {sensor_range} meters")
    print(f"{'='*60}")
    
    all_results = []
    for i in range(num_tests):
        results = test_single_episode(map_index=i, n_agents=n_agents, fov=fov, sensor_range=sensor_range)
        all_results.append(results)
        
        print(f"\nTest {i+1}/{num_tests}:")
        print(f"  Travel distance: {results['travel_dist']:.2f} m")
        print(f"  Explored rate: {results['explored_rate']:.4f}")
        print(f"  Success: {results['success_rate']}")
        print(f"  Steps taken: {results['steps_taken']}")
        if results['dist_to_0_90']:
            print(f"  Distance to 90% explored: {results['dist_to_0_90']:.2f} m")
    
    print(f"\n{'='*60}")
    print(f"Summary Statistics")
    print(f"{'='*60}")
    
    travel_dists = [r['travel_dist'] for r in all_results]
    explored_rates = [r['explored_rate'] for r in all_results]
    success_rates = [r['success_rate'] for r in all_results]
    dist_to_0_90_list = [r['dist_to_0_90'] for r in all_results if r['dist_to_0_90'] is not None]
    
    print(f"Average travel distance: {np.mean(travel_dists):.2f} ± {np.std(travel_dists):.2f} m")
    print(f"Average explored rate: {np.mean(explored_rates):.4f} ± {np.std(explored_rates):.4f}")
    print(f"Success rate: {np.mean(success_rates):.2%} ({sum(success_rates)}/{num_tests})")
    if dist_to_0_90_list:
        print(f"Average distance to 90% explored: {np.mean(dist_to_0_90_list):.2f} ± {np.std(dist_to_0_90_list):.2f} m")
    
    return all_results

if __name__ == '__main__':
    run_batch_test(num_tests=5, n_agents=4, fov=120, sensor_range=10)
import os
import sys
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Wedge

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
os.chdir(project_root)

from utils.model import PolicyNet
from utils.env_test import Env
from utils.agent import Agent
from utils.node_manager import NodeManager
from utils.utils import *
from test_parameter import *

def clean_results_dir():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    if not os.path.exists(results_dir):
        return
    gif_files = glob.glob(os.path.join(results_dir, '*.gif'))
    png_files = glob.glob(os.path.join(results_dir, '*.png'))
    for f in gif_files + png_files:
        os.remove(f)
    print(f"  Cleaned {len(gif_files)} GIFs and {len(png_files)} PNGs from results/")

def load_model(checkpoint_path, device):
    global_network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    global_network.load_state_dict(checkpoint['policy_model'])
    global_network.eval()
    return global_network

def initialize_environment(map_index, n_agents, fov, sensor_range):
    env = Env(map_index, fov, n_agents, sensor_range, plot=False)
    utility_range = 0.9 * sensor_range
    node_manager = NodeManager(fov, sensor_range, utility_range, plot=False)
    return env, node_manager

def initialize_agents(n_agents, policy_net, fov, angles, sensor_range, node_manager, device):
    agents = []
    for i in range(n_agents):
        agent = Agent(i, policy_net, fov, angles[i], sensor_range, node_manager, None, device, plot=False)
        agents.append(agent)
    return agents

def run_exploration(env, agents, max_steps=500, greedy=True, collect_frames=False):
    from utils.motion_model import compute_allowable_heading
    
    for agent in agents:
        agent.update_graph(env.belief_info, env.robot_locations[agent.id].copy())
    for agent in agents:
        agent.update_planning_state(env.robot_locations)
    
    history = {
        'steps': [],
        'explored_rates': [],
        'travel_dists': []
    }
    
    frames = []
    if collect_frames:
        frames.append({
            'belief': env.robot_belief.copy(),
            'locations': [get_cell_position_from_coords(a.location, env.belief_info) for a in agents],
            'headings': [a.heading for a in agents],
            'trajectories': [[get_cell_position_from_coords(a.location, env.belief_info)] for a in agents],
            'step': 0
        })
    
    done = False
    for step in range(max_steps):
        selected_locations = []
        dist_list = []
        next_node_index_list = []
        next_heading_index_list = []
        
        for agent in agents:
            observation = agent.get_observation(pad=False)
            next_location, next_node_index, _, next_heading_index = agent.select_next_waypoint(observation, greedy=greedy)
            selected_locations.append(next_location)
            dist_list.append(np.linalg.norm(next_location - agent.location))
            next_node_index_list.append(next_node_index)
            next_heading_index_list.append(next_heading_index)
        
        selected_locations = np.array(selected_locations).reshape(-1, 2)
        arriving_sequence = np.argsort(np.array(dist_list))
        selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]
        
        for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
            solved_locations = selected_locations_in_arriving_sequence[:j]
            while selected_location[0] + selected_location[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                id = arriving_sequence[j]
                nearby_nodes = agents[id].node_manager.nodes_dict.nearest_neighbors(
                    selected_location.tolist(), 25)
                for node in nearby_nodes:
                    coords = node.data.coords
                    if coords[0] + coords[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                        continue
                    selected_location = coords
                    break
                selected_locations_in_arriving_sequence[j] = selected_location
                selected_locations[id] = selected_location
        
        robot_locations_sim = []
        robot_headings_sim = []
        all_robots_heading_list = []
        robot_cell_paths = []
        
        for agent, next_location, next_heading_index in zip(agents, selected_locations, next_heading_index_list):
            robot_current_cell = get_cell_position_from_coords(agent.location, env.belief_info)
            robot_cell = get_cell_position_from_coords(next_location, env.belief_info)
            
            next_heading = next_heading_index * (360 / NUM_ANGLES_BIN)
            final_heading = compute_allowable_heading(
                agent.location, next_location, agent.heading, next_heading, 
                agent.velocity, agent.yaw_rate
            )
            
            sim_steps = 6
            intermediate_cells = np.linspace(robot_current_cell, robot_cell, sim_steps + 1)[1:]
            intermediate_cells = np.round(intermediate_cells).astype(int)
            
            def smooth_heading(prev, final, steps):
                prev = prev % 360
                final = final % 360
                diff = final - prev
                if abs(diff) > 180:
                    diff = diff - 360 if diff > 0 else diff + 360
                return [(prev + i * diff / steps) % 360 for i in range(1, steps)] + [final]
            
            intermediate_headings = smooth_heading(agent.heading, final_heading, sim_steps)
            
            robot_locations_sim.append(intermediate_cells)
            robot_headings_sim.append(intermediate_headings)
            all_robots_heading_list.append(final_heading)
            robot_cell_paths.append(intermediate_cells)
            
            agent.update_heading(final_heading)
        
        for l in range(6):
            for q in range(len(agents)):
                env.update_robot_belief(robot_locations_sim[q][l], robot_headings_sim[q][l])
            
            if collect_frames and l % 2 == 0:
                frames.append({
                    'belief': env.robot_belief.copy(),
                    'locations': [robot_cell_paths[q][l] for q in range(len(agents))],
                    'headings': [robot_headings_sim[q][l] for q in range(len(agents))],
                    'trajectories': [traj + [robot_cell_paths[q][l]] for q, traj in enumerate(frames[-1]['trajectories'])],
                    'step': step
                })
        
        for agent, next_location, next_node_index in zip(agents, selected_locations, next_node_index_list):
            env.final_sim_step(next_location, agent.id)
            agent.update_graph(env.belief_info, env.robot_locations[agent.id].copy())
        
        for agent in agents:
            agent.update_planning_state(env.robot_locations)
        
        history['steps'].append(step)
        history['explored_rates'].append(env.explored_rate)
        history['travel_dists'].append(max([a.travel_dist for a in agents]))
        
        if collect_frames:
            frames.append({
                'belief': env.robot_belief.copy(),
                'locations': [get_cell_position_from_coords(a.location, env.belief_info) for a in agents],
                'headings': [a.heading for a in agents],
                'trajectories': [traj + [get_cell_position_from_coords(a.location, env.belief_info)] for a, traj in zip(agents, frames[-1]['trajectories'])],
                'step': step + 1
            })
        
        if env.explored_rate > 0.99:
            done = True
            break
        
        if step % 50 == 0:
            print(f"  Step {step}: Explored {env.explored_rate:.4f}, Max travel {max([a.travel_dist for a in agents]):.2f}m")
    
    if collect_frames:
        return history, done, frames
    return history, done

def plot_results(history, map_index, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['steps'], history['explored_rates'], 'b-', linewidth=2)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Explored Rate')
    ax1.set_title(f'Exploration Progress (Map {map_index})')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    ax2.plot(history['steps'], history['travel_dists'], 'r-', linewidth=2)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Max Travel Distance (m)')
    ax2.set_title(f'Travel Distance (Map {map_index})')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved to {save_path}")

def create_animation(frames, ground_truth, map_index, save_path, fps=5, fov=120, sensor_range=10, cell_size=0.2):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    fov_colors = [(1.0, 0.42, 0.42, 0.4), (0.31, 0.80, 0.77, 0.4), 
                  (0.27, 0.72, 0.82, 0.4), (0.59, 0.81, 0.71, 0.4),
                  (1.0, 0.92, 0.65, 0.4), (0.87, 0.63, 0.87, 0.4)]
    
    im_ground = ax.imshow(ground_truth, cmap='gray', vmin=0, vmax=255, alpha=0.3)
    im_belief = ax.imshow(np.zeros_like(ground_truth), cmap='gray', vmin=0, vmax=255, alpha=0.7)
    
    scatter_plots = []
    for i in range(len(frames[0]['locations'])):
        sc = ax.scatter([], [], color=colors[i % len(colors)], 
                        s=60, edgecolor='white', linewidth=1.5, zorder=10)
        scatter_plots.append(sc)
    
    traj_lines = []
    for i in range(len(frames[0]['locations'])):
        line, = ax.plot([], [], color=colors[i % len(colors)], 
                        linestyle='-', linewidth=1.5, alpha=0.6)
        traj_lines.append(line)
    
    text_labels = []
    for i in range(len(frames[0]['locations'])):
        txt = ax.text(0, 0, str(i + 1), color='white', 
                      fontsize=8, fontweight='bold', zorder=11)
        text_labels.append(txt)
    
    fov_wedges = []
    fov_angle_labels = []
    fov_boundary_lines = []
    fov_center_lines = []
    sensor_range_cells = int(sensor_range / cell_size)
    
    for i in range(len(frames[0]['locations'])):
        wedge = Wedge((0, 0), sensor_range_cells, 0, fov, 
                      facecolor=fov_colors[i % len(fov_colors)], 
                      edgecolor=colors[i % len(colors)], 
                      linewidth=2, alpha=0.7, zorder=5)
        ax.add_patch(wedge)
        fov_wedges.append(wedge)
        
        angle_label = ax.text(0, 0, f'{fov}°', color='white', 
                              fontsize=7, fontweight='bold', 
                              ha='center', va='center', zorder=12)
        fov_angle_labels.append(angle_label)
        
        bound_line1, = ax.plot([], [], color=colors[i % len(colors)], 
                               linestyle='--', linewidth=1.5, alpha=0.8)
        bound_line2, = ax.plot([], [], color=colors[i % len(colors)], 
                               linestyle='--', linewidth=1.5, alpha=0.8)
        fov_boundary_lines.append((bound_line1, bound_line2))
        
        center_line, = ax.plot([], [], color=colors[i % len(colors)], 
                               linestyle='-', linewidth=2, alpha=0.9)
        fov_center_lines.append(center_line)
    
    title = ax.set_title(f'MARVEL Exploration - Map {map_index}\nStep 0')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    
    def update(frame_idx):
        frame = frames[frame_idx]
        belief = frame['belief']
        locations = frame['locations']
        trajectories = frame['trajectories']
        headings = frame['headings']
        
        belief_display = np.where(belief == 127, 200, 
                                  np.where(belief == 255, 50, 
                                           np.where(belief == 0, 100, belief)))
        im_belief.set_data(belief_display)
        
        for i, (loc, traj, heading) in enumerate(zip(locations, trajectories, headings)):
            scatter_plots[i].set_offsets(loc)
            
            if len(traj) > 1:
                traj_np = np.array(traj)
                traj_lines[i].set_data(traj_np[:, 0], traj_np[:, 1])
            else:
                traj_lines[i].set_data([], [])
            
            text_labels[i].set_position((loc[0] + 2, loc[1] + 2))
            
            theta1 = (heading - fov / 2) % 360
            theta2 = (heading + fov / 2) % 360
            fov_wedges[i].set_center((loc[0], loc[1]))
            fov_wedges[i].set_theta1(theta1)
            fov_wedges[i].set_theta2(theta2)
            
            heading_rad = np.radians(heading)
            half_fov_rad = np.radians(fov / 2)
            
            inner_r = sensor_range_cells * 0.6
            
            center_x = loc[0] + inner_r * np.cos(heading_rad)
            center_y = loc[1] + inner_r * np.sin(heading_rad)
            fov_angle_labels[i].set_position((center_x, center_y))
            
            bound1_x = loc[0] + sensor_range_cells * np.cos(heading_rad - half_fov_rad)
            bound1_y = loc[1] + sensor_range_cells * np.sin(heading_rad - half_fov_rad)
            fov_boundary_lines[i][0].set_data([loc[0], bound1_x], [loc[1], bound1_y])
            
            bound2_x = loc[0] + sensor_range_cells * np.cos(heading_rad + half_fov_rad)
            bound2_y = loc[1] + sensor_range_cells * np.sin(heading_rad + half_fov_rad)
            fov_boundary_lines[i][1].set_data([loc[0], bound2_x], [loc[1], bound2_y])
            
            center_end_x = loc[0] + sensor_range_cells * np.cos(heading_rad)
            center_end_y = loc[1] + sensor_range_cells * np.sin(heading_rad)
            fov_center_lines[i].set_data([loc[0], center_end_x], [loc[1], center_end_y])
        
        title.set_text(f'MARVEL Exploration - Map {map_index}\nStep {frame["step"]}')
        
        all_artists = [im_belief] + scatter_plots + traj_lines + text_labels + fov_wedges + fov_angle_labels
        for bl1, bl2 in fov_boundary_lines:
            all_artists.extend([bl1, bl2])
        all_artists.extend(fov_center_lines)
        
        return all_artists
    
    anim = FuncAnimation(fig, update, 
                         frames=len(frames), interval=1000/fps, blit=True)
    
    anim.save(save_path, writer='pillow', fps=fps, dpi=100)
    plt.close()
    print(f"  Animation saved to {save_path}")

def reproduce_single_map(map_index=0, n_agents=4, fov=120, sensor_range=10, 
                         max_steps=500, greedy=True, visualize=True, animate=False, fps=5):
    print(f"\n{'='*50}")
    print(f"Reproducing MARVEL on Map {map_index}")
    print(f"{'='*50}")
    
    device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    print(f"Using device: {device}")
    
    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoint.pth')
    policy_net = load_model(checkpoint_path, device)
    print(f"Model loaded from: {checkpoint_path}")
    
    env, node_manager = initialize_environment(map_index, n_agents, fov, sensor_range)
    print(f"Environment initialized: {env.ground_truth.shape[0]}x{env.ground_truth.shape[1]} cells")
    
    agents = initialize_agents(n_agents, policy_net, fov, env.angles, sensor_range, node_manager, device)
    print(f"Initialized {n_agents} agents")
    
    print("\nRunning exploration...")
    if animate:
        history, done, frames = run_exploration(env, agents, max_steps, greedy, collect_frames=True)
    else:
        history, done = run_exploration(env, agents, max_steps, greedy)
    
    print(f"\nResults:")
    print(f"  Final explored rate: {history['explored_rates'][-1]:.4f}")
    print(f"  Max travel distance: {history['travel_dists'][-1]:.2f} m")
    print(f"  Steps taken: {len(history['steps'])}")
    print(f"  Success (>=99% explored): {done}")
    
    if visualize or animate:
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'), exist_ok=True)
    
    if visualize:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', f'map_{map_index}_results.png')
        plot_results(history, map_index, save_path)
    
    if animate:
        anim_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', f'map_{map_index}_animation.gif')
        create_animation(frames, env.ground_truth, map_index, anim_path, fps=fps, 
                         fov=fov, sensor_range=sensor_range, cell_size=env.cell_size)
    
    return {
        'map_index': map_index,
        'explored_rate': history['explored_rates'][-1],
        'travel_dist': history['travel_dists'][-1],
        'steps': len(history['steps']),
        'success': done
    }

def batch_reproduce(num_tests=10, n_agents=4, fov=120, sensor_range=10, max_steps=500, random=False, animate=False, fps=5):
    clean_results_dir()
    
    print(f"\n{'='*60}")
    print(f"MARVEL Model Batch Reproduction")
    print(f"{'='*60}")
    print(f"Number of tests: {num_tests}")
    print(f"Number of agents: {n_agents}")
    print(f"FOV: {fov} degrees")
    print(f"Sensor range: {sensor_range} meters")
    print(f"Max steps: {max_steps}")
    print(f"Random selection: {'Yes' if random else 'No'}")
    print(f"Create animations: {'Yes' if animate else 'No'}")
    print(f"{'='*60}")
    
    all_results = []
    if random:
        test_maps = glob.glob('maps_test/*.png')
        total_maps = len(test_maps)
        if total_maps == 0:
            test_maps = glob.glob('../../maps_test/*.png')
            total_maps = len(test_maps)
        map_indices = np.random.choice(total_maps, min(num_tests, total_maps), replace=False)
        print(f"Randomly selected map indices: {map_indices}")
    else:
        map_indices = range(num_tests)
    
    for i, map_idx in enumerate(map_indices):
        result = reproduce_single_map(map_index=map_idx, n_agents=n_agents, fov=fov, 
                                      sensor_range=sensor_range, max_steps=max_steps,
                                      visualize=True, animate=animate, fps=fps)
        result['map_index'] = map_idx
        all_results.append(result)
    
    print(f"\n{'='*60}")
    print(f"Summary Statistics")
    print(f"{'='*60}")
    
    explored_rates = [r['explored_rate'] for r in all_results]
    travel_dists = [r['travel_dist'] for r in all_results]
    success_rates = [r['success'] for r in all_results]
    steps = [r['steps'] for r in all_results]
    map_indices_list = [r['map_index'] for r in all_results]
    
    print(f"Average explored rate: {np.mean(explored_rates):.4f} ± {np.std(explored_rates):.4f}")
    print(f"Average travel distance: {np.mean(travel_dists):.2f} ± {np.std(travel_dists):.2f} m")
    print(f"Average steps: {np.mean(steps):.2f} ± {np.std(steps):.2f}")
    print(f"Success rate: {np.mean(success_rates):.2%} ({sum(success_rates)}/{num_tests})")
    
    generate_summary_chart(map_indices_list, explored_rates, travel_dists, steps, success_rates)
    
    return all_results

def generate_summary_chart(map_indices, explored_rates, travel_dists, steps, success_rates):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].bar(range(len(map_indices)), explored_rates, color='#45B7D1', alpha=0.7)
    axes[0, 0].axhline(y=0.99, color='r', linestyle='--', linewidth=1.5, label='99% threshold')
    axes[0, 0].set_title('Explored Rate per Map')
    axes[0, 0].set_xlabel('Map Index')
    axes[0, 0].set_ylabel('Explored Rate')
    axes[0, 0].set_ylim(0.9, 1.0)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(explored_rates, bins=10, color='#96CEB4', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=0.99, color='r', linestyle='--', linewidth=1.5)
    axes[0, 1].set_title('Distribution of Explored Rates')
    axes[0, 1].set_xlabel('Explored Rate')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].scatter(steps, explored_rates, c=travel_dists, cmap='viridis', alpha=0.7, s=50)
    axes[1, 0].axhline(y=0.99, color='r', linestyle='--', linewidth=1.5)
    axes[1, 0].set_title('Explored Rate vs Steps')
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].set_ylabel('Explored Rate')
    axes[1, 0].set_ylim(0.9, 1.0)
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0], label='Travel Distance (m)')
    
    axes[1, 1].scatter(travel_dists, explored_rates, c=steps, cmap='plasma', alpha=0.7, s=50)
    axes[1, 1].axhline(y=0.99, color='r', linestyle='--', linewidth=1.5)
    axes[1, 1].set_title('Explored Rate vs Travel Distance')
    axes[1, 1].set_xlabel('Travel Distance (m)')
    axes[1, 1].set_ylabel('Explored Rate')
    axes[1, 1].set_ylim(0.9, 1.0)
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Steps')
    
    plt.tight_layout()
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    plt.savefig(os.path.join(results_dir, 'batch_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSummary chart saved to {os.path.join(results_dir, 'batch_summary.png')}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Reproduce MARVEL Multi-Agent Exploration')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'batch'],
                        help='Test mode: single map or batch testing')
    parser.add_argument('--map-index', type=int, default=0, help='Map index for single mode')
    parser.add_argument('--num-tests', type=int, default=5, help='Number of tests for batch mode')
    parser.add_argument('--n-agents', type=int, default=4, help='Number of agents')
    parser.add_argument('--fov', type=int, default=120, help='Field of view in degrees')
    parser.add_argument('--sensor-range', type=int, default=10, help='Sensor range in meters')
    parser.add_argument('--max-steps', type=int, default=500, help='Maximum steps per episode')
    parser.add_argument('--visualize', action='store_true', default=True, help='Save results plot')
    parser.add_argument('--animate', action='store_true', default=False, help='Create animation GIF')
    parser.add_argument('--fps', type=int, default=5, help='Frames per second for animation')
    parser.add_argument('--random', action='store_true', default=False, help='Randomly select maps for batch testing')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        reproduce_single_map(
            map_index=args.map_index,
            n_agents=args.n_agents,
            fov=args.fov,
            sensor_range=args.sensor_range,
            max_steps=args.max_steps,
            visualize=args.visualize,
            animate=args.animate,
            fps=args.fps
        )
    else:
        batch_reproduce(
            num_tests=args.num_tests,
            n_agents=args.n_agents,
            fov=args.fov,
            sensor_range=args.sensor_range,
            max_steps=args.max_steps,
            random=args.random,
            animate=args.animate,
            fps=args.fps
        )
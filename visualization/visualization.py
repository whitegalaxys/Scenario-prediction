import torch
import argparse
import glob
import os
import logging
import time
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.style as mplstyle
import pandas as pd
import tensorflow as tf
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from matplotlib import cm
from matplotlib.collections import LineCollection
from utils.test_utils import *
from utils.simulator import *
from model.planner import MotionPlanner
from model.predictor import Predictor
from waymo_open_dataset.protos import scenario_pb2

def open_loop_test():
    # logging
    log_path = f"./testing_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'test.log')

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Use integrated planning module: {}".format(args.use_planning))
    logging.info("Use device: {}".format(args.device))
    
    #matplotlib.use('agg')
    mplstyle.use('fast')

    # test file
    files = glob.glob(args.test_set+'/*')
    processor = TestDataProcess()

    # cache results
    collisions = []
    red_light, off_route = [], []
    Accs, Jerks, Lat_Accs = [], [], []
    Human_Accs, Human_Jerks, Human_Lat_Accs = [], [], []
    similarity_1s, similarity_3s, similarity_5s = [], [], []
    prediction_ADE, prediction_FDE = [], []

    # load model
    predictor = Predictor(50).to(args.device)
    predictor.load_state_dict(torch.load(args.model_path, map_location=args.device))
    predictor.eval()
    
    startcolor=(254/255,194/255,83/255)
    midcolor=(175/255,182/255,137/255)
    endcolor=(95/255,170/255,191/255) 
    my_cmap = LinearSegmentedColormap.from_list("mycmap", colors=[startcolor,midcolor,endcolor])
    plt.cm.register_cmap(name='mycmap',cmap=my_cmap)

    
    # set up planner
    if args.use_planning:
        trajectory_len, feature_len = 50, 9
        planner = MotionPlanner(trajectory_len, feature_len, device=args.device, test=True)

    # iterate test files
    for file in files:
        scenarios = tf.data.TFRecordDataset(file)
        simulator = Simulator(150) # temporal horizon 15s    

        # iterate scenarios in the test file
        for scenario in scenarios:
            parsed_data = scenario_pb2.Scenario()
            parsed_data.ParseFromString(scenario.numpy())
            simulator.load_scenario(parsed_data)
            obs = simulator.reset()
            scenario_id = parsed_data.scenario_id
            sdc_id = parsed_data.sdc_track_index
            timesteps = parsed_data.timestamps_seconds

            # build map
            processor.build_map(parsed_data.map_features, parsed_data.dynamic_map_states)

            # get a testing scenario
            for timestep in range(20, len(timesteps)-50, 1):
                logging.info(f"Scenario: {scenario_id} Time: {timestep}")
                
                # prepare data
                input_data = processor.process_frame(timestep, sdc_id, parsed_data.tracks)
                ego = torch.from_numpy(input_data[0]).to(args.device)
                neighbors = torch.from_numpy(input_data[1]).to(args.device)
                lanes = torch.from_numpy(input_data[2]).to(args.device)
                crosswalks = torch.from_numpy(input_data[3]).to(args.device)
                ref_line = torch.from_numpy(input_data[4]).to(args.device)
                neighbor_ids, norm_gt_data, gt_data = input_data[5], input_data[6], input_data[7]
                current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]

                # predict
                with torch.no_grad():
                    plans, predictions, scores, cost_function_weights = predictor(ego, neighbors, lanes, crosswalks)
                    plan, prediction = select_future(plans, predictions, scores)

               
                plan = bicycle_model(plan, ego[:, -1])[:, :, :3]
                plan = plan.cpu().numpy()[0]


                ### plot scenario ###
                if args.render:
                    # visualization
                    plt.ion()
                    ax = plt.gca()
                    fig = plt.gcf()
                    dpi = 100
                    size_inches = 800 / dpi
                    fig.set_size_inches([size_inches, size_inches])
                    fig.set_dpi(dpi)
                    fig.set_tight_layout(True)
                    fig.set_facecolor("#ffffff") 
                    
                    

                    # map
                    for vector in parsed_data.map_features:
                        vector_type = vector.WhichOneof("feature_data")
                        vector = getattr(vector, vector_type)
                        polyline = map_process(vector, vector_type)

                    # sdc
                    agent_color = ['r', 'm', 'b', 'g'] # [sdc, vehicle, pedestrian, cyclist]
                    color = agent_color[0]
                    track = parsed_data.tracks[sdc_id].states[timestep]
                    curr_state = (track.center_x, track.center_y, track.heading)
                    plan = transform(plan, curr_state, include_curr=True)

                    rect = plt.Rectangle((track.center_x-track.length/2, track.center_y-track.width/2), 
                                        track.length, track.width, linewidth=2, facecolor = 'white', edgecolor = 'red', alpha=0.6, zorder=3,
                                        transform=mpl.transforms.Affine2D().rotate_around(*(track.center_x, track.center_y), track.heading) + ax.transData)
                    ax.add_patch(rect)
                    #plt.plot(plan[::5, 0], plan[::5, 1], linewidth=2, color=color, marker='.', markersize=6, zorder=4)
                    ego_gt = np.insert(gt_data[0, :, :3], 0, curr_state, axis=0)
                    plt.plot(ego_gt[:, 0], ego_gt[:, 1], '#333631', linewidth=1.3, zorder=3, linestyle='dashed')

                    # neighbors
                    for i, id in enumerate(neighbor_ids):
                        track = parsed_data.tracks[id].states[timestep]
                        color = agent_color[parsed_data.tracks[id].object_type]
                        rect = plt.Rectangle((track.center_x-track.length/2, track.center_y-track.width/2), 
                                            track.length, track.width, linewidth=1.5, facecolor = 'white', edgecolor = '#fda700', alpha=0.6, zorder=3,
                                            transform=mpl.transforms.Affine2D().rotate_around(*(track.center_x, track.center_y), track.heading) + ax.transData)
                        ax.add_patch(rect)
                        predict_traj = prediction.cpu().numpy()[0, i]
                        predict_traj = transform(predict_traj, curr_state)
                        predict_traj = np.insert(predict_traj, 0, (track.center_x, track.center_y), axis=0)
                        #plt.plot(predict_traj[::5, 0], predict_traj[::5, 1], linewidth=1, color=color, marker='.', markersize=6, zorder=3))
                        norm = plt.Normalize(min(predict_traj[::5, 1].min(),predict_traj[::5, 0].min()), max(predict_traj[::5, 1].max(), predict_traj[::5, 0].max()))
                        norm_xy = norm(abs(predict_traj[::5, 1]+predict_traj[::5, 0]-(predict_traj[0, 1]+predict_traj[0, 0])))
                        points = np.array([predict_traj[::5, 0], predict_traj[::5, 1]]).T.reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        lc = LineCollection(segments, cmap=plt.cm.get_cmap('mycmap'))
                        # Set the values used for colormapping
                        lc.set_array(norm_xy)
                        lc.set_linewidth(3)
                        line = ax.add_collection(lc)
                        ax.set_xlim(predict_traj[::5, 0].min(), predict_traj[::5, 0].max());
                        ax.set_ylim(predict_traj[::5, 1].min() - 1, predict_traj[::5, 1].max() + 1);

                        other_gt = np.insert(gt_data[i+1, :, :3], 0, (track.center_x, track.center_y, track.heading), axis=0)
                        other_gt = other_gt[other_gt[:, 0] != 0]
                        plt.plot(other_gt[:, 0], other_gt[:, 1], '#333631', linewidth=1.3, zorder=3, linestyle='dashed')         

                    for i, track in enumerate(parsed_data.tracks):
                        if i not in [sdc_id] + neighbor_ids and track.states[timestep].valid:
                            rect = plt.Rectangle((track.states[timestep].center_x-track.states[timestep].length/2, track.states[timestep].center_y-track.states[timestep].width/2), 
                                                track.states[timestep].length, track.states[timestep].width, linewidth=2, facecolor = 'white', edgecolor='#727272', alpha=0.6, zorder=2,
                                                transform=mpl.transforms.Affine2D().rotate_around(*(track.states[timestep].center_x, track.states[timestep].center_y), track.states[timestep].heading) + ax.transData)
                            ax.add_patch(rect)

                    # dynamic_map_states
                    for signal in parsed_data.dynamic_map_states[timestep].lane_states:
                        traffic_signal_process(processor.lanes, signal)

                    # show plot
                    ax.axis([-100 + plan[0, 0], 100 + plan[0, 0], -100 + plan[0, 1], 100 + plan[0, 1]])
                    ax.set_aspect('equal')
                    ax.grid(False)
                    ax.margins(0) 
                    ax.axis('off') 
                    ax.axes.get_yaxis().set_visible(False)
                    ax.axes.get_xaxis().set_visible(False)
                    
                    fig.canvas.draw()
                    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))   
                    simulator.scene_imgs.append(data)   
                    plt.pause(0.01)
                    plt.clf()
        
            # save image
            if args.save:
                simulator.save_animation(log_path)

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name (default: "Test1")', default="Test1")
    parser.add_argument('--test_set', type=str, help='path to testing datasets')
    parser.add_argument('--model_path', type=str, help='path to saved model')
    parser.add_argument('--use_planning', action="store_true", help='if use integrated planning module (default: False)', default=False)
    parser.add_argument('--render', action="store_true", help='if render the scenario (default: False)', default=False)
    parser.add_argument('--save', action="store_true", help='if save the rendered images (default: False)', default=False)
    parser.add_argument('--device', type=str, help='run on which device (default: cpu)', default='cpu')
    args = parser.parse_args()

    # Run
    open_loop_test()

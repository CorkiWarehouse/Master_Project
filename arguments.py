'''
Packages:
 1. Time : this is used to better use PyTorch
 2. Torch : this is PyTorch
 3. Argparse: this is used to parse arguments (command line arguments)
'''

import time
import torch
import argparse


# Define our GPU or CPU to train finish our job
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Define the time
# Then transform it to String YY/MM/DD/HH/MM
time_now = time.strftime("%Y_%m_%d_%H%M%S")

# This function is used to Create ArgumentParser instance
def parse_args():
    parser = argparse.ArgumentParser("agent level inverse reinforcement learning for mean-field games")

    # system info
    # including "Time" and "Device"
    parser.add_argument("--time", type=str, default=time_now, help="system time")
    parser.add_argument("--device", default=device, help="torch device")

    # environment
    '''
    All of these are Environment variables
    Arguments:
        1. --env_name: name of environment
        2. --is_original_dynamics : whether to use original dynamics (Change some Arguments)
        3. --horizon: horizon of mean field games (Time Range)
        4. --gamma: discount factor (Training Rate Or Learning rate)
        5. --beta: entropy regularisation strength (Max Entropy Using?)
    '''
    parser.add_argument("--env_name", type=str, default="CARS", help="the environment model (LR)")
    parser.add_argument("--is_original_dynamics", type=int, default=0, help="original or new dynamics. 0: original, 1: new")
    parser.add_argument("--horizon", type=int, default=8, help="horizon of mean field games")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--beta", type=float, default=1.0, help="entropy regularisation strength")

    # core training parameters
    '''
    MLP: Multi-Layer Perceptron
    
    All of these are Training Variables 
    Arguments:
        1. --max_epoch: maximum epoch length (T=100 then we end)
        2. -max_grad_norm: max gradient norm for clip (Incase we have gradient explode)
        3. --lr: learning rate for adam optimizer (Adam optimal)
        4. --num_units_1 / num_units_2: number of units (Define the structure of the 
                                                            neural network used in the model) 
        5. --num_units : number of units
    '''
    parser.add_argument("--max_epoch", type=int, default=15, help="maximum epoch length")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="max gradient norm for clip")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for adam optimizer")
    parser.add_argument("--num_units_1", type=int, default=64, help="number of units in the MLP")
    parser.add_argument("--num_units_2", type=int, default=32, help="number of units in the MLP")
    parser.add_argument("--num_units", type=int, default=32, help="number of units in the MLP")

    # experiment control parameters

    '''
    Controlling the scale and repetitiveness of the experiments
    
    Arguments:
        1. -num_runs: Times of running for the experiment (?)
        2. --num_traj: number of trajectories for each agent to run (How many "t" in D)
        3. --max_num_game_plays: maximal number of sampled game_plays (N)
    '''

    parser.add_argument("--num_runs", type=int, default=10, help="number of independent runs")
    parser.add_argument("--num_traj", type=int, default=10, help="number of trajectories generated per game play")
    parser.add_argument("--max_num_game_plays", type=int, default=10, help="maximal number of sampled game_plays")

    # checkpointing

    '''
    Epoch : For NN 
    
    
    Arguments:  All of these are used to record the experiments and save it 
        1. --fre4save_model: episode
        2. --start_save_model: epochs
        3. --start_save_log: interval of epoch
        4. --save/--load_model_dir: directory to save model
        4. --save_results_dir: directory to save results
    '''

    parser.add_argument("--fre4save_model", type=int, default=100, help="the number of the episode for saving the model")
    parser.add_argument("--start_save_model", type=int, default=10, help="the number of the epoches for saving the model")
    parser.add_argument("--start_save_log", type=int, default=1,
                        help="interval of epoch for saving the log")
    parser.add_argument("--save_model_dir", type=str, default="./model_saved/",
            help="directory in which training state and model should be saved")
    parser.add_argument("--load_model_dir", type=str, default="./model_saved/",
            help="directory in which training state and model are loaded"),
    parser.add_argument("--save_results_dir", type=str, default="./results_output/",
                        help="directory which results are output to")

    return parser.parse_args()
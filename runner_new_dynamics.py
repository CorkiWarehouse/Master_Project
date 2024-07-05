import time

import numpy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

import arguments
import Environments
from Algorithms.expert_training import Expert
from Algorithms.mfairl import MFAIRL
from Algorithms.plirl import PLIRL


if __name__ == '__main__':
    arglist = arguments.parse_args()
    
    # prepare the output dictionary
    results_save_path = arglist.save_results_dir + 'new_dynamics/' + arglist.env_name + '/' + arglist.num_traj + '/' + arglist.time + '/'
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)
    
    # initialise environment
    arglist.is_original_dynamics = 1  # new dynamics
    env = Environments.load(arglist.env_name).Env(arglist.is_original_dynamics)
    results = pd.DataFrame(columns=['samples', 'method', 'return', 'Dev. MF', 'Dev. Policy'])
    mfirl = MFAIRL(data=None, env=env, horizon=arglist.horizon, device=arglist.device)
    mdpmfgirl = PLIRL(data=None, env=env, horizon=arglist.horizon, device=arglist.device)


    # initialise the expert
    expert = Expert(env=env, horizon=arglist.horizon)
    expert.compute_ermfne()

    for run in range(arglist.num_runs):
        print('================== Run #' + str(run) + '=========================')
        # load model
        mfirl = MFAIRL(data=None, env=env, horizon=arglist.horizon, device=arglist.device)
        mdpmfgirl = PLIRL(data=None, env=env, horizon=arglist.horizon, device=arglist.device)
        mfirl.load_model(arglist.save_model_dir + 'mfirl_' + arglist.env_name + '_' + str(arglist.max_num_game_plays) + '_' + str(run) + '.pt')
        mdpmfgirl.load_model(arglist.save_model_dir + 'mdp_' + arglist.env_name + '_' + str(arglist.max_num_game_plays) + '_' + str(run) + '.pt')
        # compute ERMFNE
        mfirl_expected_return, mfirl_dev_mf, mfirl_dev_p = mfirl.divergence(expert_mf_flow=expert.mf_flow, expert_p_flow=expert.p_flow)
        mdp_expected_return, mdp_dev_mf, mdp_dev_p = mdpmfgirl.divergence(expert_mf_flow=expert.mf_flow, expert_p_flow=expert.p_flow)

        results = results.append(pd.DataFrame([[run, 'MFIRL', float(mfirl_expected_return), float(mfirl_dev_mf), float(mfirl_dev_p)],
                                               [run, 'MDPMFG-IRL', float(mdp_expected_return), float(mdp_dev_mf), float(mdp_dev_p)]],
                                              columns=['samples', 'method', 'return', 'Dev. MF', 'Dev. Policy']))
        results = results.append(pd.DataFrame([[run, 'EXPERT', float(expert.expected_return), 0.0, 0.0]],
                                              columns=['run', 'method', 'return', 'Dev. MF', 'Dev. Policy']))


    results.to_csv(results_save_path + 'results.csv')


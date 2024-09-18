import time
import os

import numpy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

import arguments
import Environments
from Algorithms.PIIRL import PIIRL
from Algorithms.expert_training import Expert
from Algorithms.mfairl import MFAIRL
from Algorithms.plirl import PLIRL


if __name__ == '__main__':
    # arglist = arguments.parse_args()
    #
    # # prepare the output dictionary
    # results_save_path = arglist.save_results_dir + 'new_dynamics/' + arglist.env_name + '/' + arglist.num_traj + '/' + arglist.time + '/'
    # if not os.path.exists(results_save_path):
    #     os.makedirs(results_save_path)
    #

    # include all the arguments
    arglist = arguments.parse_args()

    # prepare the output dictionary
    results_save_path = arglist.save_results_dir +  'new_dynamics/' + arglist.env_name + '/' + str(
        arglist.num_traj) + '/' + str(arglist.time) + '/'
    # If not exist, create new one
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)

    # initialise environment
    arglist.is_original_dynamics = 1  # new dynamics
    env = Environments.load(arglist.env_name).Env(arglist.is_original_dynamics,arglist.beta)
    results = pd.DataFrame(columns=['samples', 'method', 'return', 'Dev. MF', 'Dev. Policy'])
    # mfirl = MFAIRL(data=None, env=env, horizon=arglist.horizon, device=arglist.device)
    # mdpmfgirl = PLIRL(data=None, env=env, horizon=arglist.horizon, device=arglist.device)

    # initialise the expert
    expert = Expert(env=env, horizon=arglist.horizon)
    expert.compute_ermfne()

    for run in range(arglist.num_runs):
        print('================== Run #' + str(run) + '=========================')
        # load model
        # mfirl = MFAIRL(data=None, env=env, horizon=arglist.horizon, device=arglist.device)
        # piirl = PIIRL(data_expert = None, env=env, horizon=arglist.horizon, device=arglist.device, num_of_game_plays=arglist.max_num_game_plays, num_traj= arglist.num_traj)
        #
        #
        # mfirl.load_model(arglist.save_model_dir+ 'original_dynamics/FLOCK/10/2024_09_03_164557/' + 'mfirl_' + str(arglist.max_num_game_plays) + '_' + str(run) + '.pt')
        # piirl.load_model(arglist.save_model_dir + 'original_dynamics/FLOCK/10/2024_09_03_164557/' + 'piairl_' + str(arglist.max_num_game_plays) + '_' + str(run) + '.pt')

        for num_game_play in range(1, arglist.max_num_game_plays + 1):
            # 加载模型
            mfirl = MFAIRL(data=None, env=env, horizon=arglist.horizon, device=arglist.device)
            piirl = PIIRL(data_expert=None, env=env, horizon=arglist.horizon, device=arglist.device,
                          num_of_game_plays=num_game_play, num_traj=arglist.num_traj)

            # 模型文件路径格式可能需要根据实际保存逻辑调整
            mfirl_model_path = f"{arglist.save_model_dir}original_dynamics/CARS/10/2024_09_03_160710/mfirl_{num_game_play}_{run}.pt"
            piirl_model_path = f"{arglist.save_model_dir}original_dynamics/CARS/10/2024_09_03_160710/piairl_{num_game_play}_{run}.pt"

            mfirl.load_model(mfirl_model_path)
            piirl.load_model(piirl_model_path)

            # compute ERMFNE
            mfirl_expected_return, mfirl_dev_mf, mfirl_dev_p = mfirl.divergence(expert_mf_flow=expert.mf_flow, expert_p_flow=expert.p_flow)
            piirl_expected_return, piirl_dev_mf, piirl_dev_p = piirl.divergence(expert_mf_flow=expert.mf_flow, expert_p_flow=expert.p_flow)

        # results = results._append(pd.DataFrame([[run, 'MFIRL', float(abs(mfirl_expected_return - expert.expected_return)), float(mfirl_dev_mf), float(mfirl_dev_p)],
        #                                        [run, 'PIIRL', float(abs(piirl_expected_return - expert.expected_return)), float(piirl_dev_mf), float(piirl_dev_p)]],
        #                                       columns=['samples', 'method', 'return', 'Dev. MF', 'Dev. Policy']))
        # results = results._append(pd.DataFrame([[run, 'EXPERT', 0.0, 0.0, 0.0]],
        #                                       columns=['run', 'method', 'return', 'Dev. MF', 'Dev. Policy']))
            new_results = pd.DataFrame([
                {'run': run, 'game_play': num_game_play, 'method': 'MFIRL',
                 'return': abs(mfirl_expected_return - expert.expected_return), 'Dev. MF': mfirl_dev_mf,
                 'Dev. Policy': mfirl_dev_p},
                {'run': run, 'game_play': num_game_play, 'method': 'PIIRL',
                 'return': abs(piirl_expected_return - expert.expected_return), 'Dev. MF': piirl_dev_mf,
                 'Dev. Policy': piirl_dev_p},
                {'run': run, 'game_play': num_game_play, 'method': 'EXPERT', 'return': expert.expected_return,
                 'Dev. MF': 0.0, 'Dev. Policy': 0.0}
            ])
            results = pd.concat([results, new_results], ignore_index=True)

            print(num_game_play)


    results.to_csv(results_save_path + 'results.csv')


'''

import pandas as pd

# 先初始化结果DataFrame
results = pd.DataFrame(columns=['run', 'game_play', 'method', 'return', 'Dev. MF', 'Dev. Policy'])

for run in range(arglist.num_runs):
    print(f'================== Run #{run} =========================')
    for num_game_play in range(1, arglist.max_num_game_plays + 1):
        # 加载模型
        mfirl = MFAIRL(data=None, env=env, horizon=arglist.horizon, device=arglist.device)
        piirl = PIIRL(data_expert=None, env=env, horizon=arglist.horizon, device=arglist.device, num_of_game_plays=num_game_play, num_traj=arglist.num_traj)

        # 模型文件路径格式可能需要根据实际保存逻辑调整
        mfirl_model_path = f"{arglist.save_model_dir}original_dynamics/FLOCK/10/2024_09_03_164557/mfirl_{num_game_play}_{run}.pt"
        piirl_model_path = f"{arglist.save_model_dir}original_dynamics/FLOCK/10/2024_09_03_164557/piairl_{num_game_play}_{run}.pt"
        
        mfirl.load_model(mfirl_model_path)
        piirl.load_model(piirl_model_path)

        # 计算ERMFE
        mfirl_expected_return, mfirl_dev_mf, mfirl_dev_p = mfirl.divergence(expert_mf_flow=expert.mf_flow, expert_p_flow=expert.p_flow)
        piirl_expected_return, piirl_dev_mf, piirl_dev_p = piirl.divergence(expert_mf_flow=expert.mf_flow, expert_p_flow=expert.p_flow)

        # 记录结果
        new_results = pd.DataFrame([
            {'run': run, 'game_play': num_game_play, 'method': 'MFIRL', 'return': abs(mfirl_expected_return - expert.expected_return), 'Dev. MF': mfirl_dev_mf, 'Dev. Policy': mfirl_dev_p},
            {'run': run, 'game_play': num_game_play, 'method': 'PIIRL', 'return': abs(piirl_expected_return - expert.expected_return), 'Dev. MF': piirl_dev_mf, 'Dev. Policy': piirl_dev_p},
            {'run': run, 'game_play': num_game_play, 'method': 'EXPERT', 'return': expert.expected_return, 'Dev. MF': 0.0, 'Dev. Policy': 0.0}
        ])
        results = pd.concat([results, new_results], ignore_index=True)

# 保存结果
results.to_csv(results_save_path + 'results.csv', index=False)

'''

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
from Algorithms.NPIIRL import NPIIRL
from Algorithms.test import TEST


if __name__ == '__main__':
    arglist = arguments.parse_args()

    # 准备输出目录
    results_save_path = (
        arglist.save_results_dir
        + 'new_dynamics/'
        + arglist.env_name
        + '/'
        + str(arglist.num_traj)
        + '/'
        + str(arglist.time)
        + '/'
    )
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)

    # 初始化环境
    arglist.is_original_dynamics = 1  # new dynamics
    env = Environments.load(arglist.env_name).Env(arglist.is_original_dynamics, arglist.beta)

    results = pd.DataFrame(columns=['samples', 'method', 'Difference_return', 'Dev.MF', 'Dev.Policy'])

    # 初始化专家
    expert = Expert(env=env, horizon=arglist.horizon)
    expert.compute_ermfne()

    for _ in range(5):
        for run in range(10):
            print('================== Run #' + str(run) + '=========================')

            for num_game_play in range(1, arglist.max_num_game_plays + 1):
                # 初始化推断器
                piirl = TEST(
                    data_expert=None,
                    env=env,
                    horizon=arglist.horizon,
                    device=arglist.device,
                    num_tra=arglist.num_traj,
                    num_game=arglist.max_num_game_plays
                )
                npimfirl = NPIIRL(
                    data_expert=None,
                    env=env,
                    horizon=arglist.horizon,
                    device=arglist.device,
                    num_tra=arglist.num_traj,
                    num_game=arglist.max_num_game_plays
                )

                # 加载模型时显式指定 map_location="cuda:0"
                npimfirl.load_model(
                    arglist.save_model_dir
                    + 'original_dynamics/ISINGN/10/2024_12_20_145445/'
                    + 'npimfirl_'
                    + str(num_game_play)
                    + '_'
                    + str(run)
                    + '.pt',
                    map_location="cuda:0"  # <-- 新增 map_location
                )
                piirl.load_model(
                    arglist.save_model_dir
                    + 'original_dynamics/ISINGN/10/2024_12_20_145445/'
                    + 'pimfirl_'
                    + str(num_game_play)
                    + '_'
                    + str(run)
                    + '.pt',
                    map_location="cuda:0"  # <-- 新增 map_location
                )

                # 计算与专家的差异
                npimfirl_expected_return, npimfirl_dev_mf, npimfirl_dev_p = npimfirl.divergence(
                    expert_mf_flow=expert.mf_flow, expert_p_flow=expert.p_flow
                )
                piirl_expected_return, piirl_dev_mf, piirl_dev_p = piirl.divergence(
                    expert_mf_flow=expert.mf_flow, expert_p_flow=expert.p_flow
                )

                # 记录结果
                new_data = pd.DataFrame(
                    [
                        [
                            run,
                            'PIIRL',
                            abs(float(expert.expected_return) - float(piirl_expected_return)),
                            float(piirl_dev_mf),
                            float(piirl_dev_p),
                        ],
                        [
                            run,
                            'NPIFIRL',
                            abs(float(expert.expected_return) - float(npimfirl_expected_return)),
                            float(npimfirl_dev_mf),
                            float(npimfirl_dev_p),
                        ],
                        [run, 'EXPERT', 0.0, 0.0, 0.0],
                    ],
                    columns=['samples', 'method', 'Difference_return', 'Dev.MF', 'Dev.Policy']
                )
                results = pd.concat([results, new_data], ignore_index=True)

    # 统计
    grouped = results.groupby('method')
    mean_values = grouped[['Difference_return', 'Dev.MF', 'Dev.Policy']].mean().reset_index()
    var_values  = grouped[['Difference_return', 'Dev.MF', 'Dev.Policy']].var().reset_index()

    mean_values['samples'] = 'mean'
    var_values['samples']  = 'var'

    result_mean_var = pd.concat([results, mean_values, var_values], ignore_index=True)
    result_mean_var.to_csv(results_save_path + 'mean_var.csv')




# import time
# import os
#
# import numpy
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
#
# import arguments
# import Environments
# from Algorithms.PIIRL import PIIRL
# from Algorithms.expert_training import Expert
# from Algorithms.NPIIRL import NPIIRL
# from Algorithms.test import TEST
#
#
#
# if __name__ == '__main__':
#     # arglist = arguments.parse_args()
#     #
#     # # prepare the output dictionary
#     # results_save_path = arglist.save_results_dir + 'new_dynamics/' + arglist.env_name + '/' + arglist.num_traj + '/' + arglist.time + '/'
#     # if not os.path.exists(results_save_path):
#     #     os.makedirs(results_save_path)
#     #
#
#     # include all the arguments
#     arglist = arguments.parse_args()
#
#     # prepare the output dictionary
#     results_save_path = arglist.save_results_dir +  'new_dynamics/' + arglist.env_name + '/' + str(
#         arglist.num_traj) + '/' + str(arglist.time) + '/'
#     # If not exist, create new one
#     if not os.path.exists(results_save_path):
#         os.makedirs(results_save_path)
#
#     # initialise environment
#     arglist.is_original_dynamics = 1  # new dynamics
#     env = Environments.load(arglist.env_name).Env(arglist.is_original_dynamics,arglist.beta)
#     results = pd.DataFrame(columns=['samples', 'method', 'Difference_return', 'Dev.MF', 'Dev.Policy'])
#     # mfirl = MFAIRL(data=None, env=env, horizon=arglist.horizon, device=arglist.device)
#     # mdpmfgirl = PLIRL(data=None, env=env, horizon=arglist.horizon, device=arglist.device)
#
#     # initialise the expert
#     expert = Expert(env=env, horizon=arglist.horizon)
#     expert.compute_ermfne()
#
#     for run in range(10):
#         print('================== Run #' + str(run) + '=========================')
#         # load model
#         # piirl = TEST(data_expert=None, env=env, horizon=arglist.horizon, device=arglist.device,
#         #              num_tra=arglist.num_traj, num_game=arglist.max_num_game_plays)
#         #
#         # npimfirl = NPIIRL(data_expert=None, env=env, horizon=arglist.horizon, device=arglist.device,
#         #                 num_tra=arglist.num_traj, num_game=arglist.max_num_game_plays)
#         #
#         # npimfirl.load_model(arglist.save_model_dir+ 'original_dynamics/CARS/10/2024_12_09_145459/' + 'npimfirl_' + str(arglist.max_num_game_plays) + '_' + str(run) + '.pt')
#         # piirl.load_model(arglist.save_model_dir + 'original_dynamics/CARS/10/2024_12_09_145459/' + 'pimfirl_' + str(arglist.max_num_game_plays) + '_' + str(run) + '.pt')
#
#         for num_game_play in range(1, arglist.max_num_game_plays + 1):
#             # 加载模型
#             piirl = TEST(data_expert=None, env=env, horizon=arglist.horizon, device=arglist.device,
#                          num_tra=arglist.num_traj, num_game=arglist.max_num_game_plays)
#
#             npimfirl = NPIIRL(data_expert=None, env=env, horizon=arglist.horizon, device=arglist.device,
#                               num_tra=arglist.num_traj, num_game=arglist.max_num_game_plays)
#
#             npimfirl.load_model(
#                 arglist.save_model_dir + 'original_dynamics/ISINGN/10/2024_12_20_145445/' + 'npimfirl_' + str(
#                     num_game_play) + '_' + str(run) + '.pt')
#             piirl.load_model(arglist.save_model_dir + 'original_dynamics/ISINGN/10/2024_12_20_145445/' + 'pimfirl_' + str(
#                 num_game_play) + '_' + str(run) + '.pt')
#
#             # compute ERMFNE
#             npimfirl_expected_return, npimfirl_dev_mf, npimfirl_dev_p = npimfirl.divergence(expert_mf_flow=expert.mf_flow,
#                                                                                 expert_p_flow=expert.p_flow)
#             piirl_expected_return, piirl_dev_mf, piirl_dev_p = piirl.divergence(expert_mf_flow=expert.mf_flow,
#                                                                                 expert_p_flow=expert.p_flow)
#
#         # # compute ERMFNE
#         # npimfirl_expected_return, npimfirl_dev_mf, npimfirl_dev_p = npimfirl.divergence(expert_mf_flow=expert.mf_flow, expert_p_flow=expert.p_flow)
#         # piirl_expected_return, piirl_dev_mf, piirl_dev_p = piirl.divergence(expert_mf_flow=expert.mf_flow, expert_p_flow=expert.p_flow)
#
#             new_data = pd.DataFrame(
#                 [[run, 'PIIRL', abs(float(expert.expected_return) - float(piirl_expected_return)),
#                   float(piirl_dev_mf), float(piirl_dev_p)],
#                  [run, 'NPIFIRL', abs(float(expert.expected_return) - float(npimfirl_expected_return)),
#                   float(npimfirl_dev_mf), float(npimfirl_dev_p)],
#                  [run, 'EXPERT', 0.0, 0.0, 0.0]],
#                 columns=['samples', 'method', 'Difference_return', 'Dev.MF', 'Dev.Policy'])
#
#             results = pd.concat([results, new_data], ignore_index=True)
#
#         # results = results.append(
#         #     pd.DataFrame([[run, 'NPIIRL', float(npimfirl_expected_return), float(npimfirl_dev_mf), float(npimfirl_dev_p)],
#         #                   [run, 'PIIRL', float(piirl_expected_return), float(piirl_dev_mf), float(piirl_dev_p)]],
#         #                  columns=['samples', 'method', 'return', 'Dev. MF', 'Dev. Policy']))
#         # results = results.append(pd.DataFrame([[run, 'EXPERT', float(expert.expected_return), 0.0, 0.0]],
#         #                                       columns=['run', 'method', 'return', 'Dev. MF', 'Dev. Policy']))
#
#     # 对 method 分组
#     grouped = results.groupby('method')
#
#     # 计算均值和方差
#     mean_values = grouped[['Difference_return', 'Dev.MF', 'Dev.Policy']].mean().reset_index()
#     var_values = grouped[['Difference_return', 'Dev.MF', 'Dev.Policy']].var().reset_index()
#
#     # 为均值和方差结果添加标记，以区分于原数据
#     mean_values['samples'] = 'mean'
#     var_values['samples'] = 'var'
#
#     # 将均值与方差数据添加回 results
#     result_mean_var = pd.concat([results, mean_values, var_values], ignore_index=True)
#
#     # results.to_csv(results_save_path + 'results.csv')
#     result_mean_var.to_csv(results_save_path + 'mean_var.csv')


'''

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

'''

import logging
import os
import pickle
import time
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool

import arguments
import Environments
from Algorithms.NPIIRL import NPIIRL
from Algorithms.PIIRL import PIIRL
from Algorithms.expert_training import Expert
from Algorithms.test import TEST
from Algorithms.mfairl import MFAIRL
from Algorithms.plirl import PLIRL

def clear_log_file(file_path):
    with open(file_path, 'w'):
        pass

def run_single_game_play(num_of_game_plays, run, arglist, env_name, horizon, beta,
                         max_epoch, lr, max_grad_norm, num_units_1,
                         save_model_dir, save_results_dir, num_traj, is_original_dynamics):
    # 根据 run 或 num_of_game_plays 来决定使用哪块 GPU
    gpu_id = run % 4
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 在子进程中重新构造 environment 和 expert
    env = Environments.load(env_name).Env(is_original_dynamics, beta)
    expert = Expert(env=env, horizon=horizon)
    expert.compute_ermfne()

    # 生成专家轨迹
    trajectories = expert.generate_trajectories_from_policy_flow(num_of_game_plays, num_traj, expert.p_flow, expert.mf_flow)

    piirl = TEST(data_expert=trajectories, env=env, horizon=horizon, device=arglist.device,
                 num_tra=num_traj, num_game=num_of_game_plays)
    npiirl = NPIIRL(data_expert=trajectories, env=env, horizon=horizon, device=arglist.device,
                    num_tra=num_traj, num_game=num_of_game_plays)

    piirl.update_mean_field_interval = int(0.4 * max_epoch)
    npiirl.update_mean_field_interval = int(0.4 * max_epoch)

    piirl.train(max_epoch=max_epoch,
                learning_rate=lr,
                max_grad_norm=max_grad_norm,
                num_of_units=num_units_1)
    npiirl.train(max_epoch=max_epoch,
                 learning_rate=lr,
                 max_grad_norm=max_grad_norm,
                 num_of_units=num_units_1)

    piairl_expected_return, piairl_dev_mf, piairl_dev_p = piirl.divergence(expert_mf_flow=expert.mf_flow, expert_p_flow=expert.p_flow)
    npiairl_expected_return, npiairl_dev_mf, npiairl_dev_p = npiirl.divergence(expert_mf_flow=expert.mf_flow, expert_p_flow=expert.p_flow)

    new_data = pd.DataFrame(
        [[num_of_game_plays, 'PIIRL', abs(float(expert.expected_return) - float(piairl_expected_return)),
          float(piairl_dev_mf), float(piairl_dev_p)],
         [num_of_game_plays, 'NPIFIRL', abs(float(expert.expected_return) - float(npiairl_expected_return)),
          float(npiairl_dev_mf), float(npiairl_dev_p)],
         [num_of_game_plays, 'EXPERT', 0.0, 0.0, 0.0]],
        columns=['samples', 'method', 'Difference return', 'Dev. MF', 'Dev. Policy'])

    piirl.save_model(save_model_dir + 'pimfirl_' + str(num_of_game_plays) + '_' + str(run) + '.pt')
    npiirl.save_model(save_model_dir + 'npimfirl_' + str(num_of_game_plays) + '_' + str(run) + '.pt')

    return new_data

if __name__ == '__main__':

    arglist = arguments.parse_args()

    model_save_path = arglist.save_model_dir + 'original_dynamics/' + arglist.env_name + '/' + str(arglist.num_traj) + '/' + str(arglist.time) + '/'
    results_save_path = arglist.save_results_dir + 'original_dynamics/' + arglist.env_name + '/' + str(arglist.num_traj) + '/' + str(arglist.time) + '/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)

    log_file_path = os.path.join(results_save_path, 'training_log.txt')
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Log arguments
    logging.info('Arguments:')
    for arg in vars(arglist):
        logging.info(f'{arg}: {getattr(arglist, arg)}')

    # 不在主进程中创建env和expert，以免pickle问题
    # arglist.is_original_dynamics = 0
    # env = Environments.load(arglist.env_name).Env(arglist.is_original_dynamics, arglist.beta)
    # expert = Expert(env=env, horizon=arglist.horizon)
    # expert.compute_ermfne()

    print(arglist.env_name)
    clear_log_file('PIMFIRL_training_log.txt')

    results = pd.DataFrame(columns=['samples', 'method', 'Difference return', 'Dev. MF', 'Dev. Policy'])

    for run in range(arglist.num_runs):
        print('===============Run: #' + str(run) + '================')
        # 使用多进程池并行处理num_of_game_plays
        with Pool(processes=4) as pool:  # 假设有4块GPU
            all_data = pool.starmap(run_single_game_play,
                                    [(num_of_game_plays,
                                      run,
                                      arglist,
                                      arglist.env_name,
                                      arglist.horizon,
                                      arglist.beta,
                                      arglist.max_epoch,
                                      arglist.lr,
                                      arglist.max_grad_norm,
                                      arglist.num_units_1,
                                      model_save_path,
                                      results_save_path,
                                      arglist.num_traj,
                                      arglist.is_original_dynamics
                                      ) for num_of_game_plays in range(1, arglist.max_num_game_plays + 1)])

        for d in all_data:
            results = pd.concat([results, d], ignore_index=True)

    # 保存结果
    results.to_csv(results_save_path + arglist.env_name + '.csv')

    print('===============visualisation===============')
    samples = [i * 10 for i in range(1, 100)]
    step_size = 1
    x_ticks = np.arange(1, arglist.max_num_game_plays + 1, step_size)

    g1 = sns.relplot(
        x="samples",
        y="Difference return",
        data=results,
        kind="line",
        hue="method",
    )
    plt.ylabel("Difference return")
    plt.xlabel("game plays")
    g1.set(xlim=(1, arglist.max_num_game_plays))
    plt.xticks(x_ticks)
    plt.title(arglist.env_name)
    fig = plt.gcf()
    fig.savefig(results_save_path + 'reward.png')

    sns.set(style="darkgrid", font_scale=2.0)
    g1 = sns.relplot(
        x="samples",
        y="Dev. MF",
        data=results,
        kind="line",
        hue="method",
    )
    plt.ylabel("Dev. MF")
    plt.xlabel("game plays")
    g1.set(xlim=(1, arglist.max_num_game_plays))
    plt.xticks(x_ticks)
    plt.title(arglist.env_name)
    fig = plt.gcf()
    fig.savefig(results_save_path + 'mf.png')

    plt.clf()
    sns.set(style="darkgrid", font_scale=2.0)
    g2 = sns.relplot(
        x="samples",
        y="Dev. Policy",
        data=results,
        kind="line",
        hue="method",
    )
    plt.ylabel("Dev. Policy")
    plt.xlabel("game plays")
    plt.xticks(x_ticks)
    plt.title(arglist.env_name)
    fig = plt.gcf()
    fig.savefig(results_save_path + 'p.png')

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
from Algorithms.expert_training import Expert
from Algorithms.mfairl import MFAIRL
from Algorithms.plirl import PLIRL


if __name__ == '__main__':
    arglist = arguments.parse_args()
    arglist.beta = 0.01 # an extremely small temperature to approximate an ord MFNE
    arglist.env_name = 'MALWARE'
    # prepare the output dictionary
    model_save_path = arglist.save_model_dir + 'bias_verification/' + arglist.env_name + '/'+ arglist.num_traj + '/' + arglist.time + '/'
    results_save_path = arglist.save_results_dir + 'bias_verification/' + arglist.env_name + '/'+ arglist.num_traj + '/' + arglist.time + '/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)

    # initialise environment
    arglist.is_original_dynamics = 0  # original dynamics
    env = Environments.load(arglist.env_name).Env(arglist.is_original_dynamics, arglist.beta)
    ord_expert = Expert(env=env, horizon=arglist.horizon)
    ord_expert.compute_ermfne()

    opt_expert = Expert(env=env, horizon=arglist.horizon)
    opt_expert.load_opt_mfe(path='../DDPG4MFG/output_demos/')

    # performance comparison with varying number of samples
    results = pd.DataFrame(columns=['samples', 'method', 'return', 'Dev. MF', 'Dev. Policy'])
    for run in range(arglist.num_runs): #arglist.num_runs
        print('===============Run: #' + str(run) + '================')
        num_of_game_plays = 1
        while num_of_game_plays <= arglist.max_num_game_plays:
            print('Game Palys: ' + str(num_of_game_plays))
            ord_trajectories = ord_expert.generate_trajectories_MDP(num_of_game_plays, arglist.num_agents)
            opt_trajectories = opt_expert.generate_trajectories_MDP(num_of_game_plays, arglist.num_agents)

            ord_mdpmfgirl = MFAIRL(data=ord_trajectories, env=env, horizon=arglist.horizon, device=arglist.device)
            opt_mdpmfgirl = PLIRL(data=opt_trajectories, env=env, horizon=arglist.horizon, device=arglist.device)

            ord_mdpmfgirl.train(max_epoch=arglist.max_epoch,
                        learning_rate=arglist.lr,
                        max_grad_norm=arglist.max_grad_norm,
                        num_of_units=arglist.num_units_1)

            opt_mdpmfgirl.train(max_epoch=arglist.max_epoch,
                            learning_rate=arglist.lr,
                            max_grad_norm=arglist.max_grad_norm,
                            num_of_units=arglist.num_units_1)

            print('=training ends')

            ord_expected_return, ord_dev_mf, ord_dev_p = ord_mdpmfgirl.divergence(expert_mf_flow=opt_expert.mf_flow, 
                                                                                                      expert_p_flow=opt_expert.p_flow)
            opt_expected_return, opt_dev_mf, opt_dev_p = opt_mdpmfgirl.divergence(expert_mf_flow=opt_expert.mf_flow,
                                                                                  expert_p_flow=opt_expert.p_flow)
            results = results.append(pd.DataFrame([[num_of_game_plays, 'ord_mdpmfgirl', float(ord_expected_return)],
                                                   [num_of_game_plays, 'opt_mdpmfgirl', float(opt_expected_return)],
                                                   [num_of_game_plays, 'ord_EXPERT', float(ord_expert.expected_return)],
                                                   [[num_of_game_plays, 'opt_EXPERT', float(opt_expert.expected_return),]]],
                                                  columns=['samples', 'method', 'return']))

            ord_mdpmfgirl.save_model(model_save_path + 'ord_' + str(num_of_game_plays) + '_' + str(run) + '.pt')
            opt_mdpmfgirl.save_model(model_save_path + 'opt_' + str(num_of_game_plays) + '_' + str(run) + '.pt')
            num_of_game_plays += 1

    # save results
    results.to_csv(results_save_path + arglist.env_name + '.csv')

    # visualisation
    print('===============visualisation===============')

    # figure for expected return
    samples = [i * 10 for i in range(1, 100)]
    sns.set(style="darkgrid", font_scale=2.0)
    g1 = sns.relplot(
        x="samples",
        y="return",
        data=results,
        kind="line",
        hue="method",
    )
    #g.set(ylim=(-16.5,-13.7))
    plt.ylabel("Expected Return")
    plt.xlabel("game plays")
    g1.set(xlim=(1, 10))
    x_ticks = np.arange(1, 11, 1)
    plt.xticks(x_ticks)
    plt.title(arglist.env_name)
    #plt.figure(figsize=(5, 3))
    fig = plt.gcf()
    #fig.show(g)
    fig.savefig(results_save_path + 'reward.pdf')





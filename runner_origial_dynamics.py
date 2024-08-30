import os

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import arguments
import Environments
from Algorithms.PIAIRL import PIAIRL
from Algorithms.expert_training import Expert
from Algorithms.mfairl import MFAIRL
from Algorithms.plirl import PLIRL



def clear_log_file(file_path):
    # Open the file in write mode to clear its contents
    with open(file_path, 'w'):
        pass  # Simply opening the file in write mode will clear its contents


if __name__ == '__main__':

    # include all the arguments
    arglist = arguments.parse_args()

    # prepare the output dictionary
    model_save_path = arglist.save_model_dir + 'original_dynamics/' + arglist.env_name + '/' + str(arglist.num_traj) + '/' + str(arglist.time) + '/'
    results_save_path = arglist.save_results_dir + 'original_dynamics/' + arglist.env_name + '/' + str(arglist.num_traj) + '/' + str(arglist.time) + '/'
    # If not exist, create new one
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)

    # initialise environment
    arglist.is_original_dynamics = 0  # original dynamics
    # Create the Environment
    env = Environments.load(arglist.env_name).Env(arglist.is_original_dynamics, arglist.beta)

    # train the expert
    '''
    After this, we have the expert policy_flow and mean_field_flow
    '''
    expert = Expert(env=env, horizon=arglist.horizon)
    #exper_data = expert.generate_trajectories(arglist.max_num_game_plays, arglist.num_traj)
    expert.compute_ermfne()

    clear_log_file('PIAIRL_training_log.txt')


    # performance comparison with varying number of samples
    results = pd.DataFrame(columns=['samples', 'method', 'return', 'Dev. MF', 'Dev. Policy'])
    for run in range(arglist.num_runs): #arglist.num_runs
        print('===============Run: #' + str(run) + '================')
        num_of_game_plays = 1
        while num_of_game_plays <= arglist.max_num_game_plays:
            print('Game Palys: ' + str(num_of_game_plays))
            # FIXME We do not have the value for num_agents
            # FIXME we only have the value for num of trajectories
            # Change from arglist.num_agent to arglist.num_traj
            trajectories = expert.generate_trajectories(num_of_game_plays, arglist.num_traj)
            # trajectoriesMDP = expert.generate_trajectories_MDP(num_of_game_plays, arglist.num_traj)




            mfirl = MFAIRL(data=trajectories, env=env, horizon=arglist.horizon, device=arglist.device)
            # mdpmfgirl = PLIRL(data=trajectoriesMDP, env=env, horizon=arglist.horizon, device=arglist.device)

            piairl = PIAIRL(data_expert = trajectories, env=env, horizon=arglist.horizon, device=arglist.device, num_of_game_plays=num_of_game_plays, num_traj= arglist.num_traj)

            mfirl.train(max_epoch=arglist.max_epoch,
                        learning_rate=arglist.lr,
                        max_grad_norm=arglist.max_grad_norm,
                        num_of_units=arglist.num_units_1)

            # mdpmfgirl.train(max_epoch=arglist.max_epoch,
            #                 learning_rate=arglist.lr,
            #                 max_grad_norm=arglist.max_grad_norm,
            #                 num_of_units=arglist.num_units_1)

            piairl.train_reward_model(max_epoch=arglist.max_epoch,
                        learning_rate=arglist.lr,
                        max_grad_norm=arglist.max_grad_norm,
                        num_of_units=arglist.num_units_1)

            print('=training ends')

            mfirl_expected_return, mfirl_dev_mf, mfirl_dev_p = mfirl.divergence(expert_mf_flow=expert.mf_flow, expert_p_flow=expert.p_flow)
            # mdp_expected_return, mdp_dev_mf, mdp_dev_p = mdpmfgirl.divergence(expert_mf_flow=expert.mf_flow, expert_p_flow=expert.p_flow)
            # results = results._append(pd.DataFrame([[num_of_game_plays, 'MFIRL', float(mfirl_expected_return), float(mfirl_dev_mf), float(mfirl_dev_p)],
            #                                        [num_of_game_plays, 'MDPMFG-IRL', float(mdp_expected_return), float(mdp_dev_mf), float(mdp_dev_p)],
            #                                        [num_of_game_plays, 'EXPERT', float(expert.expected_return), 0.0, 0.0]],
            #                                       columns=['samples', 'method', 'return', 'Dev. MF', 'Dev. Policy']))

            mfirl.save_model(model_save_path + 'mfirl_' + str(num_of_game_plays) + '_' + str(run) + '.pt')
            # mdpmfgirl.save_model(model_save_path + 'mdp_' + str(num_of_game_plays) + '_' + str(run) + '.pt')

            piairl_expected_return, piairl_dev_mf, piairl_dev_p = piairl.divergence(expert_mf_flow=expert.mf_flow,expert_p_flow=expert.p_flow)
            results = results._append(pd.DataFrame(
                [[num_of_game_plays, 'PIAIRL', float(piairl_expected_return), float(piairl_dev_mf), float(piairl_dev_p)],
                 [num_of_game_plays, 'MFIRL', float(mfirl_expected_return), float(mfirl_dev_mf), float(mfirl_dev_p)],
                                                        [num_of_game_plays, 'EXPERT', float(expert.expected_return), 0.0, 0.0]],
                                                       columns=['samples', 'method', 'return', 'Dev. MF', 'Dev. Policy']))

            piairl.save_model(model_save_path + 'piairl_' + str(num_of_game_plays) + '_' + str(run) + '.pt')
            num_of_game_plays += 1

            print("we have done one round ")

    # save results
    results.to_csv(results_save_path + arglist.env_name + '.csv')


    # visualisation
    print('===============visualisation===============')

    # figure for expected return
    samples = [i * 10 for i in range(1, 100)]

    step_size = max(1, int((arglist.max_num_game_plays - 7) / 10))  # Adjust the divisor for fewer/more ticks
    x_ticks = np.arange(7, arglist.max_num_game_plays + 1, step_size)

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
    g1.set(xlim=(7,arglist.max_num_game_plays))
    # x_ticks = np.arange(1, 11, 1)
    plt.xticks(x_ticks,rotation=45)
    plt.title(arglist.env_name)
    #plt.figure(figsize=(5, 3))
    fig = plt.gcf()
    #fig.show(g)
    fig.savefig(results_save_path + 'reward.png')

    # figure for Dev. MF
    sns.set(style="darkgrid", font_scale=2.0)
    g1 = sns.relplot(
        x="samples",
        y="Dev. MF",
        data=results,
        kind="line",
        hue="method",
    )
    #g.set(ylim=(-16.5,-13.7))
    plt.ylabel("Dev. MF")
    plt.xlabel("game plays")
    g1.set(xlim=(7,arglist.max_num_game_plays))
    # x_ticks = np.arange(1, 11, 1)
    plt.xticks(x_ticks,rotation=45)
    plt.title(arglist.env_name)
    #plt.figure(figsize=(5, 3))
    fig = plt.gcf()
    #fig.show(g)
    fig.savefig(results_save_path + 'mf.png')

# nohup python3 -u runner_origial_dynamics.py --env_name=FLOCK --num_runs=10 --max_epoch=10 --horizon=8 > test_FLOCK.log 2>&1 &
# nohup python3 -u runner_origial_dynamics.py --env_name=MAZE --num_runs=10 --max_epoch=10 --horizon=8 > test_maze.log 2>&1 &

    # figure for Dev. Policy
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
    g1.set(xlim=(7,arglist.max_num_game_plays))
    plt.xticks(x_ticks,rotation=45)
    plt.title(arglist.env_name)
    fig = plt.gcf()
    fig.savefig(results_save_path + 'p.png')




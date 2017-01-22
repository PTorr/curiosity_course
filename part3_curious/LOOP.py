import learner as lnr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import style
import string
style.use('ggplot')

def main():
    # for the first time intialize random weights for the synapses.
    # otherwise use the output synapse from run (N) for run (N+1).

    # for first run
    hl = {1: [50, 100, 140]}
    hls = hl[1]
    input_size = 2
    num_of_iterations = 1000
    # mass = [0.145, 0.50, 0.05, 1, 1.50 , 2]
    mass = np.linspace(0.01,2,31)
    # mass = [0.145]
    frame = 0

    actions = actions_creator()

    act_mat = 1000*np.ones([len(actions),len(actions)])
    R_mat = np.zeros([len(actions), len(actions)])
    q_mat = np.zeros([len(actions), len(actions)])

    errors_vec = 1000*np.ones([len(actions)])


    # for other runs
    training_error = np.zeros([len(mass), num_of_iterations, 2])
    # synapse_0, synapse_1, synapse_2, synapse_3 = lnr.initialize_synapses(hls, input_size)
    mass_std = np.zeros(len(mass))
    for m in range(len(mass)):
        actions_count = np.zeros(len(actions))
        # print m
        synapse_0, synapse_1, synapse_2, synapse_3 = lnr.initialize_synapses(hls, input_size)
        for i in range(num_of_iterations):
            training_error[m,i,0] = i
            if i == 0:
                current_action = np.random.randint(len(actions))
                prev_action = current_action

            action = actions[current_action]
            actions_count[current_action] += 1
            l4_error, training_error[m,i,1], synapse_0, synapse_1, synapse_2, synapse_3 = lnr.learner(synapse_0, synapse_1, synapse_2, synapse_3, action, mass[m])
            # print action, l4_error**2
            act_mat, next_action = critic_actor_v1(prev_action, current_action, act_mat, l4_error)
            prev_action = current_action
            current_action = next_action
            # if m == 50: TODO: random learner
            #     current_action = np.random.randint(100)
            # if i % 1000 == 999 and m == 0:
            #     st = '{0}, {1}'.format(mass[m],i)
            #     plt.figure(st)
            #     ax = plt.subplot2grid((1, 1), (0, 0))
            #     ax.bar(np.linspace(0, len(actions), len(actions)), actions_count)
            # print actions_count
        mass_std[m] = np.std(actions_count)
    plt.figure('errors')
    colormap = plt.cm.gist_ncar
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.99, 10)])

    for m in range(len(mass)):
        if m%15 == 0:
            c = ['r','b','g','y','k','m']

            # plt.figure('m = '+str(mass[m])+'kg')
            pl = 'm = %.2fkg' %(mass[m])
            plt.plot(training_error[m,:, 0], training_error[m,:, 1],label=pl)
            # plt.plot(training_error[m, :, 0], training_error[m, :, 1], label='m = ' + str(mass[m]) + 'kg')
            plt.xlabel('Steps')
            plt.ylabel('Loss function')
            plt.ylim([0,40])
            image_names = 'graphs'
            save = False
            if save:
                for t in range(10):
                    plt.savefig('./figs1/' + image_names + string.zfill(str(frame), 5) + '.png', format='png')
                    frame += 1
    plt.legend()

    plt.figure('std vs. mass')
    plt.scatter(mass,mass_std)
    plt.show()

def actions_creator():
    theta = np.linspace(0, 90, 10)
    theta = theta * np.pi/180
    energy = np.linspace(5, 25, 10)

    lt = len(theta)
    le = len(energy)
    actions = np.zeros([lt*le,2])
    k = 0
    for i in range(lt):
        for j in range(le):
            actions[k,:] = [energy[j], theta[i]]
            k += 1

    return actions

def actor_critic(errors_vec,idx,error=None):
    errors_vec[idx] = error**2
    max_idx = np.where(errors_vec == np.max(errors_vec))
    max_idx = max_idx[0]
    m_idx = np.random.randint(len(max_idx))
    next_policy = max_idx[m_idx]
    return errors_vec, next_policy

def critic_v1(errors_vec,idx,error=None):
    errors_vec[idx] = error**2
    max_idx = np.where(errors_vec == np.max(errors_vec))
    max_idx = max_idx[0]
    m_idx = np.random.randint(len(max_idx))
    next_policy = max_idx[m_idx]
    return errors_vec, next_policy

def critic_actor_v1(prev_action,current_action,act_mat,error):
    act_mat_old = act_mat
    act_mat[prev_action,current_action] = error**2
    max_idx = np.where(act_mat[current_action,:] == np.max(act_mat[current_action]))
    max_idx = max_idx[0]
    m_idx = np.random.randint(len(max_idx))
    next_policy = max_idx[m_idx]
    # epsilon greedy, 10% will be random choice
    if np.random.randint(100) < 10:
        next_policy = np.random.randint(100)
    return act_mat, next_policy

def critic_actor_q(prev_action,current_action,R_mat,q_mat,error):
    gamma = 0.8
    R_mat[prev_action,current_action] = error**2
    max_idx = np.where(R_mat[current_action,:] == np.max(R_mat[current_action]))
    max_idx = max_idx[0]
    m_idx = np.random.randint(len(max_idx))
    next_action = max_idx[m_idx]
    # epsilon greedy, 10% will be random choice
    if np.random.randint(100) < 200:
        next_action = np.random.randint(100)
    q_mat[prev_action, current_action] = R_mat[prev_action, current_action] + gamma * np.max(q_mat[next_action, :])
    return R_mat, next_action


if __name__ == '__main__':
    main()
import learner as lnr
import numpy as np
import matplotlib.pyplot as plt

def main():
    # for the first time intialize random weights for the synapses.
    # otherwise use the output synapse from run (N) for run (N+1).

    # for first run
    hl = {1: [50, 100, 140]}
    hls = hl[1]
    input_size = 2
    num_of_iterations = 1000
    mass = [0.145, 0.50, 0.100, 0.200, 0.250]

    actions = actions_creator()

    act_mat = 1000*np.ones([len(actions),len(actions)])

    errors_vec = 1000*np.ones([len(actions)])


    # for other runs
    training_error = np.zeros([len(mass), num_of_iterations, 2])
    for m in range(len(mass)):
        synapse_0, synapse_1, synapse_2, synapse_3 = lnr.initialize_synapses(hls, input_size)
        for i in range(num_of_iterations):
            training_error[m,i,0] = i
            if i == 0:
                current_action = np.random.randint(len(actions))
                prev_action = current_action

            action = actions[current_action]
            l4_error, training_error[m,i,1], synapse_0, synapse_1, synapse_2, synapse_3 = lnr.learner(synapse_0, synapse_1, synapse_2, synapse_3, action, mass[m])
            act_mat, next_action = critic_actor_v1(prev_action, current_action, act_mat, l4_error)
            prev_action = current_action
            current_action = next_action
    plt.figure('errors')
    for m in range(len(mass)):
        c = ['r','b','g','y','k']
        plt.plot(training_error[m,:, 0], training_error[m,:, 1], c[m], label='train')
    plt.show()
    print 't'
def actions_creator():
    theta = np.linspace(0, 90, 3)
    theta = theta * np.pi/180
    energy = np.linspace(5, 25, 3)

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
    act_mat[prev_action,current_action] = error**2
    max_idx = np.where(act_mat[current_action,:] == np.max(act_mat[current_action]))
    max_idx = max_idx[0]
    m_idx = np.random.randint(len(max_idx))
    next_policy = max_idx[m_idx]
    return act_mat, next_policy


if __name__ == '__main__':
    main()
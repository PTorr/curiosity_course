import learner as lnr
import numpy as np

def main():
    # for the first time intialize random weights for the synapses.
    # otherwise use the output synapse from run (N) for run (N+1).

    # for first run
    hl = {1: [50, 100, 140]}
    hls = hl[1]
    input_size = 2
    num_of_iterations = 10000

    policies = policies_creator()
    errors_vec = 1000*np.ones([len(policies)])

    synapse_0, synapse_1, synapse_2, synapse_3 = lnr.initialize_synapses(hls, input_size)

    # for other runs
    for i in range(num_of_iterations):
        if i == 0:
            idx = np.random.randint(len(policies))
            action = policies[idx]
        l4_error, synapse_0, synapse_1, synapse_2, synapse_3 = lnr.learner(synapse_0, synapse_1, synapse_2, synapse_3, action)
        errors_vec, next_policy = actor_critic(errors_vec, idx, l4_error)
        action = policies[next_policy]
        # print l4_error

def policies_creator():
    theta = np.linspace(0, 90, 3)
    theta = theta * np.pi/180
    energy = np.linspace(5, 25, 3)

    lt = len(theta)
    le = len(energy)
    policies = np.zeros([lt*le,2])
    k = 0
    for i in range(lt):
        for j in range(le):
            policies[k,:] = [energy[j], theta[i]]
            k += 1

    return policies

def actor_critic(errors_vec,idx,error=None):
    errors_vec[idx] = error**2
    max_idx = np.where(errors_vec == np.max(errors_vec))
    max_idx = max_idx[0]
    m_idx = np.random.randint(len(max_idx))
    next_policy = max_idx[m_idx]
    return errors_vec, next_policy




if __name__ == '__main__':
    main()
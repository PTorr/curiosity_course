import learner as lnr

# for the first time intialize random weights for the synapses.
# otherwise use the output synapse from run (N) for run (N+1).

# for first run
hl = {1: [50, 100, 140]}
hls = hl[1]
input_size = 2
synapse_0, synapse_1, synapse_2, synapse_3 = lnr.initialize_synapses(hls, input_size)

# for other runs
for i in range(1000):
    l4_error, synapse_0, synapse_1, synapse_2, synapse_3 = lnr.learner(synapse_0, synapse_1, synapse_2, synapse_3)
    # print l4_error
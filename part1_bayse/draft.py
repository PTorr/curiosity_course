import numpy as np
import data_generator as data_generator

def main():
    e = 15
    e = data_generator.energy_with_noise(e)
    theta = 45
    theta = theta*np.pi/180
    v0 = data_generator.initial_velocity(e)
    time = data_generator.time_calculator(v0, theta)
    d = data_generator.distance_calculator(v0, theta, time)
    print d

def distribution_calculator():
    [data, energies, angles, distances] = data_generator.generator(1)
    total_counts = sum(data[:,3])
    i = 0
    dd = distances_distribution(data,distances[10])
    de = energies_distribution(dd,energies[2])
    da = angles_distribution(de,angles[3])

    print sum(dd[:,3])/total_counts, sum(de[:,3])/total_counts, sum(da[:,3])/total_counts


def energies_distribution(d,wanted_energy):
    de = d[d[:, 0] == wanted_energy]
    return de
def angles_distribution(d,wanted_angle):
    da = d[d[:, 1] == wanted_angle]
    return da
def distances_distribution(d,wanted_distance):
    dd = d[d[:, 2] == wanted_distance]
    return dd

if __name__ == '__main__':
    main()

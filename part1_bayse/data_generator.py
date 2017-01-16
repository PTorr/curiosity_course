import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

def generator(plt_value):
    P = {}
    for i in range(0,10000):
        theta = choose_theta()
        temp, energy = choose_energy()
        v0 = initial_velocity(energy)
        time = time_calculator(v0,theta)
        distance = distance_calculator(v0,theta,time)
        P_key = str(energy[0])+'+'+str(theta[0])+'+'+str(distance)
        if P_key in P:
            P[P_key] += 1
        else:
            P[P_key] = 1

    data = np.empty([len(P), 4])
    i = 0
    for p in P:
        temp = p.split('+')
        #                           E       theta   D
        data[i][:] = np.asarray([temp[0],temp[1],temp[2],P[p]])
        i += 1

    energies  = np.unique(data[:,0])
    angles    = np.unique(data[:,1])
    distances = np.unique(data[:,2])

    if plt_value == 1:
        plot_data(energies, angles, distances, data)
    # print energy_fixed_angle_probability(energies, data, angles[0], energies[0])
    print data

    return data, energies, angles, distances


def data_to_file(N, M):
    X = np.zeros([N, 2])
    y = np.zeros([N, 1])
    d = np.linspace(0, 45, M)  # 46

    e = np.linspace(5,30,100)
    t = np.linspace(0,91,100)

    for i in range(0, N):
        theta = choose_theta()
        energy_no,energy = choose_energy()
        v0 = initial_velocity(energy)
        time = time_calculator(v0, theta)
        distance = distance_calculator(v0, theta, time)
        # X[i, 0:100] = [energy_no/25, theta/90]
        X[i, 0] = energy_no
        X[i, 1] = theta
        y[i, 0] = distance

    gen_data = np.hstack((X,y))
    np.savetxt("generated_data.csv", gen_data, delimiter=",",
               header='energy, theta, distance')

def energy_fixed_angle_probability(energies,data,fixed_angle,wanted_energy):
    dd = np.empty([len(energies), 2])
    dd[:, 0] = data[data[:, 1] == fixed_angle, 0] # energy
    dd[:, 1] = data[data[:, 1] == fixed_angle, 3] # counts
    return sum(sum([dd[dd[:, 0] == wanted_energy, 1]]) / sum(dd[:, 1]))

def plot_data(energies,angles,distances,data):
    E = np.empty([len(energies),2])
    i = 0
    for e in energies:
        E[i,0] = e
        E[i,1] = np.sum(data[data[:,0] == e,3])
        i += 1

    plt.figure('Energies distribution')
    plt.bar(E[:,0],E[:,1])
    plt.xlabel('Energy[J]')
    plt.ylabel('Counts')

    A = np.empty([len(angles),2])
    i = 0
    for a in angles:
        A[i,0] = a
        A[i,1] = np.sum(data[data[:,1] == a,3])
        i += 1

    plt.figure('Angles distribution')
    plt.bar(A[:,0]*180/np.pi,A[:,1])
    plt.xlabel('Angles[degree]')
    plt.ylabel('Counts')

    D = np.empty([len(distances),2])
    i = 0
    for d in distances:
        D[i,0] = d
        D[i,1] = np.sum(data[data[:,2] == d,3])
        i += 1

    plt.figure('Distances distribution')
    plt.bar(D[:,0],D[:,1])
    plt.xlabel('Distances[m]')
    plt.ylabel('Counts')
    plt.show()


def choose_theta():
    theta = np.random.choice(np.linspace(0,90,3), 1)
    return np.round(theta*np.pi/180,2)

def choose_energy():
    energy = np.random.choice(np.linspace(5,25,3), 1)
    return energy

def energy_with_noise(energy):
    en = np.random.normal(0, 0.05*energy, 1)
    return max(0,energy+en)

def initial_velocity(energy):
    mass = 0.145
    v0 = np.sqrt(2*energy/mass)
    return v0

def initial_velocity_mass(energy,mass):
    v0 = np.sqrt(2*energy/mass)
    return v0

def time_calculator(v0,theta):
    g = 9.81
    coeff = [-g/2, v0*np.sin(theta), 1.3]
    t = np.roots(coeff)
    return t[t>0][0]

def distance_calculator(v0,theta,time):
    distance = v0*np.cos(theta)*time
    return round(distance)

if __name__ == '__main__':
    # generator(1)
    data_to_file(10000, 10)
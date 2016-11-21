import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

def main():
    P = {}
    for i in range(0,10000):
        theta = choose_theta()
        energy = choose_energy()
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

    E = np.empty([len(energies),2])
    i = 0
    for e in energies:
        E[i,0] = e
        E[i,1] = np.sum(data[data[:,0] == e,3])
        i += 1

    plt.figure('Energies distribution')
    plt.bar(E[:,0],E[:,1])

    A = np.empty([len(angles),2])
    i = 0
    for a in angles:
        A[i,0] = a
        A[i,1] = np.sum(data[data[:,1] == a,3])
        i += 1

    plt.figure('Angles distribution')
    plt.bar(A[:,0],A[:,1])

    D = np.empty([len(distances),2])
    i = 0
    for d in distances:
        D[i,0] = d
        D[i,1] = np.sum(data[data[:,2] == d,3])
        i += 1

    plt.figure('Distances distribution')
    plt.bar(E[:,0],E[:,1])
    plt.show()

    # histogram
    # X = np.arange(len(P))
    # pl.bar(X, P.values(), align='center', width=0.5)
    # pl.xticks(X, P.keys())
    # ymax = max(P.values()) + 1
    # pl.ylim(0, ymax)
    # pl.show()



def choose_theta():
    theta = np.random.choice(np.linspace(-89,90,10), 1)
    return np.round(theta*np.pi/180,2)

def choose_energy():
    energy = np.random.choice(np.linspace(0.01,25,10), 1)
    return np.round(energy_with_noise(energy))

def energy_with_noise(energy):
    en = np.random.normal(0, 0.025*energy, 1)
    return max(0,energy+en)

def initial_velocity(energy):
    mass = 0.145
    v0 = np.sqrt(2*energy/mass)
    return v0

def time_calculator(v0,theta):
    g = 9.81
    coeff = [-g/2, v0*np.sin(theta), 1.3]
    t = np.roots(coeff)
    return t[t>0][0]

def distance_calculator(v0,theta,time):
    distance = v0*np.cos(theta)*time
    return round(distance[0])

if __name__ == '__main__':
    main()
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import data_generator as data_generator


def bayes_prob():
    '''Calculates the Bayes theorem probability'''
    max_sig = 5
    max_mu = 40
    sig = np.linspace(0.01, max_sig, 75)
    mu = np.linspace(-2, max_mu, 75)

    # Initializing energy, theta, and the structures:
    #   info_mat: dkl matrix for E & theta combination.
    #   action_count: matrix of how many times it tried a combination
    #   prior_mat: matrix of the prior
    #   post: matrix of the post
    M = 3
    N = 3
    energy = np.linspace(5.0, 25.0, M)
    theta1 = np.linspace(0.0, 90.0, N)
    theta = np.round(theta1 * np.pi / 180, 2)
    info_mat = 10e1 * np.ones([M, N])
    action_count = np.zeros([M, N])
    dkl = 10*np.ones([M, N])
    prior_mat = np.ones([M, N, len(mu), len(sig)])
    # post = np.ones([len(mu), len(sig)])

    # creates the figure with the subplots
    fig = plt.figure('Energy and angle')
    ax = [[plt.subplot2grid((4, 3), (0, 0)), plt.subplot2grid((4, 3), (0, 1)), plt.subplot2grid((4, 3), (0, 2))],
          [plt.subplot2grid((4, 3), (1, 0)), plt.subplot2grid((4, 3), (1, 1)), plt.subplot2grid((4, 3), (1, 2))],
          [plt.subplot2grid((4, 3), (2, 0)), plt.subplot2grid((4, 3), (2, 1)), plt.subplot2grid((4, 3), (2, 2))],
          [plt.subplot2grid((4, 3), (3, 0), projection='3d'),plt.subplot2grid((4, 3), (3, 1))]]
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    ax[3][0].set_zlim(0, 35)
    ax[3][0].view_init(30, 15)
    ax[3][0].set_title('action count')
    ax[3][1].set_title('info mat')

    labels = []
    for m in range(M):
        for n in range(N):
            st1 = 'e'+str(energy[m])+', '+'t'+str(theta1[n])
            labels.append(st1)

    # titles for the subplots
    for m in range(M):
        for n in range(N):
            s = 'e='+str(energy[m]),'$\theta=$'+str(round(theta[n]*180/np.pi))
            ax[m][n].set_title(s)
            ax[m][n].set_xlabel('$\sigma$ [m]')
            ax[m][n].set_ylabel('$\mu$ [m]')

    # for i in range(50):
    while (np.max(np.max(info_mat)) > 0.005):
        # [nn, mm] = np.where(info_mat == np.max(np.max(info_mat)))
        # m = np.random.choice(mm)
        # n = nn[mm == m]
        # n = n[0]

        # ---------------Be curious-------------------
        # choosing the combination with the biggest DKL
        [mm, nn] = np.where(info_mat == np.max(np.max(info_mat)))
        temp_rand = np.linspace(0,len(nn)-1,len(nn))
        tempn = np.random.choice(temp_rand)
        m = mm[tempn]
        n = nn[tempn]
        # m = m1[0]

        # calculating the distance of the throw
        v0 = data_generator.initial_velocity(data_generator.energy_with_noise(energy[m]))
        time = data_generator.time_calculator(v0, theta[n])
        d = data_generator.distance_calculator(v0, theta[n], time)


        # calculating the posterior prob
        post = posterior(prior_mat[m, n, :, :], mu, sig, d)
        post[post < 10e-20] = 10e-20
        prior_mat[m, n, [prior_mat[m, n, :, :] < 10e-20]] = 10e-20
        # calculating DKL
        dkl[m,n] = sum(sum(post * np.log(post / prior_mat[m, n, :, :])))
        # updating info mat
        count_dkl_condition = 0

        # stopping condition
        if (np.max(np.abs(info_mat-dkl))<0.001):
            break
            # count_dkl_condition += 1
            # if(count_dkl_condition == 5):
            #     break
        info_mat[m, n] = dkl[m,n]
        # updating count for specific combination
        action_count[m, n] += 1

        # updating the specific combination plot
        ax[m][n].imshow(post, interpolation='none',extent=[0.01, 1.5*max_sig, max_mu,-2],aspect=0.07)
        plt.sca(ax[m][n])
        ym = np.linspace(-2, max_mu, 10)
        plt.yticks(ym)

        # updating the count bar plot
        pl3d = 1
        if pl3d == 1:
            x1, y1 = np.meshgrid(theta,energy)
            x = np.concatenate(y1)
            y = np.concatenate(x1)
            z = np.zeros(len(x))
            dx = 5 * np.ones(9)
            dy = 0.5 * np.ones(9)
            dz = np.concatenate(action_count)
            ax[3][0].bar3d(x, y, z, dx, dy, dz, color='#00ceaa')
            # ax[3][1].pcolor(x1,y1,info_mat,cmap='gist_rainbow', vmin=0, vmax=1)
            # ax[3][1].imshow(info_mat, interpolation='none', vmin=0, vmax=1)
            zz = np.concatenate(info_mat)
            xx = np.linspace(0,len(zz),len(zz))
            ax[3][1].cla()
            ax[3][1].bar(xx,zz)
            ax[3][1].set_ylim(0, 0.3)
            ax[3][1].set_title('info mat')
            ax[3][1].xaxis.set_ticks(xx+0.25)
            ax[3][1].set_xticklabels(labels, rotation=45,  fontsize=7)

        plt.pause(0.001)
        # updating the prior of specific combination with the posterior calculated above.
        prior_mat[m, n, :, :] = post

    # this is just to leave the plot open in the end
    # plt.pause(10)

    print 'finished'
    ac = sum(sum(action_count))
    ax[3][0].set_title('action count %d' % ac)
    plt.show()

def posterior(prior_mat, mu, sig, d):
    '''Calculates the posterior based on:
        likelihood_mat, prior_mat, marginal_likelihood
        input:  prior_mat: matrix of the prior base on mu and sig
                mu: specific mu of the distribution
                sig: specific sigma of the distribution
                d: distance
        output: posterior matrix'''
    likelihood_mat = np.ones([len(mu), len(sig)])

    # prior matrix calculation
    prior_mat /= sum(sum(prior_mat))
    # likelihood matrix calculation
    for i in range(0, len(mu)):
        for j in range(0, len(sig)):
            likelihood_mat[i, j] = likelihood_func(mu[i], sig[j], d)

    M = likelihood_mat * prior_mat
    marginal_likelihood = sum(sum(M))

    # posterior calculation
    posterior = M / marginal_likelihood
    return posterior

def likelihood_func(mu, sig, d):
    '''Calculates the likelihood:
        input: mu: specific mu of the distribution
               sig: specific sigma of the distribution
               d: distance
        output: likelihood probability'''
    return (np.exp(-(d - mu) ** 2.0 / (2.0 * sig ** 2.0))) / (sig * np.sqrt(2.0 * np.pi))

if __name__ == '__main__':
    bayes_prob()

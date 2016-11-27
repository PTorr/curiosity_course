import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import data_generator as data_generator

def bayes_prob():
    d = np.random.normal(25,5,size=50)

    max_sig = 10
    max_mu = 40
    sig = np.linspace(0.01, max_sig, 10)
    mu = np.linspace(0.01, max_mu, 10)

    prior_mat = np.ones([len(mu), len(sig)])

    # for i in range(len(d)):
    #     post = posterior(prior_mat,mu,sig,d[i])
    #     post[post<10e-20] = 10e-20
    #     prior_mat[prior_mat < 10e-3] = 0.01
    #     dkl = -sum(sum(post*np.log(post/prior_mat)))
    #     print dkl
    #     # plt.imshow(post, interpolation='none')
    #     # plt.pause(0.1)
    #     # fig = plt.figure('state function')
    #     # fig.plot_trisurf(sig, mu, post, cmap=cm.jet, linewidth=0.1, alpha=1)
    #     prior_mat = post

    N = 1
    M = 3
    theta = np.linspace(0, 90, N)
    energy = np.linspace(5, 25, M)
    information_vector = np.zeros([len(theta),len(energy)])

    # matrix which contain the matrices matrix of posteriors for every E and alpha

    prior_mat = np.ones([len(mu), len(sig)])
    post = np.ones([len(mu), len(sig)])

    for i in range(len(d)):
        [nn, mm] = np.where(information_vector == min(min(information_vector)))
        m = np.random.choice(mm)
        n = nn[m]

        v0 = data_generator.initial_velocity(energy[m])
        time = data_generator.time_calculator(v0, theta[n])
        d = data_generator.distance_calculator(v0, theta[n], time)

        post = posterior(prior_mat, mu, sig, d)
        post[post < 10e-20] = 10e-20
        prior_mat[prior_mat < 10e-3] = 0.01
        dkl = -sum(sum(post * np.log(post / prior_mat))) # insert into the information vector in the m,n place
        prior_mat = post

        # counter for every m,n
        # plot DKL
        # until threshold


    # ax = fig.add_subplot(1, 1, L, projection='3d')
    # X, Y = np.meshgrid(sig, mu)


def posterior(prior_mat,mu,sig,d):

    likelihood_mat = np.ones([len(mu),len(sig)])

    # prior matrix calculation
    prior_mat /= sum(sum(prior_mat))
    # likelihood matrix calculation
    for i in range(0,len(mu)):
        for j in range(0, len(sig)):
            likelihood_mat[i,j] = likelihood_func(mu[i],sig[j],d)

    M = likelihood_mat * prior_mat
    marginal_likelihood = sum(sum(M))

    # posterior calculation
    posterior = M/marginal_likelihood
    return posterior


def likelihood_func(mu,sig,d):
    return (np.exp(-(d-mu)**2.0/(2.0*sig**2.0)))/(sig*np.sqrt(2.0*np.pi))

if __name__ == '__main__':
    bayes_prob()

from scipy.special import loggamma
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import loggamma
from math import log
import collections

# This function is given, nothing to do here.


def simulate_data(num_samples, tails_proba):
    """Simulate a sequence of i.i.d. coin flips.

    Tails are denoted as 1 and heads are denoted as 0.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    tails_proba : float in range (0, 1)
        Probability of observing tails.

    Returns
    -------
    samples : array, shape (num_samples)
        Outcomes of simulated coin flips. Tails is 1 and heads is 0.
    """
    return np.random.choice([0, 1], size=(num_samples), p=[
                            1 - tails_proba, tails_proba])


def compute_log_likelihood(theta, samples):
    """Compute log p(D | theta) for the given values of theta.

    Parameters
    ----------
    theta : array, shape (num_points)
        Values of theta for which it's necessary to evaluate the log-likelihood.
    samples : array, shape (num_samples)
        Outcomes of simulated coin flips. Tails is 1 and heads is 0.

    Returns
    -------
    log_likelihood : array, shape (num_points)
        Values of log-likelihood for each value in theta.
    """
    res = np.zeros(shape=(len(theta), 1))
    count = collections.Counter(samples)
    for i, t in enumerate(theta):
        tail = log(t)
        head = log(1 - t)
        res[i][0] = tail * count[1] + head * count[0]
    return res


def compute_log_prior(theta, a, b):
    """Compute log p(theta | a, b) for the given values of theta.

    Parameters
    ----------
    theta : array, shape (num_points)
        Values of theta for which it's necessary to evaluate the log-prior.
    a, b: float
        Parameters of the prior Beta distribution.

    Returns
    -------
    log_prior : array, shape (num_points)
        Values of log-prior for each value in theta.

    """
    res = np.zeros(shape=(len(theta), 1))
    for i, t in enumerate(theta):
        res[i][0] = loggamma(a + b) - loggamma(a) - loggamma(b) + \
            (a - 1) * log(t) + (b - 1) * log(1 - t)
    return res


def compute_log_posterior(theta, samples, a, b):
    """Compute log p(theta | D, a, b) for the given values of theta.

    Parameters
    ----------
    theta : array, shape (num_points)
        Values of theta for which it's necessary to evaluate the log-prior.
    samples : array, shape (num_samples)
        Outcomes of simulated coin flips. Tails is 1 and heads is 0.
    a, b: float
        Parameters of the prior Beta distribution.

    Returns
    -------
    log_posterior : array, shape (num_points)
        Values of log-posterior for each value in theta.
    """
    count = collections.Counter(samples)
    tail = count[1]
    head = count[0]

    res = np.zeros(shape=(len(theta), 1))
    for i, t in enumerate(theta):
        res[i][0] = loggamma(tail + a + head + b) - loggamma(tail + a) - loggamma(
            head + b) + (tail + a - 1) * log(t) + (head + b - 1) * log(1 - t)
    return np.array(res)


def compute_theta_mle(samples):
    """Compute theta_MLE for the given data.

    Parameters
    ----------
    samples : array, shape (num_samples)
        Outcomes of simulated coin flips. Tails is 1 and heads is 0.

    Returns
    -------
    theta_mle : float
        Maximum likelihood estimate of theta.
    """
    count = collections.Counter(samples)
    return count[1] / len(samples)


def compute_theta_map(samples, a, b):
    """Compute theta_MAP for the given data.

    Parameters
    ----------
    samples : array, shape (num_samples)
        Outcomes of simulated coin flips. Tails is 1 and heads is 0.
    a, b: float
        Parameters of the prior Beta distribution.

    Returns
    -------
    theta_mle : float
        Maximum a posteriori estimate of theta.
    """
    count = collections.Counter(samples)
    tail = count[1]
    head = count[0]
    return (tail + a - 1) / (tail + a + head + b - 2)


if __name__ == '__main__':
    num_samples = 20
    tails_proba = 0.7
    samples = simulate_data(num_samples, tails_proba)
    a, b = 3, 5
    print(samples)
    plt.figure(figsize=[12, 8])
    x = np.linspace(1e-5, 1 - 1e-5, 1000)

    # Plot the prior distribution
    log_prior = compute_log_prior(x, a, b)
    prior = np.exp(log_prior)
    plt.plot(x, prior, label='prior')

    # Plot the likelihood
    log_likelihood = compute_log_likelihood(x, samples)
    likelihood = np.exp(log_likelihood)
    int_likelihood = np.mean(likelihood)
    # We rescale the likelihood - otherwise it would be impossible to see in
    # the plot
    rescaled_likelihood = likelihood / int_likelihood
    plt.plot(x, rescaled_likelihood, label='scaled likelihood', color='purple')

    # Plot the posterior distribution
    log_posterior = compute_log_posterior(x, samples, a, b)
    posterior = np.exp(log_posterior)
    plt.plot(x, posterior, label='posterior')

    # Visualize theta_mle
    theta_mle = compute_theta_mle(samples)
    ymax = np.exp(
        compute_log_likelihood(
            np.array(
                [theta_mle]),
            samples)) / int_likelihood
    plt.vlines(
        x=theta_mle,
        ymin=0.00,
        ymax=ymax,
        linestyle='dashed',
        color='purple',
        label=r'$\theta_{MLE}$')

    # Visualize theta_map
    theta_map = compute_theta_map(samples, a, b)
    ymax = np.exp(compute_log_posterior(np.array([theta_map]), samples, a, b))
    plt.vlines(
        x=theta_map,
        ymin=0.00,
        ymax=ymax,
        linestyle='dashed',
        color='orange',
        label=r'$\theta_{MAP}$')

    plt.xlabel(r'$\theta$', fontsize='xx-large')
    plt.legend(fontsize='xx-large')
    plt.show()

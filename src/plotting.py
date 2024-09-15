import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_mcmc_overview(file_path: str) -> None:
    """
    Provides an overview of MCMC performance. 
    
    Args:
        file_path: An absolute path to a json file.
    
    Outputs two plots:
    - theta over iterations
    - likelihood over iterations
    
    Prints basic diagnostic info. 
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    theta_chains = np.array(data['theta_chains'])
    likelihoods = np.array(data['likelihoods'])
    accept_records = np.array(data['accept_record'])
    iterations = data['iteration']  # Assuming iterations are consistent for all chains

    accept_rates = [round(chain.sum() / len(chain), ndigits=3) for chain in accept_records[:, :iterations]]
    for chain, rate in enumerate(accept_rates):
        print(f"Chain {chain + 1}: {rate} acceptance rate.")

    # Plot Theta Chains
    plt.figure(figsize=(10, 6))
    for i in range(theta_chains.shape[0]):
        sns.lineplot(x=range(iterations), y=theta_chains[i, 0, :iterations], label=f'Theta Chain {i+1}')
    plt.title('Theta Chains Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Theta Value')
    plt.legend()
    plt.show()

    # Plot Likelihoods
    plt.figure(figsize=(10, 6))
    for i in range(likelihoods.shape[0]):
        sns.lineplot(x=range(iterations), y=likelihoods[i, :iterations], label=f'Likelihood Chain {i+1}')
    plt.title('Likelihoods Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')
    plt.legend()
    plt.show()

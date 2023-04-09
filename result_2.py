"""
Script for getting result 2.
"""

# External dependencies
import matplotlib
import matplotlib.pyplot as plt
from flwr.server.history import History

# Internal dependencies
from simulation import run_simulation

def run_experiment() -> None:
    """
    Produces the graphs found in section 3.2.2 of the paper.

    On the author's processor (M1 Pro), this experiment takes ? hours.

    Arguments:
        None
    Return Values:
        None
    """
    stds = [x * 0.25 for x in range(2, 9)]

    n_bad_clients_curves = []
    for n_bad_clients in range(1, 6):
        n_bad_clients_curve = []
        for std in stds:
            n_clients = (n_bad_clients * 2) + 1
            min_n_clients = 0
            while not min_n_clients:
                try:
                    sim_config = {
                        'cmv_on': True,
                        'n_clients': n_clients,
                        'n_bad_clients': n_bad_clients,
                        'n_scrambled_labels': 5,
                        'n_rounds': 1,
                        'dataset': 'mnist',
                        'n_client_epochs': 5,
                        'client_batch_size': 128,
                        'avg_client_std_threshold': std
                    }
                    performance : History = run_simulation(sim_config)
                    print('SIMULATION DONE FOR THE FOLLOWING CONFIG:')
                    print(sim_config)
                    n_clients_used = performance.metrics_distributed[
                        'n_clients_used'][0][1]
                    if n_clients_used == (n_clients - n_bad_clients):
                        min_n_clients = n_clients
                    else:
                        n_clients += 1
                # This is for the case that Flower fails
                # because all models were rejected
                except:
                    n_clients += 1
            n_bad_clients_curve.append(min_n_clients)
            print(f'\n\nNUMBER OF CLIENTS REQUIRED FOR {n_bad_clients} BAD CLIENTS ON STD = {std}: {n_clients}\n\n')
        n_bad_clients_curves.append(n_bad_clients_curve)
        print(f'\n\nCURVE FOR {n_bad_clients} BAD CLIENTS FINISHED\n\n')

    for n_bad_clients_curve, n_bad_clients in \
        zip(n_bad_clients_curves, list(range(1, 6))):
        plt.plot(stds, n_bad_clients_curve,
                 label=f'{n_bad_clients}')
    plt.xlabel('Avg. Client STD Threshold')
    plt.ylabel('Number of Clients Required for Detection')
    plt.legend(
        title='Number of Bad Clients', fontsize=6, loc='upper left'
    ).get_title().set_fontsize(6)
    plt.title('Minimum Client Counts Required for CMV')
    plt.savefig('paper_images/min_client_counts.png', dpi=300)

# def run_experiment() -> None:
#     """
#     Produces the graphs found in section 3.2.2 of the paper.

#     On the author's processor (M1 Pro), this experiment takes ? hours.

#     Arguments:
#         None
#     Return Values:
#         None
#     """
#     results = []
#     for n_scrambled_labels in range(2, 11):
#         n_clients = 2
#         min_n_clients = 0
#         while not min_n_clients:
#             try:
#                 performance : History = run_simulation(sim_config={
#                     'cmv_on': True,
#                     'n_clients': n_clients,
#                     'n_bad_clients': 1,
#                     'n_scrambled_labels': n_scrambled_labels,
#                     'n_rounds': 1,
#                     'dataset': 'mnist',
#                     'n_client_epochs': 5,
#                     'client_batch_size': 128,
#                     'avg_client_std_threshold': 1.0
#                 })
#                 if performance.metrics_distributed[
#                     'n_clients_used'][0][1] == (n_clients - 1):
#                     min_n_clients = n_clients
#                 elif performance.metrics_distributed[
#                     'n_clients_used'][0][1] < (n_clients - 1):
#                     results.append(f'TOO MANY MODELS REJECTED WHEN ATTEMPTING TO DETECT 1 BAD CLIENT WITH {n_scrambled_labels}/10 SCRAMBLED LABELS AMONG {n_clients} CLIENTS ({n_clients - performance.metrics_distributed["n_clients_used"][0][1]})')
#                     continue
#                 else:
#                     n_clients += 1
#                     continue
#             # This is for the case that Flower fails
#             # because all models were rejected
#             except:
#                 n_clients += 1
#                 continue
#             print(f'\n\nMIN # OF CLIENTS TO DETECT 1 BAD CLIENT WITH {n_scrambled_labels}/10 SCRAMBLED LABELS: {min_n_clients}\n\n')
#             results.append(f'MIN # OF CLIENTS TO DETECT 1 BAD CLIENT WITH {n_scrambled_labels}/10 SCRAMBLED LABELS: {min_n_clients}')

#     for result in results:
#         print(result)

if __name__ == '__main__':
    # This prevents matplotlib from producing pop-ups
    matplotlib.use('Agg')
    run_experiment()

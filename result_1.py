"""
Script for getting result 1.
"""

# External dependencies
import matplotlib.pyplot as plt
from flwr.server.history import History

# Internal dependencies
from simulation import run_simulation

def produce_graph_without_cmv():
    """
    """
    performance_curves = []
    for n_scrambled_labels in [0] + list(range(2, 11)):
        performance_curve = []
        for n_bad_clients in range(101):
            performance : History = run_simulation(sim_config={
                'cmv_on': False,
                'n_clients': 100,
                'n_bad_clients': n_bad_clients,
                'n_scrambled_labels': n_scrambled_labels,
                'n_rounds': 1,
                'dataset': 'mnist',
                'n_client_epochs': 5,
                'client_batch_size': 128,
                'avg_client_std_threshold': 1.0
            })
            performance_curve.append(
                performance.metrics_centralized['accuracy'][1][1])

    plt.figure()
    x_axis = [x for x in range(101)]
    for performance_curve, n_scrambled_labels in \
        zip(performance_curves, [0] + list(range(2, 11))):
        plt.plot(x_axis, performance_curve,
                 label=f'{n_scrambled_labels}/10 scrambled labels')
    plt.title('Accuracy without CMV')
    plt.savefig('accuracy_without_cmv.png')

def produce_graph_with_cmv():
    """
    """
    performance_curves = []
    for n_scrambled_labels in [0] + list(range(2, 11)):
        performance_curve = []
        for n_bad_clients in range(101):
            performance : History = run_simulation(sim_config={
                'cmv_on': True,
                'n_clients': 100,
                'n_bad_clients': n_bad_clients,
                'n_scrambled_labels': n_scrambled_labels,
                'n_rounds': 1,
                'dataset': 'mnist',
                'n_client_epochs': 5,
                'client_batch_size': 128,
                'avg_client_std_threshold': 1.0
            })
            performance_curve.append(
                performance.metrics_centralized['accuracy'][1][1])

    plt.figure()
    x_axis = [x for x in range(101)]
    for performance_curve, n_scrambled_labels in \
        zip(performance_curves, [0] + list(range(2, 11))):
        plt.plot(x_axis, performance_curve,
                 label=f'{n_scrambled_labels}/10 scrambled labels')
    plt.title('Accuracy with CMV')
    plt.savefig('accuracy_with_cmv.png')

if __name__ == '__main__':
    produce_graph_without_cmv()
    produce_graph_with_cmv()

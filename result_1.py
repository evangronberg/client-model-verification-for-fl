"""
Script for getting result 1.
"""

# External dependencies
import matplotlib
import matplotlib.pyplot as plt
from flwr.server.history import History

# Internal dependencies
from simulation import run_simulation

def produce_graph_without_cmv():
    """
    """
    performance_curves = []
    for n_scrambled_labels in range(2, 11):
        performance_curve = []
        for n_bad_clients in range(11):
            performance : History = run_simulation(sim_config={
                'cmv_on': False,
                'n_clients': 10,
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
            print(f'\n\nn_scrambled_labels={n_scrambled_labels}, n_bad_clients={n_bad_clients}\n\n complete')
        performance_curves.append(performance_curve)
        print(f'\n\nALL n_scrambled_labels={n_scrambled_labels} FINISHED\n\n')

    x_axis = [x for x in range(11)]
    for performance_curve, n_scrambled_labels in \
        zip(performance_curves, list(range(2, 11))):
        plt.plot(x_axis, performance_curve,
                 label=f'{n_scrambled_labels}/10')
    plt.xlabel('Number of Bad Clients')
    plt.ylabel('Accuracy')
    plt.legend(
        title='Swapped Labels', fontsize=6, loc='lower left'
    ).get_title().set_fontsize(6)
    plt.title('Accuracy without CMV')
    plt.savefig(
        'paper_images/accuracy_without_cmv.png', dpi=300)

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
    # This prevents matplotlib from producing pop-ups
    matplotlib.use('Agg')
    produce_graph_without_cmv()
    # produce_graph_with_cmv()

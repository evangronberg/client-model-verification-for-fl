"""
Script for getting result 1.
"""

# External dependencies
import matplotlib
import matplotlib.pyplot as plt
from flwr.server.history import History

# Internal dependencies
from simulation import run_simulation

def produce_graph(cmv_on: bool) -> None:
    """
    Produces the graphs found in section 3.1.2 of the paper.

    On the author's processor (M1 Pro), this experiment
    takes ~1 hour WITHOUT CMV and ~3 hours WITH CMV.

    Arguments:
        cmv_on: Whether or not the experiment
                should have CMV enabled.
    Return Values:
        None
    """
    performance_curves = []
    for n_scrambled_labels in range(2, 11):
        performance_curve = []
        for n_bad_clients in range(11):
            performance : History = run_simulation(sim_config={
                'cmv_on': cmv_on,
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
    if cmv_on:
        title = 'Accuracy with CMV'
        img_path = 'paper_images/accuracy_with_cmv.png'
    else:
        title = 'Accuracy without CMV'
        img_path = 'paper_images/accuracy_without_cmv.png'
    plt.title(title)
    plt.savefig(img_path, dpi=300)

if __name__ == '__main__':
    # This prevents matplotlib from producing pop-ups
    matplotlib.use('Agg')
    produce_graph(cmv_on=False)
    produce_graph(cmv_on=True)

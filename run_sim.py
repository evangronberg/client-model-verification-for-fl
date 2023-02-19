"""
Script for running a simulation of client model verification.
"""

# External dependencies
import click
from flwr.server import ServerConfig
from flwr.simulation import start_simulation

# Internal dependencies
from client import Client
from dataset import Dataset, get_training_set
from client_model_verification import ClientModelVerification

@click.command()
@click.argument('n_good_clients', type=int)
@click.argument('n_bad_clients', type=int)
@click.argument('n_rounds', type=int)
def run_simulation(n_good_clients: int, n_bad_clients: int,
                   n_rounds: int) -> None:
    """
    Simulates federated learning with client model verification.
    """
    strategy = ClientModelVerification()
    server_config = ServerConfig(num_rounds=n_rounds)

    # dataset = Dataset(n_good_clients, n_bad_clients)
    # TRAINING_SETS = dataset.training_sets

    n_clients = n_good_clients + n_bad_clients
    start_simulation(
        client_fn=create_client,
        num_clients=n_clients,
        config=server_config,
        strategy=strategy,
    )

def create_client(client_id: str) -> Client:
    """
    Creates and returns a client object.
    """
    client = Client(training_set=get_training_set(int(client_id)))
    return client

if __name__ == '__main__':
    run_simulation()

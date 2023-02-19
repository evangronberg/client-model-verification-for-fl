"""
Script for running a simulation of client model verification.
"""

# External dependencies
import click
from flwr.server import ServerConfig
from flwr.simulation import start_simulation

# Internal dependencies
from client import Client
from datasets.bad_data import BadTrainingSet
from datasets.good_data import GoodTrainingSet
from client_model_verification import ClientModelVerification

# Global variables
TRAIN_SETS = []

@click.command()
@click.argument('n_clients', type=int)
@click.argument('n_rounds', type=int)
def run_simulation(n_clients: int, n_rounds: int) -> None:
    """
    Simulates federated learning with client model verification.
    """
    strategy = ClientModelVerification()
    server_config = ServerConfig(num_rounds=n_rounds)

    TRAIN_SETS.append(BadTrainingSet())
    for _ in range(n_clients-1):
        TRAIN_SETS.append(GoodTrainingSet())

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
    client = Client(train_set=TRAIN_SETS[int(client_id)])
    return client

if __name__ == '__main__':
    run_simulation()

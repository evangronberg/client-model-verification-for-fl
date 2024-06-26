"""
Script for running a simulation of client model verification.
"""

# Python-native dependencies
import json
from typing import Dict

# External dependencies
from flwr.server import ServerConfig
from flwr.server.history import History
from flwr.simulation import start_simulation

# Internal dependencies
from client import Client
from models import get_model
from datasets import get_dataset
from client_model_verification import ClientModelVerification

def run_simulation(sim_config: dict = None) -> History:
    """
    Simulates federated learning with client model verification.
    """
    if sim_config is None:
        sim_config = get_sim_config()
    else:
        with open('sim_config.json', 'w') as sim_config_json:
            json.dump(sim_config, sim_config_json)

    cmv_on : bool = sim_config['cmv_on']
    n_clients : int = sim_config['n_clients']
    n_rounds : int = sim_config['n_rounds']
    dataset_name :str = sim_config['dataset']
    avg_client_std_threshold : float =\
        sim_config['avg_client_std_threshold']

    server_config = ServerConfig(num_rounds=n_rounds)
    strategy = ClientModelVerification(
        dataset_name, n_clients,
        avg_client_std_threshold, cmv_on)

    performance : History = start_simulation(
        client_fn=create_client,
        num_clients=n_clients,
        config=server_config,
        strategy=strategy
    )
    return performance

def get_sim_config() -> Dict[str, int]:
    """
    Gets the configuration for the simulation.

    Arguments:
        None
    Return Values:
        sim_config: The simulation configuration.
    """
    with open('sim_config.json') as sim_config_json:
        sim_config = json.load(sim_config_json)
    return sim_config

def create_client(client_id: str) -> Client:
    """
    Creates and returns a client object.

    Arguments:
        client_id: The sequence number of
                   the client being created.
    Return Values:
        client:    The client.
    """
    # Get the config data
    sim_config = get_sim_config()
    n_clients = sim_config['n_clients']
    n_bad_clients = sim_config['n_bad_clients']
    n_scrambled_labels = sim_config['n_scrambled_labels']
    dataset_name = sim_config['dataset']
    n_client_epochs = sim_config['n_client_epochs']
    client_batch_size = sim_config['client_batch_size']

    # Get the model and training set
    model = get_model(dataset_name)
    dataset = get_dataset(
        dataset_name, n_clients,
        n_bad_clients, n_scrambled_labels)
    training_set = dataset.training_sets[int(client_id)]

    # Create and return the client
    client = Client(
        model, training_set,
        n_epochs=n_client_epochs,
        batch_size=client_batch_size
    )
    return client

if __name__ == '__main__':
    performance = run_simulation()
    print('\n\n\nFINAL ACCURACY FOR THIS RUN')    
    print(performance.metrics_centralized['accuracy'][1][1])

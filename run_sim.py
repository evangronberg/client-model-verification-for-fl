"""
Script for running a simulation of client model verification.
"""

# Python-native dependencies
import json
from typing import Dict

# External dependencies
from flwr.server import ServerConfig
from flwr.simulation import start_simulation

# Internal dependencies
from client import Client
from dataset import Dataset
from client_model_verification import ClientModelVerification

def run_simulation() -> None:
    """
    Simulates federated learning with client model verification.
    """
    sim_config = get_sim_config()
    n_clients = sim_config['n_clients']
    n_rounds = sim_config['n_rounds']

    strategy = ClientModelVerification()
    server_config = ServerConfig(num_rounds=n_rounds)

    start_simulation(
        client_fn=create_client,
        num_clients=n_clients,
        config=server_config,
        strategy=strategy)

def get_sim_config() -> Dict[str, int]:
    """
    """
    with open('sim_config.json') as sim_config_json:
        sim_config = json.load(sim_config_json)
    return sim_config

def create_client(client_id: str) -> Client:
    """
    Creates and returns a client object.
    """
    sim_config = get_sim_config()
    n_clients = sim_config['n_clients']
    n_bad_clients = sim_config['n_bad_clients']

    dataset = Dataset(n_clients, n_bad_clients)
    client = Client(training_set=dataset.training_sets[int(client_id)])
    return client

if __name__ == '__main__':
    run_simulation()

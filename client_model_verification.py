"""
Module for defining a means of client model
verification via Flower's Strategy class.
"""

# Python-native dependencies
from typing import Optional, Tuple, Union, Dict, List

# External dependencies
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.common import (
    Parameters, Scalar,
    FitIns, FitRes, EvaluateIns, EvaluateRes,
    ndarrays_to_parameters, parameters_to_ndarrays
)

# Internal dependencies
from models import get_model
from datasets import get_dataset

class ClientModelVerification(Strategy):
    """
    Federated learning server strategy that
    implements client model verification.
    """
    def __init__(
        self,
        dataset: str
    ) -> None:
        """
        Arguments:
            dataset: The name of the dataset to be used.
        """
        super().__init__()
        dataset = get_dataset(dataset)
        self.x_test = dataset.test_set[0]
        self.y_test = dataset.test_set[1]
        self.model = get_model(dataset)

    def initialize_parameters(
        self, 
        client_manager: ClientManager
    ) -> Optional[Parameters]:
        """
        """
        return self.model.get_weights()

    def configure_fit(
        self, 
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        """
        # Collect references to all of the clients
        clients = client_manager.sample(
                    client_manager.num_available())

        # Create the instructions (i.e., the
        # parameters) that each client will receive
        fit_instructions = FitIns(parameters, {})

        # Return the configuration where each
        # client has the same instructions
        clients_w_instructions =\
            [(client, fit_instructions) for client in clients]
        return clients_w_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        """
        # TODO: This is where the magic will happen.

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        """
        # Collect references to all of the clients
        clients = client_manager.sample(
                    client_manager.num_available())

        # Create the instructions (i.e., the
        # parameters) that each client will receive
        eval_instructions = EvaluateIns(parameters, {})

        # Return the configuration where each
        # client has the same instructions
        clients_w_instructions =\
            [(client, eval_instructions) for client in clients]
        return clients_w_instructions

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        """
        aggr_loss = weighted_loss_avg([
            (eval_res.num_examples, eval_res.loss)
            for _, eval_res in results
        ])
        aggr_metrics = self.aggregate_metrics(
            [eval_res.num_examples for _, eval_res in results],
            [eval_res.metrics for _, eval_res in results]
        )
        return aggr_loss, aggr_metrics

    def aggregate_metrics(
        self,
        example_counts: List[int],
        metrics_collection: List[Dict[str, Scalar]]
    ) -> Dict[str, Scalar]:
        """
        """
        aggr_metrics = {}
        for metric_name in metrics_collection[0].keys():
            aggr_metrics[metric_name] = 0.0

        total_example_count = 0
        for example_count, metrics in \
            zip(example_counts, metrics_collection):
            total_example_count += example_count
            for metric_name, metric_value in metrics.items():
                aggr_metrics[metric_name] += \
                    (metric_value * example_count)
        
        for metric_name in aggr_metrics.keys():
            aggr_metrics[metric_name] /= total_example_count

        return aggr_metrics

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        """
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        metrics = {'accuracy': accuracy}
        return loss, metrics

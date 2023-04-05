"""
Module for defining a means of client model
verification via Flower's Strategy class.
"""

# Python-native dependencies
from typing import Optional, Tuple, Union, Dict, List

# External dependencies
import numpy as np
from keras.models import Sequential
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import weighted_loss_avg, aggregate
from flwr.common import (
    Parameters, Scalar,
    FitIns, FitRes, EvaluateIns, EvaluateRes,
    ndarrays_to_parameters, parameters_to_ndarrays)

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
        dataset_name: str,
        n_clients: int,
        avg_client_std_threshold: float,
        cmv_on: bool = True
    ) -> None:
        """
        Arguments:
            dataset: The name of the dataset to be used.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.model = get_model(dataset_name)
        dataset = get_dataset(dataset_name, n_clients)
        self.x_test = dataset.test_set[0]
        self.y_test = dataset.test_set[1]
        self.avg_client_std_threshold = avg_client_std_threshold
        self.cmv_on = cmv_on

    def initialize_parameters(
        self, 
        client_manager: ClientManager
    ) -> Optional[Parameters]:
        """
        """
        parameters = self.model.get_weights()
        parameters = ndarrays_to_parameters(parameters)
        return parameters

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
        client_example_counts : List[int] = []
        client_model_params : List[List[np.ndarray]] =\
            [[] for _ in range(len(results))]
        for i, (_, result) in enumerate(results):
            client_example_counts.append(result.num_examples)
            client_model_params[i].append(
                parameters_to_ndarrays(result.parameters))

        client_models : List[Sequential] = []
        for params in client_model_params:
            model = get_model(self.dataset_name)
            model.set_weights(params[0])
            client_models.append(model)

        if self.cmv_on: 
            accuracies = []
            for model in client_models:
                _, accuracy = model.evaluate(self.x_test, self.y_test)
                accuracies.append(accuracy)
            accuracies = np.array(accuracies).reshape((len(accuracies), 1))

            accuracy_z_scores = np.abs((accuracies - np.mean(
                accuracies)) / np.std(accuracies)) * self.avg_client_std_threshold

            bad_client_model_indices = []
            total_client_examples = np.sum(client_example_counts)
            for i, z_score in enumerate(accuracy_z_scores):
                client_examples_proportion =\
                    client_example_counts[i] / total_client_examples
                max_z_score = np.abs(np.log10(client_examples_proportion))
                if z_score >= max_z_score:
                    bad_client_model_indices.append(i)

            client_example_counts = [
                i for j, i in enumerate(client_example_counts) \
                if j not in bad_client_model_indices
            ]
            client_model_params = [
                i for j, i in enumerate(client_model_params) \
                if j not in bad_client_model_indices
            ]

        parameters_aggregated = ndarrays_to_parameters(aggregate(
            [(params[0], n_examples) for params, n_examples \
            in zip(client_model_params, client_example_counts)]
        ))

        metrics_aggregated = {}

        return parameters_aggregated, metrics_aggregated

        # avg_accuracy = np.mean(accuracies)
        # cov = EmpiricalCovariance().fit(accuracies).covariance_

        # anomalous_models = []
        # for i in range(accuracies.shape[0]):
        #     mahalanobis_distance = pairwise_distances(accuracies[i], avg_accuracy, metric='mahalanobis', VI=np.linalg.inv(cov))
        #     if mahalanobis_distance > 3:
        #         anomalous_models.append(i)

        # all_predictions = []
        # for model in client_models:
        #     all_predictions.append(model.predict(self.x_test))
        # all_predictions = np.array(all_predictions)

        # We get the average across all clients models to establish what
        # the "average" client looks like in terms of per-example predictions
        # distances = []
        # avg_client = np.mean(all_predictions, axis=0)
        # for client_predictions in all_predictions:
        #     # Get the Mahanalobis distance from the average client for this client
        #     client_distance = pairwise_distances(
        #         client_predictions, avg_client,
        #         metric='euclidean'
        #     )
        #     avg_dist = np.sum(client_distance) / client_distance.size
        #     print(avg_dist)
        #     distances.append(avg_dist)

        # # We get the average across all clients models to establish what
        # # the "average" client looks like in terms of per-example predictions
        # distances = []
        # avg_client = np.mean(all_predictions, axis=0)
        # for client_predictions in all_predictions:
        #     # Get the covariance of the current client
        #     client_covariance = EmpiricalCovariance().fit(
        #         avg_client).covariance_
        #     # Get the Mahanalobis distance from the average client for this client
        #     client_mahalanobis_distance = pairwise_distances(
        #         client_predictions, avg_client,
        #         metric='mahalanobis',
        #         VI=np.linalg.inv(client_covariance)
        #     )
        #     avg_dist = np.sum(client_mahalanobis_distance) / client_mahalanobis_distance.size
        #     print(avg_dist)
        #     distances.append(avg_dist)

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
        parameters = parameters_to_ndarrays(parameters)
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        metrics = {'accuracy': accuracy}
        return loss, metrics

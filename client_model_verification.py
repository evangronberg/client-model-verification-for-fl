"""
"""

from typing import Optional, Tuple, List, Union, Dict

# External dependencies
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common import (
    Parameters, Scalar,
    FitIns, FitRes, EvaluateIns, EvaluateRes,
    ndarrays_to_parameters, parameters_to_ndarrays
)

class ClientModelVerification(Strategy):
    """
    Federated learning server strategy that
    implements client model verification.
    """
    def __init__(
        self,
    ) -> None:
        """
        Arguments:
            None
        """
        super().__init__()

    def initialize_parameters(
        self, 
        client_manager: ClientManager
    ) -> Optional[Parameters]:
        """
        """
        return super().initialize_parameters(client_manager)

    def configure_fit(
        self, 
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        """
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        """
        return super().aggregate_fit(server_round, results, failures)

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        """
        return super().configure_evaluate(server_round, parameters, client_manager)
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        """
        return super().aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        """
        return super().evaluate(server_round, parameters)

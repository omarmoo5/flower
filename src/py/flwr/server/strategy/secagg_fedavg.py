# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Secure Aggregation (SecAgg) Bonawitz et al.

Paper: https://eprint.iacr.org/2017/281.pdf
"""

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    NDArrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg

from .aggregate import aggregate, weighted_loss_avg
from .secagg import SecAggStrategy
from .strategy import Strategy


class SecAggFedAvg(FedAvg, SecAggStrategy):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
            self,
            fraction_fit: float = 0.1,
            fraction_evaluate: float = 0.1,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            sec_agg_param_dict: Dict[str, Scalar] = {}
    ) -> None:
        FedAvg.__init__(self, fraction_fit=fraction_fit,
                        fraction_evaluate=fraction_evaluate,
                        min_fit_clients=min_fit_clients,
                        min_evaluate_clients=min_evaluate_clients,
                        min_available_clients=min_available_clients,
                        evaluate_fn=evaluate_fn,
                        on_fit_config_fn=on_fit_config_fn,
                        on_evaluate_config_fn=on_evaluate_config_fn,
                        accept_failures=accept_failures,
                        initial_parameters=initial_parameters)
        self.sec_agg_param_dict = sec_agg_param_dict

    def get_sec_agg_param(self) -> Dict[str, int]:
        return self.sec_agg_param_dict.copy()

    '''def sec_agg_configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager, sample_num: int, min_num: int
    ) -> List[Tuple[ClientProxy, FitIns]]:
        #"""Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        fit_ins = FitIns(parameters, config)

        clients = client_manager.sample(
            num_clients=sample_num, min_num_clients=min_num
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]'''

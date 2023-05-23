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
"""Flower server."""

import concurrent.futures
from logging import DEBUG
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
)
from flwr.common.typing import GetParametersIns, ConsistencyCheckRes, ConsistencyCheckIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy
from src.utils import utils

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]

config = utils.get_config()

class Server:
    """Flower server."""

    def __init__(
            self, *, client_manager: ClientManager, strategy: Optional[Strategy] = None
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: Strategy = strategy if strategy is not None else FedAvg()
        self.max_workers: Optional[int] = None

    def set_max_workers(self, max_workers: Optional[int]) -> None:
        """Set the max_workers used by ThreadPoolExecutor."""
        self.max_workers = max_workers

    def set_strategy(self, strategy: Strategy) -> None:
        """Replace server strategy."""
        self.strategy = strategy

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            if res_fit:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history

    def evaluate_round(
            self,
            server_round: int,
            timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
            self,
            server_round: int,
            timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)

    def disconnect_all_clients(self, timeout: Optional[float]) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        clients = [all_clients[k] for k in all_clients.keys()]
        instruction = ReconnectIns(seconds=None)
        client_instructions = [(client_proxy, instruction) for client_proxy in clients]
        _ = reconnect_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

    def _get_initial_parameters(self, timeout: Optional[float]) -> Parameters:
        """Get initial parameters from one of the available clients."""

        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout)
        log(INFO, "Received initial parameters from one random client")
        return get_parameters_res.parameters


def reconnect_clients(
        client_instructions: List[Tuple[ClientProxy, ReconnectIns]],
        max_workers: Optional[int],
        timeout: Optional[float],
) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(reconnect_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, DisconnectRes]] = []
    failures: List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
        client: ClientProxy,
        reconnect: ReconnectIns,
        timeout: Optional[float],
) -> Tuple[ClientProxy, DisconnectRes]:
    """Instruct client to disconnect and (optionally) reconnect later."""
    disconnect = client.reconnect(
        reconnect,
        timeout=timeout,
    )
    return client, disconnect


def fit_clients(
        client_instructions: List[Tuple[ClientProxy, FitIns]],
        max_workers: Optional[int],
        timeout: Optional[float],
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures
        )
    return results, failures


def fit_client(
        client: ClientProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout=timeout)
    return client, fit_res


def _handle_finished_future_after_fit(
        future: concurrent.futures.Future,  # type: ignore
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""

    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, FitRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


def evaluate_clients(
        client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
        max_workers: Optional[int],
        timeout: Optional[float],
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_evaluate(
            future=future, results=results, failures=failures
        )
    return results, failures


def evaluate_client(
        client: ClientProxy,
        ins: EvaluateIns,
        timeout: Optional[float],
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(ins, timeout=timeout)
    return client, evaluate_res


def _handle_finished_future_after_evaluate(
        future: concurrent.futures.Future,  # type: ignore
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""

    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, EvaluateRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


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

import concurrent.futures
import timeit
from logging import INFO, WARNING
from typing import Dict, List, Optional, Tuple

from flwr.common.logger import log
from flwr.common.parameter import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.sec_agg import sec_agg_primitives
from flwr.common.typing import (AskKeysIns, AskKeysRes,
                                AskVectorsIns, AskVectorsRes,
                                FitIns, Parameters, Scalar,
                                SetupParamIns, SetupParamRes,
                                ShareKeysIns, ShareKeysPacket, ShareKeysRes,
                                UnmaskVectorsIns, UnmaskVectorsRes)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.secagg import SecAggStrategy

SetupParamResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, SetupParamRes]], List[BaseException]
]
AskKeysResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, AskKeysRes]], List[BaseException]
]
ShareKeysResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, ShareKeysRes]], List[BaseException]
]
AskVectorsResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, AskVectorsRes]], List[BaseException]
]
ConsistencyCheckResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, ConsistencyCheckRes]], List[BaseException]
]
UnmaskVectorsResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, UnmaskVectorsRes]], List[BaseException]
]


class SecAggServer(Server):
    """Flower secure aggregation server."""

    def __init__(self, *, client_manager: ClientManager, strategy: Optional[SecAggStrategy]) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)

    def fit_round(
            self,
            server_round: int,
            timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        total_time = 0
        total_time = total_time - timeit.default_timer()
        # Sample clients
        client_instructions_list = self.strategy.configure_fit(server_round=server_round,
                                                               parameters=self.parameters,
                                                               client_manager=self._client_manager)
        setup_param_clients: Dict[int, ClientProxy] = {}
        client_instructions: Dict[int, FitIns] = {}
        for idx, (client_proxy, fit_ins) in enumerate(client_instructions_list):
            setup_param_clients[idx] = client_proxy
            client_instructions[idx] = fit_ins

        # Get sec_agg parameters from strategy
        log(INFO, "Get sec_agg_param_dict from strategy")
        sec_agg_param_dict = self.strategy.get_sec_agg_param()
        sec_agg_param_dict["sample_num"] = len(client_instructions)
        sec_agg_param_dict = process_sec_agg_param_dict(sec_agg_param_dict)

        # === Stage 0: Setup ===
        # Give rnd, sample_num, share_num, threshold, client id
        log(INFO, "SecAgg Stage 0: Setting up Params")
        setup_param_results, failures = setup_param(
            clients=setup_param_clients,
            sec_agg_param_dict=sec_agg_param_dict
        )
        if len(setup_param_results) < sec_agg_param_dict['min_num']:
            raise Exception("Not enough available clients after setup param stage")
        ask_keys_clients: Dict[int, ClientProxy] = {}
        for idx, client in setup_param_clients.items():
            if client in [result[0] for result in setup_param_results]:
                ask_keys_clients[idx] = client

        # === Stage 1: Ask Public Keys ===
        log(INFO, "SecAgg Stage 1: Asking Keys")
        total_time = total_time + timeit.default_timer()
        ask_keys_results, failures = ask_keys(ask_keys_clients)
        total_time = total_time - timeit.default_timer()
        public_keys_dict: Dict[int, AskKeysRes] = {}
        if len(ask_keys_results) < sec_agg_param_dict['min_num']:
            raise Exception("Not enough available clients after ask keys stage")
        share_keys_clients: Dict[int, ClientProxy] = {}

        # Build public keys dict
        for idx, client in ask_keys_clients.items():
            if client in [result[0] for result in ask_keys_results]:
                pos = [result[0] for result in ask_keys_results].index(client)
                public_keys_dict[idx] = ask_keys_results[pos][1]
                share_keys_clients[idx] = client

        # === Stage 2: Share Keys ===
        log(INFO, "SecAgg Stage 2: Sharing Keys")
        total_time = total_time + timeit.default_timer()
        share_keys_results, failures = share_keys(
            share_keys_clients, public_keys_dict, sec_agg_param_dict['sample_num'], sec_agg_param_dict['share_num']
        )
        total_time = total_time - timeit.default_timer()
        if len(share_keys_results) < sec_agg_param_dict['min_num']:
            raise Exception("Not enough available clients after share keys stage")

        # Build forward packet list dictionary
        total_packet_list: List[ShareKeysPacket] = []
        forward_packet_list_dict: Dict[int, List[ShareKeysPacket]] = {}
        ask_vectors_clients: Dict[int, ClientProxy] = {}
        for idx, client in share_keys_clients.items():
            if client in [result[0] for result in share_keys_results]:
                pos = [result[0] for result in share_keys_results].index(client)
                ask_vectors_clients[idx] = client
                packet_list = share_keys_results[pos][1].share_keys_res_list
                total_packet_list += packet_list

        for idx in ask_vectors_clients.keys():
            forward_packet_list_dict[idx] = []

        for packet in total_packet_list:
            destination = packet.destination
            if destination in ask_vectors_clients.keys():
                forward_packet_list_dict[destination].append(packet)

        # === Stage 3: Ask Vectors ===
        log(INFO, "SecAgg Stage 3: Asking Vectors")
        total_time = total_time + timeit.default_timer()
        ask_vectors_results, failures = ask_vectors(ask_vectors_clients, forward_packet_list_dict, client_instructions)
        total_time = total_time - timeit.default_timer()
        if len(ask_vectors_results) < sec_agg_param_dict['min_num']:
            raise Exception("Not enough available clients after ask vectors stage")
        # Get shape of vector sent by first client
        masked_vector = sec_agg_primitives.weights_zero_generate(
            [i.shape for i in parameters_to_ndarrays(ask_vectors_results[0][1].parameters)])
        # Add all collected masked vectors and compute available and dropout clients set
        consistency_check_clients: Dict[int, ClientProxy] = {}
        dropout_clients = ask_vectors_clients.copy()
        for idx, client in ask_vectors_clients.items():
            if client in [result[0] for result in ask_vectors_results]:
                pos = [result[0] for result in ask_vectors_results].index(client)
                consistency_check_clients[idx] = client
                client_parameters = ask_vectors_results[pos][1].parameters
                masked_vector = sec_agg_primitives.weights_addition(
                    masked_vector, parameters_to_ndarrays(client_parameters))


        # ===stage 4 consistency check ====
        # unmask_vectors_clients= consistency_check_clients
        log(INFO,"secAgg stage 4:consistency check")
        signatures: Dict[int, bytes]={}
        total_time = total_time + timeit.default_timer()
        consistency_check_results_and_failures = consistency_checks(consistency_check_clients)
        total_time = total_time - timeit.default_timer()
        consistency_check_results= consistency_check_results_and_failures[0]
        if len(consistency_check_results) < sec_agg_param_dict['min_num']:
            raise Exception("Not enough available clients after consistency check stage")
        unmask_vectors_clients: Dict[int, ClientProxy] = {}
        for idx, client in consistency_check_clients.items():
            if client in [result[0] for result in consistency_check_results]:
                pos = [result[0] for result in consistency_check_results].index(client)
                unmask_vectors_clients[idx] = client
                dropout_clients.pop(idx)
                signatures[pos] = consistency_check_results[pos][1].signature

        # === Stage 4: Unmask Vectors ===
        log(INFO, "SecAgg Stage 4: Unmasking Vectors")
        total_time = total_time + timeit.default_timer()
        unmask_vectors_results_and_failures = unmask_vectors(
            unmask_vectors_clients, dropout_clients, signatures, sec_agg_param_dict['sample_num'], sec_agg_param_dict['share_num'])
        unmask_vectors_results = unmask_vectors_results_and_failures[0]
        total_time = total_time - timeit.default_timer()
        # Build collected shares dict
        collected_shares_dict: Dict[int, List[bytes]] = {}
        for idx in ask_vectors_clients.keys():
            collected_shares_dict[idx] = []

        if len(unmask_vectors_results) < sec_agg_param_dict['min_num']:
            raise Exception("Not enough available clients after unmask vectors stage")
        for result in unmask_vectors_results:
            unmask_vectors_res = result[1]
            for owner_id, share in unmask_vectors_res.share_dict.items():
                collected_shares_dict[owner_id].append(share)

        # Remove mask for every client who is available before ask vectors stage,
        # Divide vector by first element
        for client_id, share_list in collected_shares_dict.items():
            if len(share_list) < sec_agg_param_dict['threshold']:
                raise Exception(
                    "Not enough shares to recover secret in unmask vectors stage")
            secret = sec_agg_primitives.combine_shares(share_list=share_list)
            if client_id in unmask_vectors_clients.keys():
                # seed is an available client's b
                private_mask = sec_agg_primitives.pseudo_rand_gen(secret,
                                                                  sec_agg_param_dict['mod_range'],
                                                                  sec_agg_primitives.weights_shape(masked_vector))
                masked_vector = sec_agg_primitives.weights_subtraction(masked_vector, private_mask)
            else:
                # seed is a dropout client's sk1
                neighbor_list: List[int] = []
                if sec_agg_param_dict['share_num'] == sec_agg_param_dict['sample_num']:
                    neighbor_list = list(ask_vectors_clients.keys())
                    neighbor_list.remove(client_id)
                else:
                    for i in range(-int(sec_agg_param_dict['share_num'] / 2),
                                   int(sec_agg_param_dict['share_num'] / 2) + 1):
                        if i != 0 and (
                                (i + client_id) % sec_agg_param_dict['sample_num']) in ask_vectors_clients.keys():
                            neighbor_list.append((i + client_id) % sec_agg_param_dict['sample_num'])

                for neighbor_id in neighbor_list:
                    shared_key = sec_agg_primitives.generate_shared_key(
                        sec_agg_primitives.bytes_to_private_key(secret),
                        sec_agg_primitives.bytes_to_public_key(public_keys_dict[neighbor_id].pk1))
                    pairwise_mask = sec_agg_primitives.pseudo_rand_gen(shared_key,
                                                                       sec_agg_param_dict['mod_range'],
                                                                       sec_agg_primitives.weights_shape(masked_vector))
                    if client_id > neighbor_id:
                        masked_vector = sec_agg_primitives.weights_addition(masked_vector, pairwise_mask)
                    else:
                        masked_vector = sec_agg_primitives.weights_subtraction(masked_vector, pairwise_mask)

        masked_vector = sec_agg_primitives.weights_mod(masked_vector, sec_agg_param_dict['mod_range'])

        masked_vector, masked_vector_uiv = masked_vector[:-4], masked_vector[-4:]
        # Divide vector by number of clients who have given us their masked vector
        # i.e. those participating in final unmask vectors stage

        # Weights
        total_weights_factor, masked_vector = sec_agg_primitives.factor_weights_extract(masked_vector)
        masked_vector = sec_agg_primitives.weights_divide(masked_vector, total_weights_factor)

        # Updated Item Vector
        total_iv_factor, masked_vector_uiv = sec_agg_primitives.factor_weights_extract(masked_vector_uiv)
        masked_vector_uiv = sec_agg_primitives.weights_divide(masked_vector_uiv, total_iv_factor)

        aggregated_weights = sec_agg_primitives.reverse_quantize(masked_vector,
                                                                sec_agg_param_dict['clipping_range'],
                                                                sec_agg_param_dict['target_range'])

        total_updated_iv = sec_agg_primitives.reverse_quantize(masked_vector_uiv,
                                                               sec_agg_param_dict['clipping_range'],
                                                               sec_agg_param_dict['target_range'])
        total_updated_iv = sec_agg_primitives.weights_multiply(total_updated_iv,total_iv_factor)

        # MF-Sec AGG ----
        total_updated_ivs, aggregated_embeddings = np.rint(total_updated_iv[0]), total_updated_iv[1:]
        no_update_mask = (total_updated_ivs == 0)
        if any(no_update_mask):
            # Division by zero handling
            log(WARNING, f"{no_update_mask.sum()} Items didn't updated during this round")
            total_updated_ivs[no_update_mask] = 1
            embeddings_t = parameters_to_ndarrays(self.parameters)[:2]
            for embedding_t, embedding_t_plus1 in zip(embeddings_t, aggregated_embeddings):
                embedding_t_plus1[no_update_mask] = embedding_t[no_update_mask]
        aggregated_embeddings = [emb/total_updated_ivs.reshape(-1, 1) for emb in aggregated_embeddings]
        # -----------------
        aggregated_parameters = ndarrays_to_parameters([*aggregated_embeddings + aggregated_weights])
        return aggregated_parameters, [0], [0]


def process_sec_agg_param_dict(sec_agg_param_dict: Dict[str, Scalar]) -> Dict[str, Scalar]:
    # min_num will be replaced with intended min_num based on sample_num
    # if both min_frac or min_num not provided, we take maximum of either 2 or 0.9 * sampled
    # if either one is provided, we use that
    # Otherwise, we take the maximum
    # Note we will eventually check whether min_num>=2
    if 'min_frac' not in sec_agg_param_dict:
        if 'min_num' not in sec_agg_param_dict:
            sec_agg_param_dict['min_num'] = max(2, int(0.9 * sec_agg_param_dict['sample_num']))
    else:
        if 'min_num' not in sec_agg_param_dict:
            sec_agg_param_dict['min_num'] = int(sec_agg_param_dict['min_frac'] * sec_agg_param_dict['sample_num'])
        else:
            sec_agg_param_dict['min_num'] = max(sec_agg_param_dict['min_num'],
                                                int(sec_agg_param_dict['min_frac'] * sec_agg_param_dict['sample_num']))

    if 'share_num' not in sec_agg_param_dict:
        # Complete graph
        sec_agg_param_dict['share_num'] = sec_agg_param_dict['sample_num']
    elif sec_agg_param_dict['share_num'] % 2 == 0 and\
            sec_agg_param_dict['share_num'] != sec_agg_param_dict['sample_num']:
        # we want share_num of each node to be either odd or sample_num
        log(WARNING,
            "share_num value changed due to sample num and share_num constraints! See documentation for reason")
        sec_agg_param_dict['share_num'] += 1

    if 'threshold' not in sec_agg_param_dict:
        sec_agg_param_dict['threshold'] = max(2, int(sec_agg_param_dict['share_num'] * 0.9))

    # Maximum number of example trained set to 1000
    if 'max_weights_factor' not in sec_agg_param_dict:
        # todo: double check this number
        sec_agg_param_dict['max_weights_factor'] = 4000

    # Quantization parameters
    if 'clipping_range' not in sec_agg_param_dict:
        sec_agg_param_dict['clipping_range'] = 3

    if 'target_range' not in sec_agg_param_dict:
        sec_agg_param_dict['target_range'] = 16777216

    if 'mod_range' not in sec_agg_param_dict:
        sec_agg_param_dict['mod_range'] = sec_agg_param_dict['sample_num'] * \
                                          sec_agg_param_dict['target_range'] * \
                                          sec_agg_param_dict['max_weights_factor']

    if 'timeout' not in sec_agg_param_dict:
        sec_agg_param_dict['timeout'] = 30

    log(
        INFO,
        f"SecAgg parameters: {sec_agg_param_dict}",
    )

    assert (
            sec_agg_param_dict['sample_num'] >= 2
            and 2 <= sec_agg_param_dict['min_num'] <= sec_agg_param_dict['sample_num']
            and sec_agg_param_dict['sample_num'] >= sec_agg_param_dict['share_num'] >= sec_agg_param_dict[
                'threshold'] >= 2
            and (sec_agg_param_dict['share_num'] % 2 == 1 or
                 sec_agg_param_dict['share_num'] == sec_agg_param_dict['sample_num'])
            and sec_agg_param_dict['target_range'] * sec_agg_param_dict['sample_num'] * sec_agg_param_dict[
                'max_weights_factor'] <= sec_agg_param_dict['mod_range']
    ), "SecAgg parameters not accepted"
    return sec_agg_param_dict


def setup_param(
        clients: Dict[int, ClientProxy],
        sec_agg_param_dict: Dict[str, Scalar]
) -> SetupParamResultsAndFailures:
    def sec_agg_param_dict_with_sec_agg_id(sec_agg_param_dict: Dict[str, Scalar], sec_agg_id: int):
        new_sec_agg_param_dict = sec_agg_param_dict.copy()
        new_sec_agg_param_dict['sec_agg_id'] = sec_agg_id
        return new_sec_agg_param_dict

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                lambda p: setup_param_client(*p),
                (
                    c,
                    SetupParamIns(
                        sec_agg_param_dict=sec_agg_param_dict_with_sec_agg_id(
                            sec_agg_param_dict, idx),
                    ),
                ),
            )
            for idx, c in clients.items()
        ]
        concurrent.futures.wait(futures)
    results: List[Tuple[ClientProxy, SetupParamRes]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            results.append(result)
    return results, failures


def setup_param_client(client: ClientProxy, setup_param_msg: SetupParamIns) -> Tuple[ClientProxy, SetupParamRes]:
    setup_param_res = client.setup_param(setup_param_msg)
    return client, setup_param_res


def ask_keys(clients: Dict[int, ClientProxy]) -> AskKeysResultsAndFailures:

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(ask_keys_client, c) for c in clients.values()]
        concurrent.futures.wait(futures)
    results: List[Tuple[ClientProxy, AskKeysRes]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            results.append(result)
    return results, failures


def ask_keys_client(client: ClientProxy) -> Tuple[ClientProxy, AskKeysRes]:
    ask_keys_res = client.ask_keys(AskKeysIns())
    return client, ask_keys_res


def share_keys(clients: Dict[int, ClientProxy],
               public_keys_dict: Dict[int, AskKeysRes],
               sample_num: int, share_num: int) -> ShareKeysResultsAndFailures:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                lambda p: share_keys_client(*p),
                (client, idx, public_keys_dict, sample_num, share_num),
            )
            for idx, client in clients.items()
        ]
        concurrent.futures.wait(futures)
    results: List[Tuple[ClientProxy, ShareKeysRes]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            results.append(result)
    return results, failures


def share_keys_client(client: ClientProxy,
                      idx: int,
                      public_keys_dict: Dict[int, AskKeysRes],
                      sample_num: int,
                      share_num: int) -> Tuple[ClientProxy, ShareKeysRes]:
    if share_num == sample_num:
        # complete graph
        return client, client.share_keys(ShareKeysIns(public_keys_dict=public_keys_dict))
    local_dict: Dict[int, AskKeysRes] = {}
    for i in range(-int(share_num / 2), int(share_num / 2) + 1):
        if ((i + idx) % sample_num) in public_keys_dict.keys():
            local_dict[(i + idx) % sample_num] = public_keys_dict[(i + idx) % sample_num]

    return client, client.share_keys(ShareKeysIns(public_keys_dict=local_dict))


def ask_vectors(clients: Dict[int, ClientProxy],
                forward_packet_list_dict: Dict[int, List[ShareKeysPacket]],
                client_instructions: Dict[int, FitIns]) -> AskVectorsResultsAndFailures:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                lambda p: ask_vectors_client(*p),
                (client, forward_packet_list_dict[idx], client_instructions[idx]),
            )
            for idx, client in clients.items()
        ]
        concurrent.futures.wait(futures)
    results: List[Tuple[ClientProxy, AskVectorsRes]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            results.append(result)
    return results, failures


def ask_vectors_client(client: ClientProxy, forward_packet_list: List[ShareKeysPacket], fit_ins: FitIns) -> Tuple[
    ClientProxy, AskVectorsRes]:
    return client, client.ask_vectors(AskVectorsIns(ask_vectors_in_list=forward_packet_list, fit_ins=fit_ins))

def consistency_checks(clients:Dict[int, ClientProxy]):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = futures = [
            executor.submit(
                lambda p: consistency_check_client(*p),
                (client, idx, list(clients.keys())),
            )
            for idx, client in clients.items()
        ]
        concurrent.futures.wait(futures)
    results: List[Tuple[ClientProxy, ConsistencyCheckRes]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            results.append(result)
    return results, failures

def consistency_check_client(client: ClientProxy, idx: int, clients: List[ClientProxy]) -> Tuple[ClientProxy, ConsistencyCheckRes]:
    consistency_check_res = client.consistency_checks(ConsistencyCheckIns(available_clients=clients))
    return client, consistency_check_res



def unmask_vectors(clients: Dict[int, ClientProxy],
                   dropout_clients: Dict[int, ClientProxy],
                   signatures: Dict[int, bytes],
                   sample_num: int, share_num: int) -> UnmaskVectorsResultsAndFailures:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                lambda p: unmask_vectors_client(*p),
                (client, idx, list(clients.keys()), list(
                    dropout_clients.keys()), signatures, sample_num, share_num),
            )
            for idx, client in clients.items()
        ]
        concurrent.futures.wait(futures)
    results: List[Tuple[ClientProxy, UnmaskVectorsRes]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            results.append(result)
    return results, failures


def unmask_vectors_client(client: ClientProxy, idx: int, clients: List[ClientProxy], dropout_clients: List[ClientProxy],
                          signatures: Dict[int, bytes],sample_num: int, share_num: int) -> Tuple[ClientProxy, UnmaskVectorsRes]:
    if share_num == sample_num:
        # complete graph
        return client, client.unmask_vectors(UnmaskVectorsIns(signatures=signatures, available_clients=clients,
                                                              dropout_clients=dropout_clients))
    local_clients: List[int] = []
    local_dropout_clients: List[int] = []
    for i in range(-int(share_num / 2), int(share_num / 2) + 1):
        if ((i + idx) % sample_num) in clients:
            local_clients.append((i + idx) % sample_num)
        if ((i + idx) % sample_num) in dropout_clients:
            local_dropout_clients.append((i + idx) % sample_num)
    return client, client.unmask_vectors(UnmaskVectorsIns(signatures=signatures, available_clients=local_clients,
                                                          dropout_clients=local_dropout_clients))

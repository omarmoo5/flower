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
"""Flower client app."""

import sys
import time
from logging import INFO
from typing import Callable, Dict, Optional, Union

from flwr.common import (
    GRPC_MAX_MESSAGE_LENGTH,
    EventType,
    event,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.address import parse_address
from flwr.common.constant import MISSING_EXTRA_REST
from flwr.common.logger import log
from flwr.common.typing import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    NDArrays,
    Status, SetupParamIns, SetupParamRes, AskKeysIns, AskKeysRes, UnmaskVectorsIns, AskVectorsIns, AskVectorsRes,
    ShareKeysIns, UnmaskVectorsRes, ShareKeysRes, ConsistencyCheckIns, ConsistencyCheckRes,
)

from .client import Client
from .grpc_client.connection import grpc_connection
from .message_handler.message_handler import handle
from .numpy_client import NumPyClient
from .numpy_client import has_evaluate as numpyclient_has_evaluate
from .numpy_client import has_fit as numpyclient_has_fit
from .numpy_client import has_get_parameters as numpyclient_has_get_parameters
from .numpy_client import has_get_properties as numpyclient_has_get_properties

EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_FIT = """
NumPyClient.fit did not return a tuple with 3 elements.
The returned values should have the following type signature:

    Tuple[NDArrays, int, Dict[str, Scalar]]

Example
-------

    model.get_weights(), 10, {"accuracy": 0.95}

"""

EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_EVALUATE = """
NumPyClient.evaluate did not return a tuple with 3 elements.
The returned values should have the following type signature:

    Tuple[float, int, Dict[str, Scalar]]

Example
-------

    0.5, 10, {"accuracy": 0.95}

"""

ClientLike = Union[Client, NumPyClient]


# pylint: disable=import-outside-toplevel,too-many-locals
def start_client(
        *,
        server_address: str,
        client: Client,
        grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
        root_certificates: Optional[Union[bytes, str]] = None,
        rest: bool = False,
) -> None:
    """Start a Flower client node which connects to a Flower server.

    Parameters
    ----------
    server_address : str
        The IPv4 or IPv6 address of the server. If the Flower
        server runs on the same machine on port 8080, then `server_address`
        would be `"[::]:8080"`.
    client : flwr.client.Client
        An implementation of the abstract base
        class `flwr.client.Client`.
    grpc_max_message_length : int (default: 536_870_912, this equals 512MB)
        The maximum length of gRPC messages that can be exchanged with the
        Flower server. The default should be sufficient for most models.
        Users who train very large models might need to increase this
        value. Note that the Flower server needs to be started with the
        same value (see `flwr.server.start_server`), otherwise it will not
        know about the increased limit and block larger messages.
    root_certificates : Optional[Union[bytes, str]] (default: None)
        The PEM-encoded root certificates as a byte string or a path string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.
    rest : bool (default: False)
        Defines whether or not the client is interacting with the server using the
        experimental REST API. This feature is experimental, it might change
        considerably in future versions of Flower.

    Examples
    --------
    Starting a gRPC client with an insecure server connection:

    >>> start_client(
    >>>     server_address=localhost:8080,
    >>>     client=FlowerClient(),
    >>> )

    Starting an SSL-enabled gRPC client:

    >>> from pathlib import Path
    >>> start_client(
    >>>     server_address=localhost:8080,
    >>>     client=FlowerClient(),
    >>>     root_certificates=Path("/crts/root.pem").read_bytes(),
    >>> )
    """

    event(EventType.START_CLIENT_ENTER)

    # Parse IP address
    parsed_address = parse_address(server_address)
    if not parsed_address:
        sys.exit(f"Server address ({server_address}) cannot be parsed.")
    host, port, is_v6 = parsed_address
    address = f"[{host}]:{port}" if is_v6 else f"{host}:{port}"

    # Use either gRPC bidirectional streaming or REST request/response
    if rest:
        try:
            from .rest_client.connection import http_request_response
        except ModuleNotFoundError:
            sys.exit(MISSING_EXTRA_REST)
        if server_address[:4] != "http":
            sys.exit(
                "When using the REST API, please provide `https://` or "
                "`http://` before the server address (e.g. `http://127.0.0.1:8080`)"
            )
        connection = http_request_response
    else:
        connection = grpc_connection
    while True:
        sleep_duration: int = 0
        with connection(
                address,
                max_message_length=grpc_max_message_length,
                root_certificates=root_certificates,
        ) as conn:
            receive, send = conn

            while True:
                server_message = receive()
                if server_message is None:
                    time.sleep(3)  # Wait for 3s before asking again
                    continue
                client_message, sleep_duration, keep_going = handle(
                    client, server_message
                )
                send(client_message)
                if not keep_going:
                    break
        if sleep_duration == 0:
            log(INFO, "Disconnect and shut down")
            break
        # Sleep and reconnect afterwards
        log(
            INFO,
            "Disconnect, then re-establish connection after %s second(s)",
            sleep_duration,
        )
        time.sleep(sleep_duration)

    event(EventType.START_CLIENT_LEAVE)


def start_numpy_client(
        *,
        server_address: str,
        client: NumPyClient,
        grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
        root_certificates: Optional[bytes] = None,
        rest: bool = False,
) -> None:
    """Start a Flower NumPyClient which connects to a gRPC server.

    Parameters
    ----------
    server_address : str
        The IPv4 or IPv6 address of the server. If the Flower server runs on
        the same machine on port 8080, then `server_address` would be
        `"[::]:8080"`.
    client : flwr.client.NumPyClient
        An implementation of the abstract base class `flwr.client.NumPyClient`.
    grpc_max_message_length : int (default: 536_870_912, this equals 512MB)
        The maximum length of gRPC messages that can be exchanged with the
        Flower server. The default should be sufficient for most models.
        Users who train very large models might need to increase this
        value. Note that the Flower server needs to be started with the
        same value (see `flwr.server.start_server`), otherwise it will not
        know about the increased limit and block larger messages.
    root_certificates : bytes (default: None)
        The PEM-encoded root certificates as a byte string or a path string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.
    rest : bool (default: False)
        Defines whether or not the client is interacting with the server using the
        experimental REST API. This feature is experimental, it might be change
        considerably in future versions of Flower.

    Examples
    --------
    Starting a client with an insecure server connection:

    >>> start_client(
    >>>     server_address=localhost:8080,
    >>>     client=FlowerClient(),
    >>> )

    Starting a SSL-enabled client:

    >>> from pathlib import Path
    >>> start_client(
    >>>     server_address=localhost:8080,
    >>>     client=FlowerClient(),
    >>>     root_certificates=Path("/crts/root.pem").read_bytes(),
    >>> )
    """

    # Start
    start_client(
        server_address=server_address,
        client=_wrap_numpy_client(client=client),
        grpc_max_message_length=grpc_max_message_length,
        root_certificates=root_certificates,
        rest=rest,
    )


def to_client(client_like: ClientLike) -> Client:
    """Take any Client-like object and return it as a Client."""
    if isinstance(client_like, NumPyClient):
        return _wrap_numpy_client(client=client_like)
    return client_like


def _constructor(self: Client, numpy_client: NumPyClient) -> None:
    self.numpy_client = numpy_client  # type: ignore


def _get_properties(self: Client, ins: GetPropertiesIns) -> GetPropertiesRes:
    """Return the current client properties."""
    properties = self.numpy_client.get_properties(config=ins.config)  # type: ignore
    return GetPropertiesRes(
        status=Status(code=Code.OK, message="Success"),
        properties=properties,
    )


def _get_parameters(self: Client, ins: GetParametersIns) -> GetParametersRes:
    """Return the current local model parameters."""
    parameters = self.numpy_client.get_parameters(None)  # type: ignore
    parameters_proto = ndarrays_to_parameters(parameters)
    return GetParametersRes(
        status=Status(code=Code.OK, message="Success"), parameters=parameters_proto
    )


def _fit(self: Client, ins: FitIns) -> FitRes:
    """Refine the provided parameters using the locally held dataset."""

    # Deconstruct FitIns
    parameters: NDArrays = parameters_to_ndarrays(ins.parameters)

    # Train
    results = self.numpy_client.fit(parameters, ins.config)  # type: ignore
    if not (
            len(results) == 3
            and isinstance(results[0], list)
            and isinstance(results[1], int)
            and isinstance(results[2], dict)
    ):
        raise Exception(EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_FIT)

    # Return FitRes
    parameters_prime, num_examples, metrics = results
    parameters_prime_proto = ndarrays_to_parameters(parameters_prime)
    return FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=parameters_prime_proto,
        num_examples=num_examples,
        metrics=metrics,
    )


def _evaluate(self: Client, ins: EvaluateIns) -> EvaluateRes:
    """Evaluate the provided parameters using the locally held dataset."""
    parameters: NDArrays = parameters_to_ndarrays(ins.parameters)

    results = self.numpy_client.evaluate(parameters, ins.config)  # type: ignore
    if not (
            len(results) == 3
            and isinstance(results[0], float)
            and isinstance(results[1], int)
            and isinstance(results[2], dict)
    ):
        raise Exception(EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_EVALUATE)

    # Return EvaluateRes
    loss, num_examples, metrics = results
    return EvaluateRes(
        status=Status(code=Code.OK, message="Success"),
        loss=loss,
        num_examples=num_examples,
        metrics=metrics,
    )


def _setup_param(self: Client, ins: SetupParamIns) -> SetupParamRes:
    """Set up the parameters of the client."""
    res = self.numpy_client.setup_param(ins)  # type: ignore
    return res


def _ask_keys(self: Client, ins: AskKeysIns) -> AskKeysRes:
    """Ask for the keys of the client."""
    res = self.numpy_client.ask_keys(ins)  # type: ignore
    return res


def _share_keys(self: Client, ins: ShareKeysIns) -> ShareKeysRes:
    """Share the keys of the client."""
    res = self.numpy_client.share_keys(ins)  # type: ignore
    return res


def _ask_vectors(self: Client, ins: AskVectorsIns) -> AskVectorsRes:
    """Ask for the vectors of the client."""
    res = self.numpy_client.ask_vectors(ins)  # type: ignore
    return res

def _consistency_checks(self: Client, ins: ConsistencyCheckIns) -> ConsistencyCheckRes:
    """Ask for the vectors of the client."""
    res = self.numpy_client.consistency_checks(ins)  # type: ignore
    return res

def _unmask_vectors(self: Client, ins: UnmaskVectorsIns) -> UnmaskVectorsRes:
    """Unmask the vectors of the client."""
    res = self.numpy_client.unmask_vectors(ins)  # type: ignore
    return res


def _wrap_numpy_client(client: NumPyClient) -> Client:
    member_dict: Dict[str, Callable] = {  # type: ignore
        "__init__": _constructor,
    }

    # Add wrapper type methods (if overridden)

    if numpyclient_has_get_properties(client=client):
        member_dict["get_properties"] = _get_properties

    if numpyclient_has_get_parameters(client=client):
        member_dict["get_parameters"] = _get_parameters

    if numpyclient_has_fit(client=client):
        member_dict["fit"] = _fit

    if numpyclient_has_evaluate(client=client):
        member_dict["evaluate"] = _evaluate

    member_dict["setup_param"] = _setup_param
    member_dict["ask_keys"] = _ask_keys
    member_dict["share_keys"] = _share_keys
    member_dict["ask_vectors"] = _ask_vectors
    member_dict["consistency_checks"]=_consistency_checks
    member_dict["unmask_vectors"] = _unmask_vectors

    # Create wrapper class
    wrapper_class = type("NumPyClientWrapper", (Client,), member_dict)

    # Create and return an instance of the newly created class
    return wrapper_class(numpy_client=client)  # type: ignore


def run_client() -> None:
    """Run Flower client."""
    log(INFO, "Running Flower client...")
    time.sleep(3)

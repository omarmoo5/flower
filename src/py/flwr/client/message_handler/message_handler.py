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
"""Client-side message handler."""

from typing import Tuple

from flwr.client.client import (
    Client,
    maybe_call_evaluate,
    maybe_call_fit,
    maybe_call_get_parameters,
    maybe_call_get_properties,
)
from flwr.common import serde
from flwr.proto.transport_pb2 import ClientMessage, Reason, ServerMessage


class UnknownServerMessage(Exception):
    """Exception indicating that the received message is unknown."""


def handle(
        client: Client, server_msg: ServerMessage
) -> Tuple[ClientMessage, int, bool]:
    """Handle incoming messages from the server.

    Parameters
    ----------
    client : Client
        The Client instance provided by the user.
    server_msg: ServerMessage
        The message coming from the server, to be processed by the client.

    Returns
    -------
    client_msg: ClientMessage
        The result message that should be returned to the server.
    sleep_duration : int
        Number of seconds that the client should disconnect from the server.
    keep_going : bool
        Flag that indicates whether the client should continue to process the
        next message from the server (True) or disconnect and optionally
        reconnect later (False).
    """
    field = server_msg.WhichOneof("msg")
    if field == "reconnect_ins":
        disconnect_msg, sleep_duration = _reconnect(server_msg.reconnect_ins)
        return disconnect_msg, sleep_duration, False
    if field == "get_properties_ins":
        return _get_properties(client, server_msg.get_properties_ins), 0, True
    if field == "get_parameters_ins":
        return _get_parameters(client, server_msg.get_parameters_ins), 0, True
    if field == "fit_ins":
        return _fit(client, server_msg.fit_ins), 0, True
    if field == "evaluate_ins":
        return _evaluate(client, server_msg.evaluate_ins), 0, True
    if field == "sec_agg_msg":
        field = server_msg.sec_agg_msg.WhichOneof("msg")
        if field == "setup_param":
            return _setup_param(client, server_msg.sec_agg_msg), 0, True
        elif field == "ask_keys":
            return _ask_keys(client, server_msg.sec_agg_msg), 0, True
        elif field == "share_keys":
            return _share_keys(client, server_msg.sec_agg_msg), 0, True
        elif field == "ask_vectors":
            return _ask_vectors(client, server_msg.sec_agg_msg), 0, True
        #TODO add consistency
        elif field == "consistency_checks":
            return _consistency_checks(client,server_msg.sec_agg_msg), 0, True
        elif field == "unmask_vectors":
            return _unmask_vectors(client, server_msg.sec_agg_msg), 0, True
    raise UnknownServerMessage()


def _reconnect(
        reconnect_msg: ServerMessage.ReconnectIns,
) -> Tuple[ClientMessage, int]:
    # Determine the reason for sending DisconnectRes message
    reason = Reason.ACK
    sleep_duration = None
    if reconnect_msg.seconds is not None:
        reason = Reason.RECONNECT
        sleep_duration = reconnect_msg.seconds
    # Build DisconnectRes message
    disconnect_res = ClientMessage.DisconnectRes(reason=reason)
    return ClientMessage(disconnect_res=disconnect_res), sleep_duration


def _get_properties(
        client: Client, get_properties_msg: ServerMessage.GetPropertiesIns
) -> ClientMessage:
    # Deserialize `get_properties` instruction
    get_properties_ins = serde.get_properties_ins_from_proto(get_properties_msg)

    # Request properties
    get_properties_res = maybe_call_get_properties(
        client=client,
        get_properties_ins=get_properties_ins,
    )

    # Serialize response
    get_properties_res_proto = serde.get_properties_res_to_proto(get_properties_res)
    return ClientMessage(get_properties_res=get_properties_res_proto)


def _get_parameters(
        client: Client, get_parameters_msg: ServerMessage.GetParametersIns
) -> ClientMessage:
    # Deserialize `get_parameters` instruction
    get_parameters_ins = serde.get_parameters_ins_from_proto(get_parameters_msg)

    # Request parameters
    get_parameters_res = maybe_call_get_parameters(
        client=client,
        get_parameters_ins=get_parameters_ins,
    )

    # Serialize response
    get_parameters_res_proto = serde.get_parameters_res_to_proto(get_parameters_res)
    return ClientMessage(get_parameters_res=get_parameters_res_proto)


def _fit(client: Client, fit_msg: ServerMessage.FitIns) -> ClientMessage:
    # Deserialize fit instruction
    fit_ins = serde.fit_ins_from_proto(fit_msg)

    # Perform fit
    fit_res = maybe_call_fit(
        client=client,
        fit_ins=fit_ins,
    )

    # Serialize fit result
    fit_res_proto = serde.fit_res_to_proto(fit_res)
    return ClientMessage(fit_res=fit_res_proto)


def _evaluate(client: Client, evaluate_msg: ServerMessage.EvaluateIns) -> ClientMessage:
    # Deserialize evaluate instruction
    evaluate_ins = serde.evaluate_ins_from_proto(evaluate_msg)

    # Perform evaluation
    evaluate_res = maybe_call_evaluate(
        client=client,
        evaluate_ins=evaluate_ins,
    )

    # Serialize evaluate result
    evaluate_res_proto = serde.evaluate_res_to_proto(evaluate_res)
    return ClientMessage(evaluate_res=evaluate_res_proto)


def _setup_param(client: Client, setup_param_msg: ServerMessage.SecAggMsg) -> ClientMessage:
    try:
        setup_param_ins = serde.setup_param_ins_from_proto(setup_param_msg)
        setup_param_res = client.setup_param(setup_param_ins)
        setup_param_res_proto = serde.setup_param_res_to_proto(setup_param_res)
        return ClientMessage(sec_agg_res=setup_param_res_proto)
    except Exception as e:
        print(e)
        return _error_res(e)


def _ask_keys(client: Client, ask_keys_msg: ServerMessage.SecAggMsg) -> ClientMessage:
    try:
        ask_keys_ins = serde.ask_keys_ins_from_proto(ask_keys_msg)
        ask_keys_res = client.ask_keys(ask_keys_ins)
        ask_keys_res_proto = serde.ask_keys_res_to_proto(ask_keys_res)
        return ClientMessage(sec_agg_res=ask_keys_res_proto)
    except Exception as e:
        return _error_res(e)


def _share_keys(client: Client, share_keys_msg: ServerMessage.SecAggMsg) -> ClientMessage:
    try:
        share_keys_in = serde.share_keys_ins_from_proto(share_keys_msg)
        share_keys_res = client.share_keys(share_keys_in)
        share_keys_res_proto = serde.share_keys_res_to_proto(share_keys_res)
        return ClientMessage(sec_agg_res=share_keys_res_proto)
    except Exception as e:
        return _error_res(e)


def _ask_vectors(client: Client, ask_vectors_msg: ServerMessage.SecAggMsg) -> ClientMessage:
    try:
        ask_vectors_ins = serde.ask_vectors_ins_from_proto(ask_vectors_msg)
        ask_vectors_res = client.ask_vectors(ask_vectors_ins)
        ask_vectors_res_proto = serde.ask_vectors_res_to_proto(ask_vectors_res)
        return ClientMessage(sec_agg_res=ask_vectors_res_proto)
    except Exception as e:
        return _error_res(e)

#TODO add consistency check
def _consistency_checks(client: Client, consistency_checks_msg: ServerMessage.SecAggMsg) -> ClientMessage:
    try:
        consistency_checks_ins = serde.consistency_checks_res_from_proto(consistency_checks_msg)
        consistency_checks_res = client.consistency_checks(consistency_checks_ins)
        consistency_checks_res_proto = serde.consistency_checks_res_to_proto(consistency_checks_res)
        return ClientMessage(sec_agg_res=consistency_checks_res_proto)
    except Exception as e:
        return _error_res(e)


def _unmask_vectors(client: Client, unmask_vectors_msg: ServerMessage.SecAggMsg) -> ClientMessage:
    try:
        unmask_vectors_ins = serde.unmask_vectors_ins_from_proto(unmask_vectors_msg)
        unmask_vectors_res = client.unmask_vectors(unmask_vectors_ins)
        unmask_vectors_res_proto = serde.unmask_vectors_res_to_proto(unmask_vectors_res)
        return ClientMessage(sec_agg_res=unmask_vectors_res_proto)
    except Exception as e:
        return _error_res(e)


def _error_res(e: Exception) -> ClientMessage:
    error_res = ClientMessage.SecAggRes(
        error_res=ClientMessage.SecAggRes.ErrorRes(error=e.args[0])
    )
    return ClientMessage(sec_agg_res=error_res)

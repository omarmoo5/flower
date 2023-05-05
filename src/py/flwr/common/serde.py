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
"""ProtoBuf serialization and deserialization."""


from typing import Any, List, cast

from flwr.proto.transport_pb2 import (
    ClientMessage,
    Code,
    Parameters,
    Reason,
    Scalar,
    ServerMessage,
    Status,
)

from . import typing

#  === ServerMessage message ===


def server_message_to_proto(server_message: typing.ServerMessage) -> ServerMessage:
    """Serialize `ServerMessage` to ProtoBuf."""
    if server_message.get_properties_ins is not None:
        return ServerMessage(
            get_properties_ins=get_properties_ins_to_proto(
                server_message.get_properties_ins,
            )
        )
    if server_message.get_parameters_ins is not None:
        return ServerMessage(
            get_parameters_ins=get_parameters_ins_to_proto(
                server_message.get_parameters_ins,
            )
        )
    if server_message.fit_ins is not None:
        return ServerMessage(
            fit_ins=fit_ins_to_proto(
                server_message.fit_ins,
            )
        )
    if server_message.evaluate_ins is not None:
        return ServerMessage(
            evaluate_ins=evaluate_ins_to_proto(
                server_message.evaluate_ins,
            )
        )
    raise Exception("No instruction set in ServerMessage, cannot serialize to ProtoBuf")


def server_message_from_proto(
    server_message_proto: ServerMessage,
) -> typing.ServerMessage:
    """Deserialize `ServerMessage` from ProtoBuf."""
    field = server_message_proto.WhichOneof("msg")
    if field == "get_properties_ins":
        return typing.ServerMessage(
            get_properties_ins=get_properties_ins_from_proto(
                server_message_proto.get_properties_ins,
            )
        )
    if field == "get_parameters_ins":
        return typing.ServerMessage(
            get_parameters_ins=get_parameters_ins_from_proto(
                server_message_proto.get_parameters_ins,
            )
        )
    if field == "fit_ins":
        return typing.ServerMessage(
            fit_ins=fit_ins_from_proto(
                server_message_proto.fit_ins,
            )
        )
    if field == "evaluate_ins":
        return typing.ServerMessage(
            evaluate_ins=evaluate_ins_from_proto(
                server_message_proto.evaluate_ins,
            )
        )
    raise Exception(
        "Unsupported instruction in ServerMessage, cannot deserialize from ProtoBuf"
    )


#  === ClientMessage message ===


def client_message_to_proto(client_message: typing.ClientMessage) -> ClientMessage:
    """Serialize `ClientMessage` to ProtoBuf."""
    if client_message.get_properties_res is not None:
        return ClientMessage(
            get_properties_res=get_properties_res_to_proto(
                client_message.get_properties_res,
            )
        )
    if client_message.get_parameters_res is not None:
        return ClientMessage(
            get_parameters_res=get_parameters_res_to_proto(
                client_message.get_parameters_res,
            )
        )
    if client_message.fit_res is not None:
        return ClientMessage(
            fit_res=fit_res_to_proto(
                client_message.fit_res,
            )
        )
    if client_message.evaluate_res is not None:
        return ClientMessage(
            evaluate_res=evaluate_res_to_proto(
                client_message.evaluate_res,
            )
        )
    raise Exception("No instruction set in ClientMessage, cannot serialize to ProtoBuf")


def client_message_from_proto(
    client_message_proto: ClientMessage,
) -> typing.ClientMessage:
    """Deserialize `ClientMessage` from ProtoBuf."""
    field = client_message_proto.WhichOneof("msg")
    if field == "get_properties_res":
        return typing.ClientMessage(
            get_properties_res=get_properties_res_from_proto(
                client_message_proto.get_properties_res,
            )
        )
    if field == "get_parameters_res":
        return typing.ClientMessage(
            get_parameters_res=get_parameters_res_from_proto(
                client_message_proto.get_parameters_res,
            )
        )
    if field == "fit_res":
        return typing.ClientMessage(
            fit_res=fit_res_from_proto(
                client_message_proto.fit_res,
            )
        )
    if field == "evaluate_res":
        return typing.ClientMessage(
            evaluate_res=evaluate_res_from_proto(
                client_message_proto.evaluate_res,
            )
        )
    raise Exception(
        "Unsupported instruction in ClientMessage, cannot deserialize from ProtoBuf"
    )


#  === Parameters message ===


def parameters_to_proto(parameters: typing.Parameters) -> Parameters:
    """Serialize `Parameters` to ProtoBuf."""
    return Parameters(tensors=parameters.tensors, tensor_type=parameters.tensor_type)


def parameters_from_proto(msg: Parameters) -> typing.Parameters:
    """Deserialize `Parameters` from ProtoBuf."""
    tensors: List[bytes] = list(msg.tensors)
    return typing.Parameters(tensors=tensors, tensor_type=msg.tensor_type)


#  === ReconnectIns message ===


def reconnect_ins_to_proto(ins: typing.ReconnectIns) -> ServerMessage.ReconnectIns:
    """Serialize `ReconnectIns` to ProtoBuf."""
    if ins.seconds is not None:
        return ServerMessage.ReconnectIns(seconds=ins.seconds)
    return ServerMessage.ReconnectIns()


def reconnect_ins_from_proto(msg: ServerMessage.ReconnectIns) -> typing.ReconnectIns:
    """Deserialize `ReconnectIns` from ProtoBuf."""
    return typing.ReconnectIns(seconds=msg.seconds)


# === DisconnectRes message ===


def disconnect_res_to_proto(res: typing.DisconnectRes) -> ClientMessage.DisconnectRes:
    """Serialize `DisconnectRes` to ProtoBuf."""
    reason_proto = Reason.UNKNOWN
    if res.reason == "RECONNECT":
        reason_proto = Reason.RECONNECT
    elif res.reason == "POWER_DISCONNECTED":
        reason_proto = Reason.POWER_DISCONNECTED
    elif res.reason == "WIFI_UNAVAILABLE":
        reason_proto = Reason.WIFI_UNAVAILABLE
    return ClientMessage.DisconnectRes(reason=reason_proto)


def disconnect_res_from_proto(msg: ClientMessage.DisconnectRes) -> typing.DisconnectRes:
    """Deserialize `DisconnectRes` from ProtoBuf."""
    if msg.reason == Reason.RECONNECT:
        return typing.DisconnectRes(reason="RECONNECT")
    if msg.reason == Reason.POWER_DISCONNECTED:
        return typing.DisconnectRes(reason="POWER_DISCONNECTED")
    if msg.reason == Reason.WIFI_UNAVAILABLE:
        return typing.DisconnectRes(reason="WIFI_UNAVAILABLE")
    return typing.DisconnectRes(reason="UNKNOWN")


# === GetParameters messages ===


def get_parameters_ins_to_proto(
    ins: typing.GetParametersIns,
) -> ServerMessage.GetParametersIns:
    """Serialize `GetParametersIns` to ProtoBuf."""
    config = properties_to_proto(ins.config)
    return ServerMessage.GetParametersIns(config=config)


def get_parameters_ins_from_proto(
    msg: ServerMessage.GetParametersIns,
) -> typing.GetParametersIns:
    """Deserialize `GetParametersIns` from ProtoBuf."""
    config = properties_from_proto(msg.config)
    return typing.GetParametersIns(config=config)


def get_parameters_res_to_proto(
    res: typing.GetParametersRes,
) -> ClientMessage.GetParametersRes:
    """Serialize `GetParametersRes` to ProtoBuf."""
    status_msg = status_to_proto(res.status)
    if res.status.code == typing.Code.GET_PARAMETERS_NOT_IMPLEMENTED:
        return ClientMessage.GetParametersRes(status=status_msg)
    parameters_proto = parameters_to_proto(res.parameters)
    return ClientMessage.GetParametersRes(
        status=status_msg, parameters=parameters_proto
    )


def get_parameters_res_from_proto(
    msg: ClientMessage.GetParametersRes,
) -> typing.GetParametersRes:
    """Deserialize `GetParametersRes` from ProtoBuf."""
    status = status_from_proto(msg=msg.status)
    parameters = parameters_from_proto(msg.parameters)
    return typing.GetParametersRes(status=status, parameters=parameters)


# === Fit messages ===


def fit_ins_to_proto(ins: typing.FitIns) -> ServerMessage.FitIns:
    """Serialize `FitIns` to ProtoBuf."""
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    return ServerMessage.FitIns(parameters=parameters_proto, config=config_msg)


def fit_ins_from_proto(msg: ServerMessage.FitIns) -> typing.FitIns:
    """Deserialize `FitIns` from ProtoBuf."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.FitIns(parameters=parameters, config=config)


def fit_res_to_proto(res: typing.FitRes) -> ClientMessage.FitRes:
    """Serialize `FitIns` to ProtoBuf."""
    status_msg = status_to_proto(res.status)
    if res.status.code == typing.Code.FIT_NOT_IMPLEMENTED:
        return ClientMessage.FitRes(status=status_msg)
    parameters_proto = parameters_to_proto(res.parameters)
    metrics_msg = None if res.metrics is None else metrics_to_proto(res.metrics)
    return ClientMessage.FitRes(
        status=status_msg,
        parameters=parameters_proto,
        num_examples=res.num_examples,
        metrics=metrics_msg,
    )


def fit_res_from_proto(msg: ClientMessage.FitRes) -> typing.FitRes:
    """Deserialize `FitRes` from ProtoBuf."""
    status = status_from_proto(msg=msg.status)
    parameters = parameters_from_proto(msg.parameters)
    metrics = None if msg.metrics is None else metrics_from_proto(msg.metrics)
    return typing.FitRes(
        status=status,
        parameters=parameters,
        num_examples=msg.num_examples,
        metrics=metrics,
    )


# === SecAgg Messages ===
# === Check if error ===
def check_error(msg: ClientMessage.SecAggRes):
    if msg.HasField("error_res"):
        raise Exception(msg.error_res.error)


# === Setup Param ===
def setup_param_ins_to_proto(
    setup_param_ins: typing.SetupParamIns,
) -> ServerMessage.SecAggMsg:
    return ServerMessage.SecAggMsg(
        setup_param=ServerMessage.SecAggMsg.SetupParam(
            sec_agg_param_dict=metrics_to_proto(setup_param_ins.sec_agg_param_dict)
        )
    )


def setup_param_ins_from_proto(
    setup_param_msg: ServerMessage.SecAggMsg,
) -> typing.SetupParamIns:
    return typing.SetupParamIns(
        sec_agg_param_dict=metrics_from_proto(setup_param_msg.setup_param.sec_agg_param_dict)
    )


def setup_param_res_to_proto(setup_param_res: typing.SetupParamRes):
    return ClientMessage.SecAggRes(
        setup_param_res=ClientMessage.SecAggRes.SetupParamRes()
    )


def setup_param_res_from_proto(setup_param_res: ServerMessage.SecAggMsg) -> typing.SetupParamRes:
    return typing.SetupParamRes()
# === Ask Keys ===


def ask_keys_ins_to_proto(Ask_keys_ins: typing.AskKeysIns) -> ServerMessage.SecAggMsg:
    config_msg = metrics_to_proto({})
    return ServerMessage.SecAggMsg(ask_keys=ServerMessage.SecAggMsg.AskKeys(config=config_msg))


def ask_keys_ins_from_proto(ask_keys_msg: ServerMessage.SecAggMsg) -> typing.AskKeysIns:
    return typing.AskKeysIns()


def ask_keys_res_to_proto(res: typing.AskKeysRes) -> ClientMessage.SecAggRes:
    return ClientMessage.SecAggRes(ask_keys_res=ClientMessage.SecAggRes.AskKeysRes(pk1=res.pk1,
                                                                                   pk2=res.pk2,
                                                                                   signature=res.signature,
                                                                                   sig_pub=res.sig_pub))


def ask_keys_res_from_proto(msg: ClientMessage.SecAggRes) -> typing.AskKeysRes:
    return typing.AskKeysRes(pk1=msg.ask_keys_res.pk1, pk2=msg.ask_keys_res.pk2,signature=msg.ask_keys_res.signature,sig_pub=msg.ask_keys_res.sig_pub)


# === Share Keys ===
def share_keys_ins_to_proto(share_keys_ins: typing.ShareKeysIns) -> ServerMessage.SecAggMsg:
    public_keys_dict = share_keys_ins.public_keys_dict
    proto_public_keys_dict = {}
    for i in public_keys_dict.keys():
        proto_public_keys_dict[i] = ServerMessage.SecAggMsg.ShareKeys.KeysPair(pk1=public_keys_dict[i].pk1,
                                                                               pk2=public_keys_dict[i].pk2,
                                                                               signature=public_keys_dict[i].signature,
                                                                               sig_pub=public_keys_dict[i].sig_pub
                                                                               )
    return ServerMessage.SecAggMsg(
        share_keys=ServerMessage.SecAggMsg.ShareKeys(
            public_keys_dict=proto_public_keys_dict
        )
    )


def share_keys_ins_from_proto(share_keys_msg: ServerMessage.SecAggMsg) -> typing.ShareKeysIns:
    proto_public_keys_dict = share_keys_msg.share_keys.public_keys_dict
    public_keys_dict = {}
    for i in proto_public_keys_dict.keys():
        public_keys_dict[i] = typing.AskKeysRes(pk1=proto_public_keys_dict[i].pk1,
                                                pk2=proto_public_keys_dict[i].pk2,
                                                signature=proto_public_keys_dict[i].signature,
                                                sig_pub=proto_public_keys_dict[i].sig_pub)
    return typing.ShareKeysIns(public_keys_dict=public_keys_dict)


def share_keys_res_to_proto(share_keys_res: typing.ShareKeysRes) -> ClientMessage.SecAggRes:
    share_keys_res_msg = ClientMessage.SecAggRes.ShareKeysRes()
    for packet in share_keys_res.share_keys_res_list:
        #print(("send stage 2", len(packet.ciphertext)))
        proto_packet = ClientMessage.SecAggRes.ShareKeysRes.Packet(
            source=packet.source, destination=packet.destination, ciphertext=packet.ciphertext
        )
        share_keys_res_msg.packet_list.append(proto_packet)
    return ClientMessage.SecAggRes(share_keys_res=share_keys_res_msg)


def share_keys_res_from_proto(share_keys_res_msg: ClientMessage.SecAggRes) -> typing.ShareKeysRes:
    proto_packet_list = share_keys_res_msg.share_keys_res.packet_list
    packet_list = []
    for proto_packet in proto_packet_list:
        packet = typing.ShareKeysPacket(
            source=proto_packet.source, destination=proto_packet.destination, ciphertext=proto_packet.ciphertext
        )
        #print(("receive stage 2", len(packet.ciphertext)))
        packet_list.append(packet)
    return typing.ShareKeysRes(share_keys_res_list=packet_list)

# === Ask vectors ===


def ask_vectors_ins_to_proto(ask_vectors_ins: typing.AskVectorsIns) -> ServerMessage.SecAggMsg:
    packet_list = ask_vectors_ins.ask_vectors_in_list
    proto_packet_list = []
    for packet in packet_list:
        proto_packet = ServerMessage.SecAggMsg.AskVectors.Packet(
            source=packet.source, destination=packet.destination, ciphertext=packet.ciphertext)
        #print(("send stage 3", len(packet.ciphertext)))
        proto_packet_list.append(proto_packet)
    fit_ins = ServerMessage.SecAggMsg.AskVectors.FitIns(parameters=parameters_to_proto(
        ask_vectors_ins.fit_ins.parameters), config=metrics_to_proto(ask_vectors_ins.fit_ins.config))
    return ServerMessage.SecAggMsg(ask_vectors=ServerMessage.SecAggMsg.AskVectors(packet_list=proto_packet_list, fit_ins=fit_ins))


def ask_vectors_ins_from_proto(ask_vectors_msg: ServerMessage.SecAggMsg) -> typing.AskVectorsIns:
    proto_packet_list = ask_vectors_msg.ask_vectors.packet_list
    packet_list = []
    for proto_packet in proto_packet_list:
        packet = typing.ShareKeysPacket(
            source=proto_packet.source, destination=proto_packet.destination, ciphertext=proto_packet.ciphertext)
        #print(("receive stage 3", len(proto_packet.ciphertext)))
        packet_list.append(packet)
    fit_ins = typing.FitIns(parameters=parameters_from_proto(
        ask_vectors_msg.ask_vectors.fit_ins.parameters), config=metrics_from_proto(ask_vectors_msg.ask_vectors.fit_ins.config))
    return typing.AskVectorsIns(ask_vectors_in_list=packet_list, fit_ins=fit_ins)


def ask_vectors_res_to_proto(ask_vectors_res: typing.AskVectorsRes) -> ClientMessage.SecAggRes:
    parameters_proto = parameters_to_proto(ask_vectors_res.parameters)
    return ClientMessage.SecAggRes(ask_vectors_res=ClientMessage.SecAggRes.AskVectorsRes(parameters=parameters_proto))


def ask_vectors_res_from_proto(ask_vectors_res_msg: ClientMessage.SecAggRes) -> typing.AskVectorsRes:
    parameters = parameters_from_proto(ask_vectors_res_msg.ask_vectors_res.parameters)
    return typing.AskVectorsRes(parameters=parameters)

# === Unmask Vectors ===


def unmask_vectors_ins_to_proto(unmask_vectors_ins: typing.UnmaskVectorsIns) -> ServerMessage.SecAggMsg:
    return ServerMessage.SecAggMsg(unmask_vectors=ServerMessage.SecAggMsg.UnmaskVectors(
        available_clients=unmask_vectors_ins.available_clients,
        dropout_clients=unmask_vectors_ins.dropout_clients))


def unmask_vectors_ins_from_proto(unmask_vectors_ins: ServerMessage.SecAggMsg) -> typing.UnmaskVectorsIns:
    return typing.UnmaskVectorsIns(
        available_clients=unmask_vectors_ins.unmask_vectors.available_clients,
        dropout_clients=unmask_vectors_ins.unmask_vectors.dropout_clients)


def unmask_vectors_res_to_proto(unmask_vectors_res: typing.UnmaskVectorsRes) -> ClientMessage.SecAggRes:
    return ClientMessage.SecAggRes(unmask_vectors_res=ClientMessage.SecAggRes.UnmaskVectorsRes(
        share_dict=unmask_vectors_res.share_dict))


def unmask_vectors_res_from_proto(unmask_vectors_res: ClientMessage.SecAggRes) -> typing.UnmaskVectorsRes:
    return typing.UnmaskVectorsRes(share_dict=unmask_vectors_res.unmask_vectors_res.share_dict)
# === GetProperties messages ===


def get_properties_ins_to_proto(
    ins: typing.GetPropertiesIns,
) -> ServerMessage.GetPropertiesIns:
    """Serialize `GetPropertiesIns` to ProtoBuf."""
    config = properties_to_proto(ins.config)
    return ServerMessage.GetPropertiesIns(config=config)


def get_properties_ins_from_proto(
    msg: ServerMessage.GetPropertiesIns,
) -> typing.GetPropertiesIns:
    """Deserialize `GetPropertiesIns` from ProtoBuf."""
    config = properties_from_proto(msg.config)
    return typing.GetPropertiesIns(config=config)


def get_properties_res_to_proto(
    res: typing.GetPropertiesRes,
) -> ClientMessage.GetPropertiesRes:
    """Serialize `GetPropertiesIns` to ProtoBuf."""
    status_msg = status_to_proto(res.status)
    if res.status.code == typing.Code.GET_PROPERTIES_NOT_IMPLEMENTED:
        return ClientMessage.GetPropertiesRes(status=status_msg)
    properties_msg = properties_to_proto(res.properties)
    return ClientMessage.GetPropertiesRes(status=status_msg, properties=properties_msg)


def get_properties_res_from_proto(
    msg: ClientMessage.GetPropertiesRes,
) -> typing.GetPropertiesRes:
    """Deserialize `GetPropertiesRes` from ProtoBuf."""
    status = status_from_proto(msg=msg.status)
    properties = properties_from_proto(msg.properties)
    return typing.GetPropertiesRes(status=status, properties=properties)


# === Evaluate messages ===


def evaluate_ins_to_proto(ins: typing.EvaluateIns) -> ServerMessage.EvaluateIns:
    """Serialize `EvaluateIns` to ProtoBuf."""
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    return ServerMessage.EvaluateIns(parameters=parameters_proto, config=config_msg)


def evaluate_ins_from_proto(msg: ServerMessage.EvaluateIns) -> typing.EvaluateIns:
    """Deserialize `EvaluateIns` from ProtoBuf."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.EvaluateIns(parameters=parameters, config=config)


def evaluate_res_to_proto(res: typing.EvaluateRes) -> ClientMessage.EvaluateRes:
    """Serialize `EvaluateIns` to ProtoBuf."""
    status_msg = status_to_proto(res.status)
    if res.status.code == typing.Code.EVALUATE_NOT_IMPLEMENTED:
        return ClientMessage.EvaluateRes(status=status_msg)
    metrics_msg = None if res.metrics is None else metrics_to_proto(res.metrics)
    return ClientMessage.EvaluateRes(
        status=status_msg,
        loss=res.loss,
        num_examples=res.num_examples,
        metrics=metrics_msg,
    )


def evaluate_res_from_proto(msg: ClientMessage.EvaluateRes) -> typing.EvaluateRes:
    """Deserialize `EvaluateRes` from ProtoBuf."""
    status = status_from_proto(msg=msg.status)
    metrics = None if msg.metrics is None else metrics_from_proto(msg.metrics)
    return typing.EvaluateRes(
        status=status,
        loss=msg.loss,
        num_examples=msg.num_examples,
        metrics=metrics,
    )


# === Status messages ===


def status_to_proto(status: typing.Status) -> Status:
    """Serialize `Status` to ProtoBuf."""
    code = Code.OK
    if status.code == typing.Code.GET_PROPERTIES_NOT_IMPLEMENTED:
        code = Code.GET_PROPERTIES_NOT_IMPLEMENTED
    if status.code == typing.Code.GET_PARAMETERS_NOT_IMPLEMENTED:
        code = Code.GET_PARAMETERS_NOT_IMPLEMENTED
    if status.code == typing.Code.FIT_NOT_IMPLEMENTED:
        code = Code.FIT_NOT_IMPLEMENTED
    if status.code == typing.Code.EVALUATE_NOT_IMPLEMENTED:
        code = Code.EVALUATE_NOT_IMPLEMENTED
    return Status(code=code, message=status.message)


def status_from_proto(msg: Status) -> typing.Status:
    """Deserialize `Status` from ProtoBuf."""
    code = typing.Code.OK
    if msg.code == Code.GET_PROPERTIES_NOT_IMPLEMENTED:
        code = typing.Code.GET_PROPERTIES_NOT_IMPLEMENTED
    if msg.code == Code.GET_PARAMETERS_NOT_IMPLEMENTED:
        code = typing.Code.GET_PARAMETERS_NOT_IMPLEMENTED
    if msg.code == Code.FIT_NOT_IMPLEMENTED:
        code = typing.Code.FIT_NOT_IMPLEMENTED
    if msg.code == Code.EVALUATE_NOT_IMPLEMENTED:
        code = typing.Code.EVALUATE_NOT_IMPLEMENTED
    return typing.Status(code=code, message=msg.message)


# === Properties messages ===


def properties_to_proto(properties: typing.Properties) -> Any:
    """Serialize `Properties` to ProtoBuf."""
    proto = {}
    for key in properties:
        proto[key] = scalar_to_proto(properties[key])
    return proto


def properties_from_proto(proto: Any) -> typing.Properties:
    """Deserialize `Properties` from ProtoBuf."""
    properties = {}
    for k in proto:
        properties[k] = scalar_from_proto(proto[k])
    return properties


# === Metrics messages ===


def metrics_to_proto(metrics: typing.Metrics) -> Any:
    """Serialize `Metrics` to ProtoBuf."""
    proto = {}
    for key in metrics:
        proto[key] = scalar_to_proto(metrics[key])
    return proto


def metrics_from_proto(proto: Any) -> typing.Metrics:
    """Deserialize `Metrics` from ProtoBuf."""
    metrics = {}
    for k in proto:
        metrics[k] = scalar_from_proto(proto[k])
    return metrics


# === Scalar messages ===


def scalar_to_proto(scalar: typing.Scalar) -> Scalar:
    """Serialize `Scalar` to ProtoBuf."""

    if isinstance(scalar, bool):
        return Scalar(bool=scalar)

    if isinstance(scalar, bytes):
        return Scalar(bytes=scalar)

    if isinstance(scalar, float):
        return Scalar(double=scalar)

    if isinstance(scalar, int):
        return Scalar(sint64=scalar)

    if isinstance(scalar, str):
        return Scalar(string=scalar)

    raise Exception(
        f"Accepted types: {bool, bytes, float, int, str} (but not {type(scalar)})"
    )


def scalar_from_proto(scalar_msg: Scalar) -> typing.Scalar:
    """Deserialize `Scalar` from ProtoBuf."""
    scalar_field = scalar_msg.WhichOneof("scalar")
    scalar = getattr(scalar_msg, cast(str, scalar_field))
    return cast(typing.Scalar, scalar)

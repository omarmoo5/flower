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
"""gRPC-based Flower ClientProxy implementation."""


from typing import Optional

from flwr import common
from flwr.common import serde
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage
from flwr.server.client_proxy import ClientProxy
from flwr.server.grpc_server.grpc_bridge import GrpcBridge, InsWrapper, ResWrapper
from flwr.common.typing import AskKeysIns, AskVectorsIns, AskVectorsRes, SetupParamIns, ShareKeysIns, ShareKeysRes, \
    UnmaskVectorsIns, UnmaskVectorsRes, ConsistencyCheckIns, ConsistencyCheckRes


class GrpcClientProxy(ClientProxy):
    """Flower ClientProxy that uses gRPC to delegate tasks over the network."""

    def __init__(
        self,
        cid: str,
        bridge: GrpcBridge,
    ):
        super().__init__(cid)
        self.bridge = bridge

    def get_properties(
        self,
        ins: common.GetPropertiesIns,
        timeout: Optional[float],
    ) -> common.GetPropertiesRes:
        """Requests client's set of internal properties."""
        get_properties_msg = serde.get_properties_ins_to_proto(ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(get_properties_ins=get_properties_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        get_properties_res = serde.get_properties_res_from_proto(
            client_msg.get_properties_res
        )
        return get_properties_res

    def get_parameters(
        self,
        ins: common.GetParametersIns,
        timeout: Optional[float],
    ) -> common.GetParametersRes:
        """Return the current local model parameters."""
        get_parameters_msg = serde.get_parameters_ins_to_proto(ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(get_parameters_ins=get_parameters_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        get_parameters_res = serde.get_parameters_res_from_proto(client_msg.get_parameters_res)
        return get_parameters_res

    def fit(
        self,
        ins: common.FitIns,
        timeout: Optional[float],
    ) -> common.FitRes:
        """Refine the provided parameters using the locally held dataset."""
        fit_ins_msg = serde.fit_ins_to_proto(ins)

        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(fit_ins=fit_ins_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        fit_res = serde.fit_res_from_proto(client_msg.fit_res)
        return fit_res

    def evaluate(
        self,
        ins: common.EvaluateIns,
        timeout: Optional[float],
    ) -> common.EvaluateRes:
        """Evaluate the provided parameters using the locally held dataset."""
        evaluate_msg = serde.evaluate_ins_to_proto(ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(evaluate_ins=evaluate_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        evaluate_res = serde.evaluate_res_from_proto(client_msg.evaluate_res)
        return evaluate_res

    def reconnect(
        self,
        ins: common.ReconnectIns,
        timeout: Optional[float],
    ) -> common.DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        reconnect_ins_msg = serde.reconnect_ins_to_proto(ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(reconnect_ins=reconnect_ins_msg),
                timeout=timeout,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        disconnect = serde.disconnect_res_from_proto(client_msg.disconnect_res)
        return disconnect

    def setup_param(self, setup_param_ins: SetupParamIns):
        setup_param_msg = serde.setup_param_ins_to_proto(setup_param_ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(sec_agg_msg=setup_param_msg),
                timeout=None,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        serde.check_error(client_msg.sec_agg_res)
        setup_param_res = serde.setup_param_res_from_proto(client_msg.sec_agg_res)
        return setup_param_res

    def ask_keys(self, ask_keys_ins: AskKeysIns) -> common.AskKeysRes:
        ask_keys_msg = serde.ask_keys_ins_to_proto(ask_keys_ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message= ServerMessage(sec_agg_msg=ask_keys_msg),
                timeout=None,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        serde.check_error(client_msg.sec_agg_res)
        ask_keys_res = serde.ask_keys_res_from_proto(client_msg.sec_agg_res)
        return ask_keys_res

    def share_keys(self, share_keys_ins: ShareKeysIns) -> ShareKeysRes:
        share_keys_msg = serde.share_keys_ins_to_proto(share_keys_ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(sec_agg_msg=share_keys_msg),
                timeout=None,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        serde.check_error(client_msg.sec_agg_res)
        share_keys_res = serde.share_keys_res_from_proto(client_msg.sec_agg_res)
        return share_keys_res

    def ask_vectors(self, ask_vectors_ins: AskVectorsIns) -> AskVectorsRes:
        ask_vectors_msg = serde.ask_vectors_ins_to_proto(ask_vectors_ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(sec_agg_msg=ask_vectors_msg),
                timeout=None,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        serde.check_error(client_msg.sec_agg_res)
        ask_vectors_res = serde.ask_vectors_res_from_proto(client_msg.sec_agg_res)
        return ask_vectors_res

    #TODO consistency check proxy
    def consistency_checks(self,consistency_checks_ins: ConsistencyCheckIns) -> ConsistencyCheckRes:
        consistency_checks_msg = serde.consistency_checks_ins_to_proto(consistency_checks_ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(sec_agg_msg=consistency_checks_msg),
                timeout=None,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        serde.check_error(client_msg.sec_agg_res)
        consistency_checks_res = serde.consistency_checks_res_from_proto(client_msg.sec_agg_res)
        return consistency_checks_res

    def unmask_vectors(self, unmask_vectors_ins: UnmaskVectorsIns) -> UnmaskVectorsRes:
        unmask_vectors_msg = serde.unmask_vectors_ins_to_proto(unmask_vectors_ins)
        res_wrapper: ResWrapper = self.bridge.request(
            ins_wrapper=InsWrapper(
                server_message=ServerMessage(sec_agg_msg=unmask_vectors_msg),
                timeout=None,
            )
        )
        client_msg: ClientMessage = res_wrapper.client_message
        serde.check_error(client_msg.sec_agg_res)
        unmask_vectors_res = serde.unmask_vectors_res_from_proto(client_msg.sec_agg_res)
        return unmask_vectors_res

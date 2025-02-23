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
from typing import Dict, Tuple

from flwr.client import NumPyClient
from flwr.common import (
    AskKeysRes,
)
from flwr.common.sec_agg import sec_agg_client_logic
from flwr.common.typing import (AskKeysIns, AskVectorsIns,
                                AskVectorsRes, SetupParamIns,
                                ShareKeysIns, ShareKeysRes,
                                UnmaskVectorsIns, UnmaskVectorsRes, NDArrays, Scalar, SetupParamRes,
                                ConsistencyCheckIns, ConsistencyCheckRes)


class SecAggClient(NumPyClient):
    """Wrapper which adds SecAgg methods."""

    def __init__(self, c: NumPyClient) -> None:
        self.client = c

    def __str__(self):
        return "Wrapper for SecAgg Client"

    def get_parameters(self, ins: Dict[str, Scalar]) -> NDArrays:
        """Return the current local model parameters."""
        return self.client.get_parameters(ins)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        return self.client.fit(parameters, config)

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        return self.client.evaluate(parameters, config)

    def setup_param(self, setup_param_ins: SetupParamIns) -> SetupParamRes:
        return sec_agg_client_logic.setup_param(self, setup_param_ins)

    def ask_keys(self, ask_keys_ins: AskKeysIns) -> AskKeysRes:
        return sec_agg_client_logic.ask_keys(self, ask_keys_ins)

    def share_keys(self, share_keys_in: ShareKeysIns) -> ShareKeysRes:
        return sec_agg_client_logic.share_keys(self, share_keys_in)

    def ask_vectors(self, ask_vectors_ins: AskVectorsIns) -> AskVectorsRes:
        return sec_agg_client_logic.ask_vectors(self, ask_vectors_ins)

    def consistency_checks(self, consistency_checks_ins: ConsistencyCheckIns) -> ConsistencyCheckRes:
        return sec_agg_client_logic.consistency_checks(self, consistency_checks_ins)

    def unmask_vectors(self, unmask_vectors_ins: UnmaskVectorsIns) -> UnmaskVectorsRes:
        return sec_agg_client_logic.unmask_vectors(self, unmask_vectors_ins)

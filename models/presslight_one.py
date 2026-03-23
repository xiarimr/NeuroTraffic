"""
PressLight agent, based on LIT model structure.
"One" means parameter sharing, Ape-X solution.
Observations: [cur_phase, lane_num_vechile_in, lane_num_vehicle_out]
Reward: -Pressure
"""

import numpy as np
import random
import torch
import torch.nn as nn

from .network_agent import NetworkAgent


class PressLightNet(nn.Module):
    def __init__(self, feature_dims, hidden_dim, num_actions, phase_map):
        super().__init__()
        self.phase_map = {tuple(v): idx for idx, v in phase_map.items()}
        self.shared_hidden = nn.Linear(sum(feature_dims), hidden_dim)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_actions),
            )
            for _ in range(len(phase_map))
        ])

    def forward(self, inputs):
        cur_phase = inputs[0].float()
        all_flatten_feature = torch.cat([x.float() for x in inputs], dim=1)
        shared_dense = torch.sigmoid(self.shared_hidden(all_flatten_feature))

        outputs = []
        for row_idx in range(cur_phase.shape[0]):
            phase_key = tuple(int(v) for v in cur_phase[row_idx].tolist())
            branch_idx = self.phase_map.get(phase_key, 0)
            outputs.append(self.branches[branch_idx](shared_dense[row_idx:row_idx + 1]))
        return torch.cat(outputs, dim=0)


class PressLightAgentOne(NetworkAgent):
    def build_network(self):
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
        feature_dims = [8 if "cur_phase" in feat_name else 12 for feat_name in used_feature]
        return PressLightNet(
            feature_dims=feature_dims,
            hidden_dim=self.dic_agent_conf["D_DENSE"],
            num_actions=self.num_actions,
            phase_map=self.dic_traffic_env_conf["PHASE"],
        )

    def prepare_Xs_Y(self, memory):
        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))

        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
        memory_after_forget = memory[ind_sta: ind_end]
        print("memory size after forget:", len(memory_after_forget))

        sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
        sample_slice = random.sample(memory_after_forget, sample_size)
        print("memory samples number:", sample_size)

        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
        _state = [[] for _ in range(len(used_feature))]
        _next_state = [[] for _ in range(len(used_feature))]
        _action = []
        _reward = []
        for i in range(len(sample_slice)):
            state, action, next_state, reward, _, _, _ = sample_slice[i]
            for feat_idx, feat_name in enumerate(used_feature):
                _state[feat_idx].append(state[feat_name])
                _next_state[feat_idx].append(next_state[feat_name])
            _action.append(action)
            _reward.append(reward)

        _state2 = [np.array(ss, dtype=np.float32) for ss in _state]
        _next_state2 = [np.array(ss, dtype=np.float32) for ss in _next_state]

        cur_qvalues = self._forward(self.q_network, _state2)
        next_qvalues = self._forward(self.q_network_bar, _next_state2)
        target = np.copy(cur_qvalues)

        for i in range(len(sample_slice)):
            target[i, _action[i]] = _reward[i] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                    np.max(next_qvalues[i, :])
        self.Xs = _state2
        self.Y = target.astype(np.float32)

    def choose_action(self, count, states):
        dic_state_feature_arrays = {}
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            dic_state_feature_arrays[feature_name] = []

        for s in states:
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if feature_name == "cur_phase":
                    dic_state_feature_arrays[feature_name].append(
                        self.dic_traffic_env_conf['PHASE'][s[feature_name][0]])
                else:
                    dic_state_feature_arrays[feature_name].append(s[feature_name])

        state_input = [np.array(dic_state_feature_arrays[feature_name], dtype=np.float32) for feature_name in
                       self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]
        q_values = self._forward(self.q_network, state_input)
        if random.random() <= self.dic_agent_conf["EPSILON"]:
            action = np.random.randint(len(q_values[0]), size=len(q_values))
        else:
            action = np.argmax(q_values, axis=1)

        return action

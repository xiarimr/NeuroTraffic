"""
MPLight agent, based on FRAP model structure.
Observations: [cur_phase, traffic_movement_pressure_num]
Reward: -Pressure
"""

import numpy as np
import random
import torch
import torch.nn as nn

from .network_agent import NetworkAgent, relation


class MPLightNet(nn.Module):
    def __init__(self, num_actions, d_dense, phase_list, lane_order, dic_index):
        super().__init__()
        self.num_actions = num_actions
        self.phase_list = phase_list
        self.lane_order = lane_order
        self.dic_index = dic_index
        self.phase_embed = nn.Embedding(2, 4)
        self.num_vec_mapping = nn.Linear(1, 4)
        self.lane_embedding = nn.Linear(8, 16)
        self.relation_embedding = nn.Embedding(2, 4)
        self.lane_conv = nn.Conv2d(32, d_dense, kernel_size=1)
        self.relation_conv = nn.Conv2d(4, d_dense, kernel_size=1)
        self.combine_conv = nn.Conv2d(d_dense, d_dense, kernel_size=1)
        self.before_merge = nn.Conv2d(d_dense, 1, kernel_size=1)

        relation_matrix = relation(np.zeros((1, 12), dtype=np.float32), phase_list)
        self.register_buffer("relation_matrix", torch.as_tensor(relation_matrix, dtype=torch.long))

    def forward(self, inputs):
        feat1, feat2 = inputs
        feat1 = feat1.long()
        phase_embeddings = torch.sigmoid(self.phase_embed(feat1))

        lane_features = {}
        for i, movement in enumerate(self.lane_order):
            idx = self.dic_index[movement]
            lane_scalar = feat2[:, idx:idx + 1]
            mapped_vec = torch.sigmoid(self.num_vec_mapping(lane_scalar))
            phase_vec = phase_embeddings[:, i, :]
            lane_features[movement] = torch.cat([mapped_vec, phase_vec], dim=-1)

        phase_pressures = []
        for phase in self.phase_list:
            m1, m2 = phase.split("_")
            p1 = torch.relu(self.lane_embedding(lane_features[m1]))
            p2 = torch.relu(self.lane_embedding(lane_features[m2]))
            phase_pressures.append(p1 + p2)

        recombined = []
        num_phase = len(phase_pressures)
        for i in range(num_phase):
            for j in range(num_phase):
                if i != j:
                    recombined.append(torch.cat([phase_pressures[i], phase_pressures[j]], dim=-1))

        feature_map = torch.stack(recombined, dim=1)
        feature_map = feature_map.reshape(feat2.shape[0], num_phase, num_phase - 1, 32).permute(0, 3, 1, 2)

        relation_embedding = self.relation_embedding(
            self.relation_matrix.expand(feat2.shape[0], -1, -1)
        ).permute(0, 3, 1, 2)

        lane_conv = torch.relu(self.lane_conv(feature_map))
        relation_conv = torch.relu(self.relation_conv(relation_embedding))
        combine_feature = lane_conv * relation_conv
        hidden_layer = torch.relu(self.combine_conv(combine_feature))
        before_merge = self.before_merge(hidden_layer).squeeze(1)
        return before_merge.sum(dim=2)


class MPLightAgent(NetworkAgent):
    def build_network(self):
        dic_index = {
            "WL": 0,
            "WT": 1,
            "EL": 3,
            "ET": 4,
            "NL": 6,
            "NT": 7,
            "SL": 9,
            "ST": 10,
        }
        return MPLightNet(
            num_actions=self.num_actions,
            d_dense=self.dic_agent_conf["D_DENSE"],
            phase_list=self.dic_traffic_env_conf["PHASE_LIST"],
            lane_order=self.dic_traffic_env_conf["list_lane_order"],
            dic_index=dic_index,
        )

    def convert_state_to_input(self, s):
        if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
            inputs = []
            for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if feature == "cur_phase":
                    inputs.append(np.array([self.dic_traffic_env_conf['PHASE'][s[feature][0]]], dtype=np.float32))
                else:
                    inputs.append(np.array([s[feature]], dtype=np.float32))
            return inputs
        return [np.array([s[feature]], dtype=np.float32) for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]

    def choose_action(self, count, states):
        dic_state_feature_arrays = {}
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:2]
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            dic_state_feature_arrays[feature_name] = []
        for s in states:
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if feature_name == "cur_phase":
                    dic_state_feature_arrays[feature_name].append(self.dic_traffic_env_conf['PHASE'][s[feature_name][0]])
                else:
                    dic_state_feature_arrays[feature_name].append(s[feature_name])
        state_input = [np.array(dic_state_feature_arrays[feature_name], dtype=np.float32) for feature_name in used_feature]

        q_values = self._forward(self.q_network, state_input)
        if random.random() <= self.dic_agent_conf["EPSILON"]:
            action = np.random.randint(len(q_values[0]), size=len(q_values))
        else:
            action = np.argmax(q_values, axis=1)

        return action

    def prepare_Xs_Y(self, memory):
        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))

        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
        memory_after_forget = memory[ind_sta: ind_end]
        print("memory size after forget:", len(memory_after_forget))

        sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
        sample_slice = random.sample(memory_after_forget, sample_size)
        print("memory samples number:", sample_size)

        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:2]
        _state = [[], []]
        _next_state = [[], []]
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

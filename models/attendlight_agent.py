import numpy as np
import random
import torch
import torch.nn as nn

from .network_agent import NetworkAgent


class AttendLightNet(nn.Module):
    def __init__(self, num_lane, num_phases, num_actions, phase_map):
        super().__init__()
        self.num_lane = num_lane
        self.num_phases = num_phases
        self.phase_map = phase_map
        self.feat_proj = nn.Linear(4, 32)
        self.phase_attention = nn.MultiheadAttention(32, 4, batch_first=True)
        self.global_attention = nn.MultiheadAttention(32, 4, batch_first=True)
        self.fc1 = nn.Linear(32, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)
        self.num_actions = num_actions

    def forward(self, inputs):
        ins0 = inputs if isinstance(inputs, torch.Tensor) else inputs[0]
        feat1 = ins0.reshape(ins0.shape[0], self.num_lane * 2, 4)
        feat1 = torch.relu(self.feat_proj(feat1))

        lane_feats = torch.split(feat1, 1, dim=1)
        phase_feats_map = []
        for i in range(self.num_phases):
            tmp_feat = torch.cat([lane_feats[idx] for idx in self.phase_map[i]], dim=1)
            tmp_feat_mean = tmp_feat.mean(dim=1, keepdim=True)
            tmp_out, _ = self.phase_attention(tmp_feat_mean, tmp_feat, tmp_feat)
            phase_feats_map.append(tmp_out)

        phase_feat_all = torch.cat(phase_feats_map, dim=1)
        phase_attention, _ = self.global_attention(phase_feat_all, phase_feat_all, phase_feat_all)

        hidden = torch.relu(self.fc1(phase_attention))
        hidden = torch.relu(self.fc2(hidden))
        q_values = self.fc3(hidden).squeeze(-1)
        return q_values


class AttendLightAgent(NetworkAgent):
    def build_network(self):
        return AttendLightNet(
            num_lane=self.num_lane,
            num_phases=self.num_phases,
            num_actions=self.num_actions,
            phase_map=self.phase_map,
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
        _state = []
        _next_state = []
        _action = []
        _reward = []
        for i in range(len(sample_slice)):
            state, action, next_state, reward, _, _, _ = sample_slice[i]
            _state.append(state[used_feature[0]])
            _next_state.append(next_state[used_feature[0]])
            _action.append(action)
            _reward.append(reward)

        _state2 = np.array(_state, dtype=np.float32)
        _next_state2 = np.array(_next_state, dtype=np.float32)

        cur_qvalues = self._forward(self.q_network, _state2)
        next_qvalues = self._forward(self.q_network_bar, _next_state2)
        target = np.copy(cur_qvalues)
        for i in range(len(sample_slice)):
            target[i, _action[i]] = _reward[i] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                    np.max(next_qvalues[i, :])

        self.Xs = _state2
        self.Y = target.astype(np.float32)

    def choose_action(self, step_num, states):
        feats = []
        for s in states:
            tmp_feat0 = s[self.dic_traffic_env_conf["LIST_STATE_FEATURE"][0]]
            feats.append(tmp_feat0)

        feats = np.array(feats, dtype=np.float32)
        q_values = self._forward(self.q_network, feats)

        action = self.epsilon_choice(q_values)
        return action

    def epsilon_choice(self, q_values):
        max_1 = np.expand_dims(np.argmax(q_values, axis=-1), axis=-1)
        rand_1 = np.random.randint(self.num_actions, size=(len(q_values), 1))
        _p = np.concatenate([max_1, rand_1], axis=-1)
        select = np.random.choice([0, 1], size=len(q_values), p=[1 - self.dic_agent_conf["EPSILON"],
                                                                 self.dic_agent_conf["EPSILON"]])
        act = _p[np.arange(len(q_values)), select]
        return act


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, loss):
        score = -loss
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop

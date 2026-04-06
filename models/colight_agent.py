"""
Colight agent.
observations: [lane_num_vehicle, cur_phase]
reward: -queue_length
"""

import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .agent import Agent
from .colight_net import ColightEncoder
from utils.phase_utils import get_phase_encoding


def build_memory():
    return []


class CoLightQNet(nn.Module):
    def __init__(self, input_dim, cnn_layers, num_agents, num_neighbors, num_actions):
        super().__init__()
        self.encoder = ColightEncoder(input_dim, [32, 32], cnn_layers, num_agents, num_neighbors)
        self.q_head = nn.Linear(self.encoder.output_dim, num_actions)

    def forward(self, inputs):
        features, adjacency = inputs
        encoded = self.encoder(features.float(), adjacency.float())
        return self.q_head(encoded)


class CoLightAgent(Agent):
    def __init__(self, dic_agent_conf=None, dic_traffic_env_conf=None, dic_path=None, cnt_round=None,
                 intersection_id="0"):
        super(CoLightAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.CNN_layers = dic_agent_conf['CNN_layers']
        self.num_agents = dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_neighbors = min(dic_traffic_env_conf['TOP_K_ADJACENCY'], self.num_agents)

        self.num_actions = len(self.dic_traffic_env_conf["PHASE"])
        self.len_feature = self._cal_len_feature()
        self.memory = build_memory()
        self.Xs, self.Y = None, None

        if cnt_round == 0:
            self.q_network = self.build_network().to(self.device)
            if os.listdir(self.dic_path["PATH_TO_MODEL"]):
                try:
                    self.load_network("round_0_inter_{0}".format(intersection_id))
                except Exception:
                    self.q_network = self.build_network().to(self.device)
            self.q_network_bar = self.build_network_from_copy(self.q_network)
        else:
            try:
                self.load_network("round_{0}_inter_{1}".format(cnt_round - 1, self.intersection_id))
                if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                    if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            max((cnt_round - 1) // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"] * self.dic_agent_conf[
                                "UPDATE_Q_BAR_FREQ"], 0),
                            self.intersection_id))
                    else:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                            self.intersection_id))
                else:
                    self.load_network_bar("round_{0}_inter_{1}".format(
                        max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))
            except Exception:
                print("fail to load network, current round: {0}".format(cnt_round))
                self.q_network = self.build_network().to(self.device)
                self.q_network_bar = self.build_network_from_copy(self.q_network)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
        self.loss_fn = nn.MSELoss()

        decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
        self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

    def _checkpoint_path(self, file_name, file_path=None):
        base_dir = file_path or self.dic_path["PATH_TO_MODEL"]
        return os.path.join(base_dir, "%s.pt" % file_name)

    def _legacy_checkpoint_exists(self, file_name, file_path=None):
        base_dir = file_path or self.dic_path["PATH_TO_MODEL"]
        legacy_candidates = [
            os.path.join(base_dir, "%s.h5" % file_name),
            os.path.join(base_dir, "%s.keras" % file_name),
        ]
        return any(os.path.exists(path) for path in legacy_candidates)

    def _to_tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _forward(self, model, inputs):
        model.eval()
        with torch.no_grad():
            outputs = model([self._to_tensor(inputs[0]), self._to_tensor(inputs[1])])
        return outputs.detach().cpu().numpy()

    def _cal_len_feature(self):
        total = 0
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        for feat_name in used_feature:
            if "cur_phase" in feat_name:
                total += 8
            else:
                total += 12
        return total

    def adjacency_index2matrix(self, adjacency_index):
        adjacency_index_new = np.sort(adjacency_index, axis=-1)
        eye = np.eye(self.num_agents, dtype=np.float32)
        return eye[adjacency_index_new]

    def convert_state_to_input(self, s):
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        feats0 = []
        adj = []
        for i in range(self.num_agents):
            adj.append(s[i]["adjacency_matrix"])
            tmp = []
            for feature in used_feature:
                if feature == "cur_phase":
                    if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
                        tmp.extend(get_phase_encoding(self.dic_traffic_env_conf['PHASE'], s[i][feature][0]))
                    else:
                        tmp.extend(s[i][feature])
                else:
                    tmp.extend(s[i][feature])
            feats0.append(tmp)
        feats = np.array([feats0], dtype=np.float32)
        adj = self.adjacency_index2matrix(np.array([adj]))
        return [feats, adj.astype(np.float32)]

    def choose_action(self, count, states):
        xs = self.convert_state_to_input(states)
        q_values = self._forward(self.q_network, xs)
        if random.random() <= self.dic_agent_conf["EPSILON"]:
            action = np.random.randint(self.num_actions, size=len(q_values[0]))
        else:
            action = np.argmax(q_values[0], axis=1)
        return action

    @staticmethod
    def _concat_list(ls):
        tmp = []
        for i in range(len(ls)):
            tmp += ls[i]
        return [tmp]

    def prepare_Xs_Y(self, memory):
        valid_lengths = [len(one_inter_samples) for one_inter_samples in memory if len(one_inter_samples) > 0]
        if len(valid_lengths) == 0:
            self.Xs = None
            self.Y = None
            return
        slice_size = min(valid_lengths)
        adjs = []
        state_data = [[] for _ in range(self.num_agents)]
        next_state_data = [[] for _ in range(self.num_agents)]
        action_data = [[] for _ in range(self.num_agents)]
        reward_data = [[] for _ in range(self.num_agents)]

        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]

        for i in range(slice_size):
            one_adj = []
            for j in range(self.num_agents):
                state, action, next_state, reward, _, _, _ = memory[j][i]
                action_data[j].append(action)
                reward_data[j].append(reward)
                one_adj.append(state["adjacency_matrix"])
                state_data[j].append(self._concat_list([state[used_feature[k]] for k in range(len(used_feature))]))
                next_state_data[j].append(
                    self._concat_list([next_state[used_feature[k]] for k in range(len(used_feature))]))
            adjs.append(one_adj)

        adjs = self.adjacency_index2matrix(np.array(adjs)).astype(np.float32)
        states = np.concatenate([np.array(ss, dtype=np.float32) for ss in state_data], axis=1)
        next_states = np.concatenate([np.array(ss, dtype=np.float32) for ss in next_state_data], axis=1)
        target = self._forward(self.q_network, [states, adjs])
        next_state_qvalues = self._forward(self.q_network_bar, [next_states, adjs])
        final_target = np.copy(target)
        for i in range(slice_size):
            for j in range(self.num_agents):
                final_target[i, j, action_data[j][i]] = reward_data[j][i] / self.dic_agent_conf["NORMAL_FACTOR"] + \
                                                        self.dic_agent_conf["GAMMA"] * np.max(next_state_qvalues[i, j])

        self.Xs = [states, adjs]
        self.Y = final_target.astype(np.float32)

    def build_network(self, MLP_layers=None):
        return CoLightQNet(
            input_dim=self.len_feature,
            cnn_layers=self.CNN_layers,
            num_agents=self.num_agents,
            num_neighbors=self.num_neighbors,
            num_actions=self.num_actions,
        )

    def train_network(self):
        if self.Xs is None or self.Y is None or len(self.Y) == 0:
            return

        epochs = self.dic_agent_conf["EPOCHS"]
        total_size = len(self.Y)
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], total_size)
        val_size = int(total_size * 0.3)
        train_size = max(total_size - val_size, 1)

        states = np.asarray(self.Xs[0], dtype=np.float32)
        adjs = np.asarray(self.Xs[1], dtype=np.float32)
        targets = np.asarray(self.Y, dtype=np.float32)

        train_states = states[:train_size]
        train_adjs = adjs[:train_size]
        train_targets = targets[:train_size]
        if val_size > 0:
            val_states = states[train_size:]
            val_adjs = adjs[train_size:]
            val_targets = targets[train_size:]
        else:
            val_states = train_states
            val_adjs = train_adjs
            val_targets = train_targets

        best_state = copy.deepcopy(self.q_network.state_dict())
        best_val_loss = float("inf")
        patience = self.dic_agent_conf["PATIENCE"]
        patience_count = 0

        for _ in range(epochs):
            self.q_network.train()
            indices = np.arange(train_size)
            for start in range(0, train_size, batch_size):
                batch_idx = indices[start:start + batch_size]
                batch_states = self._to_tensor(train_states[batch_idx])
                batch_adjs = self._to_tensor(train_adjs[batch_idx])
                batch_targets = self._to_tensor(train_targets[batch_idx])

                self.optimizer.zero_grad()
                outputs = self.q_network([batch_states, batch_adjs])
                loss = self.loss_fn(outputs, batch_targets)
                loss.backward()
                self.optimizer.step()

            current_val_loss = self._evaluate_loss(val_states, val_adjs, val_targets)
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_state = copy.deepcopy(self.q_network.state_dict())
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    break

        self.q_network.load_state_dict(best_state)
        self.q_network.eval()

    def _evaluate_loss(self, states, adjs, targets):
        self.q_network.eval()
        with torch.no_grad():
            outputs = self.q_network([self._to_tensor(states), self._to_tensor(adjs)])
            loss = self.loss_fn(outputs, self._to_tensor(targets))
        return float(loss.detach().cpu().item())

    def build_network_from_copy(self, network_copy):
        network = copy.deepcopy(network_copy).to(self.device)
        network.eval()
        return network

    def load_network(self, file_name, file_path=None):
        model_path = self._checkpoint_path(file_name, file_path)
        if not os.path.exists(model_path) and self._legacy_checkpoint_exists(file_name, file_path):
            raise FileNotFoundError(
                "Found legacy Keras checkpoints for %s, but CoLight now expects a PyTorch .pt checkpoint." % file_name
            )
        self.q_network = self.build_network().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.q_network.load_state_dict(state_dict)
        self.q_network.eval()
        print("succeed in loading model %s" % file_name)

    def load_network_bar(self, file_name, file_path=None):
        model_path = self._checkpoint_path(file_name, file_path)
        if not os.path.exists(model_path) and self._legacy_checkpoint_exists(file_name, file_path):
            raise FileNotFoundError(
                "Found legacy Keras checkpoints for %s, but CoLight now expects a PyTorch .pt checkpoint." % file_name
            )
        self.q_network_bar = self.build_network().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.q_network_bar.load_state_dict(state_dict)
        self.q_network_bar.eval()
        print("succeed in loading model %s" % file_name)

    def save_network(self, file_name):
        torch.save(self.q_network.state_dict(), self._checkpoint_path(file_name))

    def save_network_bar(self, file_name):
        torch.save(self.q_network_bar.state_dict(), self._checkpoint_path(file_name))

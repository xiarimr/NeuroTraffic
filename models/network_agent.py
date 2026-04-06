import copy
import os
import random
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .agent import Agent
from utils.phase_utils import get_phase_encoding


class NetworkAgent(Agent):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id="0"):
        super(NetworkAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id=intersection_id)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = len(dic_traffic_env_conf["PHASE"])
        self.num_phases = len(dic_traffic_env_conf["PHASE"])

        self.memory = self.build_memory()
        self.cnt_round = cnt_round

        self.Xs, self.Y = None, None
        self.num_lane = dic_traffic_env_conf["NUM_LANE"]
        self.phase_map = dic_traffic_env_conf["PHASE_MAP"]

        if cnt_round == 0:
            if os.listdir(self.dic_path["PATH_TO_MODEL"]):
                try:
                    self.load_network("round_0_inter_{0}".format(intersection_id))
                except Exception:
                    self.q_network = self.build_network().to(self.device)
            else:
                self.q_network = self.build_network().to(self.device)
            self.q_network_bar = self.build_network_from_copy(self.q_network)
        else:
            try:
                self.load_network("round_{0}_inter_{1}".format(cnt_round - 1, self.intersection_id))

                if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                    if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            max((cnt_round - 1) // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"] *
                                self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))
                    else:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                            self.intersection_id))
                else:
                    self.load_network_bar("round_{0}_inter_{1}".format(
                        max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))
            except Exception:
                print('traceback.format_exc():\n%s' % traceback.format_exc())
                self.q_network = self.build_network().to(self.device)
                self.q_network_bar = self.build_network_from_copy(self.q_network)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
        self.loss_fn = nn.MSELoss()

        decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
        self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

    def get_feature_dim(self, feature_name):
        if "cur_phase" in feature_name:
            if self.dic_traffic_env_conf.get("BINARY_PHASE_EXPANSION", False):
                phase_encoding = next(iter(self.dic_traffic_env_conf["PHASE"].values()))
                return len(phase_encoding)
            return 1

        if feature_name == "adjacency_matrix":
            return min(
                self.dic_traffic_env_conf["TOP_K_ADJACENCY"],
                self.dic_traffic_env_conf["NUM_INTERSECTIONS"],
            )

        if feature_name == "pressure":
            return self.num_lane * 2

        if feature_name == "num_in_seg_attend":
            return self.num_lane * 8

        if feature_name in {
            "time_this_phase",
            "queue_length",
            "delay",
            "throughput",
            "phase_switch",
            "pressure_total",
            "main_road_queue_length",
            "main_road_throughput",
        }:
            return 1

        if feature_name in {
            "lane_num_vehicle",
            "lane_num_vehicle_downstream",
            "delta_lane_num_vehicle",
            "lane_num_waiting_vehicle_in",
            "lane_num_waiting_vehicle_out",
            "traffic_movement_pressure_queue",
            "traffic_movement_pressure_queue_efficient",
            "traffic_movement_pressure_num",
            "lane_enter_running_part",
        }:
            return self.num_lane

        raise ValueError("Unsupported feature dimension inference for feature: {0}".format(feature_name))

    def get_feature_dims(self, feature_names):
        return [self.get_feature_dim(feature_name) for feature_name in feature_names]

    def _checkpoint_path(self, file_name, file_path=None):
        base_dir = file_path or self.dic_path["PATH_TO_MODEL"]
        return os.path.join(base_dir, "%s.pt" % file_name)

    def _to_tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _prepare_inputs(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return [self._to_tensor(item) for item in inputs]
        return self._to_tensor(inputs)

    def _forward(self, model, inputs):
        prepared_inputs = self._prepare_inputs(inputs)
        model.eval()
        with torch.no_grad():
            outputs = model(prepared_inputs)
        return outputs.detach().cpu().numpy()

    def load_network(self, file_name, file_path=None):
        model_path = self._checkpoint_path(file_name, file_path)
        if not os.path.exists(model_path):
            legacy_dir = file_path or self.dic_path["PATH_TO_MODEL"]
            legacy_candidates = [
                os.path.join(legacy_dir, "%s.h5" % file_name),
                os.path.join(legacy_dir, "%s.keras" % file_name),
            ]
            if any(os.path.exists(path) for path in legacy_candidates):
                raise FileNotFoundError(
                    "Found legacy Keras checkpoints for %s, but this agent now expects a PyTorch .pt checkpoint." %
                    file_name
                )
        self.q_network = self.build_network().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.q_network.load_state_dict(state_dict)
        self.q_network.eval()
        print("succeed in loading model %s" % file_name)

    def load_network_transfer(self, file_name, file_path=None):
        self.load_network(file_name, file_path or self.dic_path["PATH_TO_TRANSFER_MODEL"])

    def load_network_bar(self, file_name, file_path=None):
        model_path = self._checkpoint_path(file_name, file_path)
        if not os.path.exists(model_path):
            legacy_dir = file_path or self.dic_path["PATH_TO_MODEL"]
            legacy_candidates = [
                os.path.join(legacy_dir, "%s.h5" % file_name),
                os.path.join(legacy_dir, "%s.keras" % file_name),
            ]
            if any(os.path.exists(path) for path in legacy_candidates):
                raise FileNotFoundError(
                    "Found legacy Keras checkpoints for %s, but this agent now expects a PyTorch .pt checkpoint." %
                    file_name
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

    def build_network(self):
        raise NotImplementedError

    @staticmethod
    def build_memory():
        return []

    def build_network_from_copy(self, network_copy):
        network = copy.deepcopy(network_copy).to(self.device)
        network.eval()
        return network

    def prepare_Xs_Y(self, memory):
        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))
        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
        memory_after_forget = memory[ind_sta: ind_end]
        print("memory size after forget:", len(memory_after_forget))

        sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
        sample_slice = random.sample(memory_after_forget, sample_size)
        print("memory samples number:", sample_size)

        dic_state_feature_arrays = {}
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            dic_state_feature_arrays[feature_name] = []
        Y = []

        for i in range(len(sample_slice)):
            state, action, next_state, reward, instant_reward, _, _ = sample_slice[i]
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                dic_state_feature_arrays[feature_name].append(state[feature_name])
            _state = []
            _next_state = []
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                _state.append(np.array([state[feature_name]]))
                _next_state.append(np.array([next_state[feature_name]]))

            target = self._forward(self.q_network, _state)
            next_state_qvalues = self._forward(self.q_network_bar, _next_state)

            if self.dic_agent_conf["LOSS_FUNCTION"] == "mean_squared_error":
                final_target = np.copy(target[0])
                final_target[action] = reward / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                       np.max(next_state_qvalues[0])
            elif self.dic_agent_conf["LOSS_FUNCTION"] == "categorical_crossentropy":
                raise NotImplementedError

            Y.append(final_target)

        self.Xs = [np.array(dic_state_feature_arrays[feature_name]) for feature_name in
                   self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]
        self.Y = np.array(Y, dtype=np.float32)

    def convert_state_to_input(self, s):
        if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
            inputs = []
            for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if "cur_phase" in feature:
                    inputs.append(np.array([get_phase_encoding(
                        self.dic_traffic_env_conf['PHASE'], s[feature][0]
                    )], dtype=np.float32))
                else:
                    inputs.append(np.array([s[feature]], dtype=np.float32))
            return inputs
        else:
            return [np.array([s[feature]], dtype=np.float32) for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]

    def choose_action(self, count, state):
        state_input = self.convert_state_to_input(state)
        q_values = self._forward(self.q_network, state_input)
        if random.random() <= self.dic_agent_conf["EPSILON"]:
            action = random.randrange(len(q_values[0]))
        else:
            action = int(np.argmax(q_values[0]))
        return action

    def train_network(self):
        if self.Xs is None or self.Y is None or len(self.Y) == 0:
            return

        epochs = self.dic_agent_conf["EPOCHS"]
        total_size = len(self.Y)
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], total_size)
        val_size = int(total_size * 0.3)
        train_size = max(total_size - val_size, 1)

        Xs = [np.asarray(x, dtype=np.float32) for x in self.Xs]
        Y = np.asarray(self.Y, dtype=np.float32)

        train_inputs = [x[:train_size] for x in Xs]
        train_targets = Y[:train_size]
        if val_size > 0:
            val_inputs = [x[train_size:] for x in Xs]
            val_targets = Y[train_size:]
        else:
            val_inputs = None
            val_targets = None

        best_state = copy.deepcopy(self.q_network.state_dict())
        best_val_loss = float("inf")
        patience = self.dic_agent_conf["PATIENCE"]
        patience_count = 0

        for _ in range(epochs):
            self.q_network.train()
            indices = np.arange(train_size)
            for start in range(0, train_size, batch_size):
                batch_idx = indices[start:start + batch_size]
                batch_inputs = [self._to_tensor(x[batch_idx]) for x in train_inputs]
                batch_targets = self._to_tensor(train_targets[batch_idx])

                self.optimizer.zero_grad()
                outputs = self.q_network(batch_inputs)
                loss = self.loss_fn(outputs, batch_targets)
                loss.backward()
                self.optimizer.step()

            current_val_loss = self._evaluate_loss(val_inputs, val_targets) if val_inputs is not None else \
                self._evaluate_loss(train_inputs, train_targets)
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

    def _evaluate_loss(self, inputs, targets):
        self.q_network.eval()
        with torch.no_grad():
            preds = self.q_network([self._to_tensor(x) for x in inputs])
            target_tensor = self._to_tensor(targets)
            loss = self.loss_fn(preds, target_tensor)
        return float(loss.detach().cpu().item())


class Selector(object):
    def __init__(self, select, d_phase_encoding, d_action, **kwargs):
        self.select = select
        self.d_phase_encoding = d_phase_encoding
        self.d_action = d_action


def slice_tensor(x, index):
    if x.ndim == 3:
        return x[:, index, :]
    if x.ndim == 2:
        return x[:, index:index + 1]
    raise ValueError("Unsupported tensor shape for slice_tensor: %s" % (x.shape,))


def relation(x, phase_list):
    relations = []
    num_phase = len(phase_list)
    if num_phase == 8:
        for p1 in phase_list:
            zeros = [0, 0, 0, 0, 0, 0, 0]
            count = 0
            for p2 in phase_list:
                if p1 == p2:
                    continue
                m1 = p1.split("_")
                m2 = p2.split("_")
                if len(list(set(m1 + m2))) == 3:
                    zeros[count] = 1
                count += 1
            relations.append(zeros)
        return np.array(relations).reshape((1, 8, 7))
    return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]).reshape((1, 4, 3))


class RepeatVector3D(nn.Module):
    def __init__(self, times):
        super().__init__()
        self.times = times

    def forward(self, inputs):
        return inputs.unsqueeze(1).repeat(1, self.times, 1, 1)

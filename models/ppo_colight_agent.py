"""
PPO version of CoLight agent.
observations: [lane_num_vehicle, cur_phase, adjacency_matrix]
reward: -queue_length
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .agent import Agent
from .colight_net import ColightEncoder
from utils.phase_utils import get_phase_encoding


class PPOCoLightNet(nn.Module):
    def __init__(self, input_dim, cnn_layers, num_agents, num_neighbors, num_actions):
        super().__init__()
        self.encoder = ColightEncoder(input_dim, [32, 32], cnn_layers, num_agents, num_neighbors)
        self.actor_logits = nn.Linear(self.encoder.output_dim, num_actions)
        self.critic_value = nn.Linear(self.encoder.output_dim, 1)

    def forward(self, inputs):
        features, adjacency = inputs
        encoded = self.encoder(features.float(), adjacency.float())
        actor_logits = self.actor_logits(encoded)
        actor_probs = torch.softmax(actor_logits, dim=-1)
        critic_value = self.critic_value(encoded)
        return actor_probs, critic_value


class PPOCoLightAgent(Agent):
    def __init__(self, dic_agent_conf=None, dic_traffic_env_conf=None, dic_path=None, cnt_round=None,
                 intersection_id="0"):
        super(PPOCoLightAgent, self).__init__(dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.CNN_layers = dic_agent_conf['CNN_layers']
        self.num_agents = dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_neighbors = min(dic_traffic_env_conf['TOP_K_ADJACENCY'], self.num_agents)
        self.num_actions = len(self.dic_traffic_env_conf["PHASE"])
        self.len_feature = self._cal_len_feature()

        self.ppo_clip = self.dic_agent_conf.get("PPO_CLIP", 0.2)
        self.value_coef = self.dic_agent_conf.get("PPO_VALUE_COEF", 0.5)
        self.entropy_coef = self.dic_agent_conf.get("PPO_ENTROPY_COEF", 0.01)
        self.max_grad_norm = self.dic_agent_conf.get("PPO_MAX_GRAD_NORM", 0.5)

        if cnt_round == 0:
            self.policy_model = self.build_network().to(self.device)
            if os.listdir(self.dic_path["PATH_TO_MODEL"]):
                try:
                    self.load_network("round_0_inter_{0}".format(intersection_id))
                except Exception:
                    self.policy_model = self.build_network().to(self.device)
        else:
            try:
                self.load_network("round_{0}_inter_{1}".format(cnt_round - 1, self.intersection_id))
            except Exception:
                print("fail to load network, current round: {0}".format(cnt_round))
                self.policy_model = self.build_network().to(self.device)

        decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
        self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

        self.Xs = None
        self.actions = None
        self.old_action_probs = None
        self.advantages = None
        self.returns = None

        self.optimizer = optim.Adam(
            self.policy_model.parameters(),
            lr=self.dic_agent_conf["LEARNING_RATE"]
        )

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

    @staticmethod
    def _to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.asarray(x)

    def _cal_len_feature(self):
        feat_len = 0
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        for feat_name in used_feature:
            if "cur_phase" in feat_name:
                feat_len += 8
            else:
                feat_len += 12
        return feat_len

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
        with torch.no_grad():
            policy_probs, _ = self.policy_model([self._to_tensor(xs[0]), self._to_tensor(xs[1])])
        policy_probs = self._to_numpy(policy_probs)[0]

        if self.dic_agent_conf["EPSILON"] <= 0:
            return np.argmax(policy_probs, axis=1)

        actions = []
        for i in range(self.num_agents):
            if np.random.rand() < self.dic_agent_conf["EPSILON"]:
                actions.append(np.random.choice(self.num_actions, p=policy_probs[i]))
            else:
                actions.append(int(np.argmax(policy_probs[i])))
        return np.array(actions)

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
        actions = np.array(action_data, dtype=np.int32).T
        rewards = np.array(reward_data, dtype=np.float32).T / float(self.dic_agent_conf["NORMAL_FACTOR"])

        with torch.no_grad():
            policy_probs, values = self.policy_model([self._to_tensor(states), self._to_tensor(adjs)])
            _, next_values = self.policy_model([self._to_tensor(next_states), self._to_tensor(adjs)])
        policy_probs = self._to_numpy(policy_probs)
        values = np.squeeze(self._to_numpy(values), axis=-1)
        next_values = np.squeeze(self._to_numpy(next_values), axis=-1)

        batch_idx = np.arange(slice_size)[:, None]
        agent_idx = np.arange(self.num_agents)[None, :]
        old_action_probs = policy_probs[batch_idx, agent_idx, actions]

        td_target = rewards + self.dic_agent_conf["GAMMA"] * next_values
        advantages = td_target - values
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        self.Xs = [states, adjs]
        self.actions = actions
        self.old_action_probs = np.clip(old_action_probs, 1e-8, 1.0).astype(np.float32)
        self.advantages = advantages.astype(np.float32)
        self.returns = td_target.astype(np.float32)

    def build_network(self, MLP_layers=None):
        return PPOCoLightNet(
            input_dim=self.len_feature,
            cnn_layers=self.CNN_layers,
            num_agents=self.num_agents,
            num_neighbors=self.num_neighbors,
            num_actions=self.num_actions,
        )

    def train_network(self):
        if self.Xs is None:
            return

        epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.actions))

        states = self.Xs[0]
        adjs = self.Xs[1]
        actions = self.actions
        old_action_probs = self.old_action_probs
        advantages = self.advantages
        returns = self.returns

        data_size = states.shape[0]

        for _ in range(epochs):
            indices = np.arange(data_size)
            np.random.shuffle(indices)

            for start in range(0, data_size, batch_size):
                end = min(start + batch_size, data_size)
                mb_idx = indices[start:end]

                mb_states = self._to_tensor(states[mb_idx])
                mb_adjs = self._to_tensor(adjs[mb_idx])
                mb_actions = self._to_tensor(actions[mb_idx], dtype=torch.int64)
                mb_old_probs = self._to_tensor(old_action_probs[mb_idx])
                mb_adv = self._to_tensor(advantages[mb_idx])
                mb_returns = self._to_tensor(returns[mb_idx])

                self.optimizer.zero_grad()

                new_probs, values = self.policy_model([mb_states, mb_adjs])
                values = torch.squeeze(values, dim=-1)

                action_mask = F.one_hot(mb_actions, num_classes=self.num_actions).to(torch.float32)
                chosen_new_probs = torch.sum(new_probs * action_mask, dim=-1)
                chosen_new_probs = torch.clamp(chosen_new_probs, min=1e-8, max=1.0)

                ratio = chosen_new_probs / torch.clamp(mb_old_probs, min=1e-8, max=1.0)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * mb_adv
                actor_loss = -torch.mean(torch.minimum(surr1, surr2))

                critic_loss = torch.mean((mb_returns - values) ** 2)
                entropy = -torch.mean(torch.sum(new_probs * torch.log(torch.clamp(new_probs, min=1e-8, max=1.0)), dim=-1))

                total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def load_network(self, file_name, file_path=None):
        model_path = self._checkpoint_path(file_name, file_path)
        if not os.path.exists(model_path) and self._legacy_checkpoint_exists(file_name, file_path):
            raise FileNotFoundError(
                "Found legacy Keras checkpoints for %s, but PPOCoLight now expects a PyTorch .pt checkpoint." %
                file_name
            )
        self.policy_model = self.build_network().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.policy_model.load_state_dict(state_dict)
        self.policy_model.eval()
        self.optimizer = optim.Adam(
            self.policy_model.parameters(),
            lr=self.dic_agent_conf["LEARNING_RATE"]
        )
        print("succeed in loading model %s" % file_name)

    def save_network(self, file_name):
        torch.save(self.policy_model.state_dict(), self._checkpoint_path(file_name))

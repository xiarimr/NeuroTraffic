"""
PPO version of CoLight agent.
observations: [lane_num_vehicle, cur_phase, adjacency_matrix]
reward: -queue_length
"""
import numpy as np
import os
import torch
import torch.nn.functional as F
from .keras_backend import setup_keras_backend

setup_keras_backend()

from keras import backend as K
from keras import ops
from keras import Input, Model
from keras.layers import Dense, Lambda, Layer, Permute, Reshape
from keras.models import load_model
from keras.utils import to_categorical
from .agent import Agent


class PPOCoLightAgent(Agent):
    def __init__(self, dic_agent_conf=None, dic_traffic_env_conf=None, dic_path=None, cnt_round=None,
                 intersection_id="0"):
        super(PPOCoLightAgent, self).__init__(dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)

        self.CNN_layers = dic_agent_conf['CNN_layers']
        self.num_agents = dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_neighbors = min(dic_traffic_env_conf['TOP_K_ADJACENCY'], self.num_agents)
        self.num_actions = len(self.dic_traffic_env_conf["PHASE"])
        self.len_feature = self._cal_len_feature()

        # PPO hyper-parameters (with defaults compatible with old conf files)
        self.ppo_clip = self.dic_agent_conf.get("PPO_CLIP", 0.2)
        self.value_coef = self.dic_agent_conf.get("PPO_VALUE_COEF", 0.5)
        self.entropy_coef = self.dic_agent_conf.get("PPO_ENTROPY_COEF", 0.01)
        self.max_grad_norm = self.dic_agent_conf.get("PPO_MAX_GRAD_NORM", 0.5)

        if cnt_round == 0:
            self.policy_model = self.build_network()
            if os.listdir(self.dic_path["PATH_TO_MODEL"]):
                model_path = os.path.join(self.dic_path["PATH_TO_MODEL"],
                                          "round_0_inter_{0}.keras".format(intersection_id))
                if os.path.exists(model_path):
                    self.policy_model = load_model(model_path, custom_objects={'RepeatVector3D': RepeatVector3D})
                    self.torch_optimizer = torch.optim.Adam(
                        self._torch_trainable_params(),
                        lr=self.dic_agent_conf["LEARNING_RATE"]
                    )
        else:
            try:
                self.load_network("round_{0}_inter_{1}".format(cnt_round - 1, self.intersection_id))
            except Exception:
                print("fail to load network, current round: {0}".format(cnt_round))
                self.policy_model = self.build_network()

        decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
        self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

        self.Xs = None
        self.actions = None
        self.old_action_probs = None
        self.advantages = None
        self.returns = None

        if not hasattr(self, "torch_optimizer"):
            self.torch_optimizer = torch.optim.Adam(
                self._torch_trainable_params(),
                lr=self.dic_agent_conf["LEARNING_RATE"]
            )

    @staticmethod
    def _to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.asarray(x)

    def _torch_trainable_params(self):
        params = []
        for var in self.policy_model.trainable_variables:
            params.append(var.value if hasattr(var, "value") else var)
        return params

    def _cal_len_feature(self):
        feat_len = 0
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        for feat_name in used_feature:
            if "cur_phase" in feat_name:
                feat_len += 8
            else:
                feat_len += 12
        return feat_len

    @staticmethod
    def MLP(ins, layers=None):
        if layers is None:
            layers = [128, 128]
        for layer_index, layer_size in enumerate(layers):
            if layer_index == 0:
                h = Dense(layer_size, activation='relu', kernel_initializer='random_normal',
                          name='Dense_embed_%d' % layer_index)(ins)
            else:
                h = Dense(layer_size, activation='relu', kernel_initializer='random_normal',
                          name='Dense_embed_%d' % layer_index)(h)
        return h

    def MultiHeadsAttModel(self, in_feats, in_nei, d_in=128, h_dim=16, dout=128, head=8, suffix=-1):
        agent_repr = Reshape((self.num_agents, 1, d_in))(in_feats)
        neighbor_repr = RepeatVector3D(self.num_agents)(in_feats)
        neighbor_repr = Lambda(lambda x: ops.matmul(x[0], x[1]))([in_nei, neighbor_repr])

        agent_repr_head = Dense(h_dim * head, activation='relu', kernel_initializer='random_normal',
                                name='agent_repr_%d' % suffix)(agent_repr)
        agent_repr_head = Reshape((self.num_agents, 1, h_dim, head))(agent_repr_head)
        agent_repr_head = Permute((1, 4, 2, 3))(agent_repr_head)

        neighbor_repr_head = Dense(h_dim * head, activation='relu', kernel_initializer='random_normal',
                                   name='neighbor_repr_%d' % suffix)(neighbor_repr)
        neighbor_repr_head = Reshape((self.num_agents, self.num_neighbors, h_dim, head))(neighbor_repr_head)
        neighbor_repr_head = Permute((1, 4, 2, 3))(neighbor_repr_head)

        att = Lambda(lambda x: ops.softmax(ops.matmul(x[0], ops.swapaxes(x[1], -1, -2)), axis=-1))(
            [agent_repr_head, neighbor_repr_head])
        neighbor_hidden_repr_head = Dense(h_dim * head, activation='relu', kernel_initializer='random_normal',
                                          name='neighbor_hidden_repr_%d' % suffix)(neighbor_repr)
        neighbor_hidden_repr_head = Reshape((self.num_agents, self.num_neighbors, h_dim, head))(
            neighbor_hidden_repr_head)
        neighbor_hidden_repr_head = Permute((1, 4, 2, 3))(neighbor_hidden_repr_head)
        out = Lambda(lambda x: ops.mean(ops.matmul(x[0], x[1]), axis=2))([att, neighbor_hidden_repr_head])
        out = Reshape((self.num_agents, h_dim))(out)
        out = Dense(dout, activation="relu", kernel_initializer='random_normal',
                    name='MLP_after_relation_%d' % suffix)(out)
        return out

    def adjacency_index2matrix(self, adjacency_index):
        adjacency_index_new = np.sort(adjacency_index, axis=-1)
        return to_categorical(adjacency_index_new, num_classes=self.num_agents)

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
                        tmp.extend(self.dic_traffic_env_conf['PHASE'][s[i][feature][0]])
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
        policy_probs, _ = self.policy_model(xs, training=False)
        policy_probs = self._to_numpy(policy_probs)[0]

        # Keep behavior deterministic when EPSILON == 0 (used by test pipeline).
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

        policy_probs, values = self.policy_model([states, adjs], training=False)
        _, next_values = self.policy_model([next_states, adjs], training=False)
        policy_probs = self._to_numpy(policy_probs)
        values = np.squeeze(self._to_numpy(values), axis=-1)
        next_values = np.squeeze(self._to_numpy(next_values), axis=-1)

        batch_idx = np.arange(slice_size)[:, None]
        agent_idx = np.arange(self.num_agents)[None, :]
        old_action_probs = policy_probs[batch_idx, agent_idx, actions]

        # One-step advantage estimate for each intersection.
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

    def build_network(self, MLP_layers=[32, 32]):
        CNN_layers = self.CNN_layers
        CNN_heads = [5] * len(CNN_layers)

        features_in = Input(shape=(self.num_agents, self.len_feature), name="feature")
        adjacency_in = Input(shape=(self.num_agents, self.num_neighbors, self.num_agents), name="adjacency_matrix")

        h = self.MLP(features_in, MLP_layers)
        for layer_index, layer_size in enumerate(CNN_layers):
            if layer_index == 0:
                h = self.MultiHeadsAttModel(
                    h,
                    adjacency_in,
                    d_in=MLP_layers[-1],
                    h_dim=layer_size[0],
                    dout=layer_size[1],
                    head=CNN_heads[layer_index],
                    suffix=layer_index
                )
            else:
                h = self.MultiHeadsAttModel(
                    h,
                    adjacency_in,
                    d_in=CNN_layers[layer_index - 1][1],
                    h_dim=layer_size[0],
                    dout=layer_size[1],
                    head=CNN_heads[layer_index],
                    suffix=layer_index
                )

        actor_logits = Dense(self.num_actions, kernel_initializer='random_normal', name='actor_logits')(h)
        actor_probs = Lambda(lambda x: ops.softmax(x, axis=-1), name='actor_probs')(actor_logits)
        critic_value = Dense(1, kernel_initializer='random_normal', name='critic_value')(h)

        model = Model(inputs=[features_in, adjacency_in], outputs=[actor_probs, critic_value])
        return model

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

                mb_states = torch.as_tensor(states[mb_idx], dtype=torch.float32)
                mb_adjs = torch.as_tensor(adjs[mb_idx], dtype=torch.float32)
                mb_actions = torch.as_tensor(actions[mb_idx], dtype=torch.int64)
                mb_old_probs = torch.as_tensor(old_action_probs[mb_idx], dtype=torch.float32)
                mb_adv = torch.as_tensor(advantages[mb_idx], dtype=torch.float32)
                mb_returns = torch.as_tensor(returns[mb_idx], dtype=torch.float32)

                params = self._torch_trainable_params()
                optimizer = self.torch_optimizer
                optimizer.zero_grad()

                new_probs, values = self.policy_model([mb_states, mb_adjs], training=True)
                values = torch.squeeze(values, dim=-1)

                action_mask = F.one_hot(mb_actions, num_classes=self.num_actions).to(torch.float32)
                chosen_new_probs = torch.sum(new_probs * action_mask, dim=-1)
                chosen_new_probs = torch.clamp(chosen_new_probs, min=1e-8, max=1.0)

                ratio = chosen_new_probs / torch.clamp(mb_old_probs, min=1e-8, max=1.0)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * mb_adv
                actor_loss = -torch.mean(torch.minimum(surr1, surr2))

                critic_loss = torch.mean(torch.square(mb_returns - values))
                entropy = -torch.mean(torch.sum(new_probs * torch.log(torch.clamp(new_probs, min=1e-8, max=1.0)), dim=-1))

                total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                optimizer.step()

    def load_network(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.policy_model = load_model(
            os.path.join(file_path, "%s.keras" % file_name),
            custom_objects={'RepeatVector3D': RepeatVector3D})
        self.torch_optimizer = torch.optim.Adam(
            self._torch_trainable_params(),
            lr=self.dic_agent_conf["LEARNING_RATE"]
        )
        print("succeed in loading model %s" % file_name)

    def save_network(self, file_name):
        self.policy_model.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.keras" % file_name))


class RepeatVector3D(Layer):
    def __init__(self, times, **kwargs):
        super(RepeatVector3D, self).__init__(**kwargs)
        self.times = times

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.times, input_shape[1], input_shape[2]

    def call(self, inputs):
        return K.tile(K.expand_dims(inputs, 1), [1, self.times, 1, 1])

    def get_config(self):
        config = {'times': self.times}
        base_config = super(RepeatVector3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers

# import agent_config
from configs import agent_config
import copy

tf.config.experimental_run_functions_eagerly(True)


def build_network(state_dim, action_dim, sub_action_num):
    print("build Braching DQ network")
    state_input = layers.Input(shape=(state_dim,))  # input layer
    # build shared representation hidden layer
    shared_representation = layers.Dense(512, activation="relu")(state_input)
    shared_representation = layers.Dense(256, activation="relu")(shared_representation)
    # build common state value layer
    common_state_value = layers.Dense(128, activation="relu")(shared_representation)
    common_state_value = layers.Dense(1)(common_state_value)
    # build action branch q value layer iteratively
    subaction_q_layers = []
    subaction_advantage_layers = []
    for _ in range(action_dim):
        action_branch_layer = layers.Dense(128, activation="relu")(shared_representation)
        action_branch_layer = layers.Dense(sub_action_num)(action_branch_layer)
        subaction_advantage_layers.append(action_branch_layer)
        subaction_q_value = common_state_value + (action_branch_layer - tf.reduce_mean(action_branch_layer))
        # subaction_q_value = common_state_value + (action_branch_layer -tf.expand_dims( tf.reduce_mean(action_branch_layer,1),1))
        # subaction_q_value = common_state_value + (action_branch_layer - tf.maximum(action_branch_layer))
        subaction_q_layers.append(subaction_q_value)
    subaction_q_layers = tf.stack(subaction_q_layers, axis=1)
    model = tf.keras.Model(state_input, subaction_q_layers)
    return model

class AdaptiveBDQ_agent:
    def __init__(self,
                 env_config
                 ):

        self.state_dim = env_config["ITSX_STATE_DIM"] * env_config["ACTION_DIM"]
        self.action_dim = env_config["ACTION_DIM"]
        self.subaction_num = env_config["ITSX_ACTION_DIM"]

        # build model
        self.eval_model = build_network(self.state_dim, self.action_dim, self.subaction_num)
        self.target_model = build_network(self.state_dim, self.action_dim, self.subaction_num)
        self.target_model.set_weights(self.eval_model.get_weights())
        AGENT_CONFIG = copy.deepcopy(agent_config.AGENT_CONFIG)
        AGENT_CONFIG.update(agent_config.BDQ_AGENT_CONFIG)
        print(AGENT_CONFIG)
        self.td_operator_type = AGENT_CONFIG["TD_OPERATOR"]

        # policy parameters
        self.max_epsilon = AGENT_CONFIG["MAX_EPSILON"]
        self.epsilon = self.max_epsilon
        self.min_epsilon = AGENT_CONFIG["MIN_EPSILON"]
        self.decay_steps = AGENT_CONFIG["DECAY_STEPS"]

        self.gamma = AGENT_CONFIG["GAMMA"]
        self.learn_count = 0


        # memory replay
        self.memory_counter = 0
        self.memory_size = AGENT_CONFIG["MEMORY_SIZE"]
        self.batch_size = AGENT_CONFIG["BATCH_SIZE"]
        self.state_memory = np.zeros((self.memory_size, self.state_dim))
        self.action_memory = np.zeros((self.memory_size, self.action_dim))
        self.reward_memory = np.zeros((self.memory_size, 1))
        self.next_state_memory = np.zeros((self.memory_size, self.state_dim))

        # target network update setting
        self.replace_target_iter = AGENT_CONFIG["REPLACE_INTERVAL"]
        self.replace_count = 0
        self.replace_mode = AGENT_CONFIG["NET_REPLACE_TYPE"]
        if self.replace_mode == 'HARD':
            self.tau = 1
        else:
            self.tau = AGENT_CONFIG["TAU"]
        self.loss_his = []
        #
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=AGENT_CONFIG["LEARNING_RATE"])
        # print()

    def choose_action(self, state, idle_id=None):
        """
        in dynamic DBQ, some branches are not activated.
        The idle_id store which branch is not activated
        for idle branches, the action is replaced by -1.
        :param state:
        :param idle_id: a list of idle_branch. the elements in idle_id < self.action_dim
        :return:
        """
        # print()
        state = state[np.newaxis, :]
        action_branch_value = self.eval_model(state)[0]
        joint_action = []
        if np.random.random() > self.epsilon:
            # greedy choice
            for i in range(self.action_dim):
                joint_action.append(np.argmax(action_branch_value[i]))
        else:
            # random
            for i in range(self.action_dim):
                joint_action.append(np.random.randint(0, self.subaction_num))
        if idle_id is not None:
            #
            for id in idle_id:
                joint_action[id] = -1
        return np.array(joint_action)

    def learn(self):
        # print("learn")
        available_count = min(self.memory_size, self.memory_counter)
        self.learn_count += 1
        if available_count > 10000:
            self.replace_count += 1
            batch_indices = np.random.choice(available_count, self.batch_size)

            s = tf.convert_to_tensor(self.state_memory[batch_indices])
            a = tf.convert_to_tensor(self.action_memory[batch_indices])
            r = tf.convert_to_tensor(self.reward_memory[batch_indices])
            r = tf.cast(r, dtype=tf.float32)
            s_ = tf.convert_to_tensor(self.next_state_memory[batch_indices])
            # update_start=time.time()
            self.update_gradient(s, a, r, s_)
            # print("update ",time.time()-update_start)
            if self.replace_count % self.replace_target_iter == 0:
                print("replace para")
                # start_time=time.time()
                self.replace_para(self.target_model.variables, self.eval_model.variables)
                # print("replace cost",time.time()-start_time)
        # update epsilon
        fraction = min(float(self.learn_count) / self.decay_steps, 1)
        self.epsilon = self.max_epsilon + fraction * (self.min_epsilon - self.max_epsilon)
        # self.epsilon *= 0.99965467#0.999965
        if self.epsilon < 0.001:
            self.epsilon = 0.001

    @tf.function
    def replace_para(self, target_var, source_var):
        for (a, b) in zip(target_var, source_var):
            a.assign(a * (1 - self.tau) + b * self.tau)

    @tf.function
    def update_gradient(self, s, a, r, s_):
        with tf.GradientTape() as tape:
            q_eval = self.eval_model(s)
            q_target = q_eval.numpy().copy()
            # mask construction
            eval_act_index = a.numpy().astype(int)  # action index of experience
            eval_act_index_mask = tf.one_hot(eval_act_index, self.subaction_num)
            eval_act_index_reverse_mask = tf.one_hot(eval_act_index, self.subaction_num, on_value=0.0,
                                                     off_value=1.0)  # we need reversed one hot matrix to reserve the value of not eval action index
            idle_index = np.where(eval_act_index == -1)
            idle_mask = np.ones((self.batch_size, self.action_dim))
            idle_mask[idle_index] = 0
            # batch_index=np.where(eval_act_index == -1)
            q_target = tf.multiply(q_target, eval_act_index_reverse_mask)  # remove the value on eval index

            # on policy estimate next value
            q_next_eval = self.eval_model(s_).numpy().copy()
            greedy_next_action = np.argmax(q_next_eval, axis=2)  # get greedy action index of next eval
            greedy_next_action_mask = tf.one_hot(greedy_next_action,
                                                 self.subaction_num)  # mask to select only correpsond q value

            q_target_s_next = self.target_model(s_).numpy().copy()  # compute with target network estimate
            masked_q_target_s_next = tf.multiply(q_target_s_next,
                                                 greedy_next_action_mask)  # selected q value based on on-policy greedy action
            if self.td_operator_type == 'MEAN':
                operator = tf.reduce_sum(masked_q_target_s_next, 2)  # compute greedy
                operator = tf.multiply(operator, idle_mask)  # remove value of idle branches
                a = tf.reduce_sum(operator, axis=1, keepdims=True)
                b = tf.cast(1 / tf.reduce_sum(idle_mask, axis=1, keepdims=True), tf.float32)
                operator = tf.multiply(a, b)
                operator = tf.multiply(tf.expand_dims(operator, -1), eval_act_index_mask)
            elif self.td_operator_type == "MAX":
                # print("max operator")
                operator = tf.reduce_sum(masked_q_target_s_next, 2)
                operator = tf.expand_dims(tf.reduce_max(operator, 1), -1)
                operator = tf.multiply(tf.expand_dims(operator, -1), eval_act_index_mask)
                # operator=tf.reduce_max(operator)
            elif self.td_operator_type == "NAIVE":
                # print("naive")
                operator = tf.reduce_sum(masked_q_target_s_next, 2)
                operator = tf.multiply(tf.expand_dims(operator, -1), eval_act_index_mask)
            else:
                raise Exception("unknown")

            masked_r = tf.multiply(tf.expand_dims(r, -1), eval_act_index_mask)
            q_target = q_target + masked_r + self.gamma * operator  # fill the q_target value of eval act
            # only replace the value of action index in experience
            loss = tf.keras.losses.mean_squared_error(q_eval, tf.convert_to_tensor(q_target))
            self.loss_his.append(np.average(loss.numpy()))

            # print(np.average(loss.numpy()))
        gradients = tape.gradient(loss, self.eval_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.eval_model.trainable_variables))

    def store_transition(self, s, a, r, s_):
        index = self.memory_counter % self.memory_size  # store transition at position ,first coming first replaced.
        self.state_memory[index] = s
        self.action_memory[index] = a
        self.reward_memory[index] = r
        self.next_state_memory[index] = s_
        self.memory_counter += 1


    def save_model(self, model_folder):
        """
        save critic and actor model in model_folder
        critic model is saved in model_folder/critic.h5
        actor model is saved in model_folder/actor.h5
        model file type is .h5
        :param model_folder: relative path to model folder
        :return:
        """
        # save critic model
        self.eval_model.save(os.path.join(model_folder, "eval_model.h5"))


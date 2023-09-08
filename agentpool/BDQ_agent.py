import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers
# import matplotlib.pyplot as plt
import copy
from configs import agent_config
# tf.random.set_seed(0)

tf.config.experimental_run_functions_eagerly(True)
# np.random.se
# np.random.seed(0)


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

    model = tf.keras.Model(state_input, tf.convert_to_tensor(subaction_q_layers))
    return model

def build_network_double_branch(state_dim, action_dim, sub_action_num):
    print("build Braching DQ network")
    flattened_state_input = layers.Input(shape=(state_dim,))  # input shape (intersection_n*agent_o,1)
   # more than one intersection to control
    state_input_list = tf.split(flattened_state_input, [25 for _ in range(5)],axis=1)  # split input into multi branch
    state_action_embed_list = []
    for i in range(5):
        state_input=state_input_list[i]
        state_embeded=layers.Dense(64,activation="relu")(state_input)
        state_action_embed_list.append(state_embeded)
    concat=layers.Concatenate()(state_action_embed_list)
    # build shared representation hidden layer
    shared_representation = layers.Dense(512, activation="relu")(concat)
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

    model = tf.keras.Model(flattened_state_input, tf.convert_to_tensor(subaction_q_layers))
    return model

class BrachingDQ_agent:
    def __init__(self,
                 env_config,
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


        self.gamma =  AGENT_CONFIG["GAMMA"]
        self.learn_count = 0
        self.batch_size = AGENT_CONFIG["BATCH_SIZE"]

        # memory buffer
        self.memory_counter = 0
        self.memory_size = AGENT_CONFIG["MEMORY_SIZE"]
        self.state_memory = np.zeros((self.memory_size, self.state_dim))
        self.action_memory = np.zeros((self.memory_size, self.action_dim))
        self.reward_memory = np.zeros((self.memory_size, 1))
        self.next_state_memory = np.zeros((self.memory_size, self.state_dim))

        # target network update setting
        self.replace_count=0
        self.replace_target_iter = AGENT_CONFIG["REPLACE_INTERVAL"]

        self.replace_mode = AGENT_CONFIG["NET_REPLACE_TYPE"]
        if self.replace_mode == 'hard':
            self.tau = 1
        else:
            self.tau = AGENT_CONFIG["TAU"]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=AGENT_CONFIG["LEARNING_RATE"])
        # print()

    def choose_action(self, state,invalid_index):

        state = state[np.newaxis, :]
        action_branch_value = self.eval_model(state)

        joint_action = []
        if np.random.random()>self.epsilon:
            for i in range(self.action_dim):
                joint_action.append(np.argmax(action_branch_value[i]))
        else:
            for i in range(self.action_dim):
                joint_action.append(np.random.randint(0, self.subaction_num))

        return np.array(joint_action)

    def learn(self):
        # print("learn")
        available_count = min(self.memory_size, self.memory_counter)
        self.learn_count += 1
        if available_count>10000:
            self.replace_count += 1
            batch_indices = np.random.choice(available_count, self.batch_size)

            s = tf.convert_to_tensor(self.state_memory[batch_indices])
            a = tf.convert_to_tensor(self.action_memory[batch_indices])
            r = tf.convert_to_tensor(self.reward_memory[batch_indices])
            r = tf.cast(r, dtype=tf.float32)
            s_ = tf.convert_to_tensor(self.next_state_memory[batch_indices])

            self.update_gradient(s, a, r, s_)
            if self.replace_count % self.replace_target_iter == 0:
                print("replace para")
                self.replace_para(self.target_model.variables, self.eval_model.variables)

        # update epsilon
        fraction = min(float(self.learn_count) / self.decay_steps, 1)
        self.epsilon = self.max_epsilon + fraction * (self.min_epsilon - self.max_epsilon)
        if self.epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon

    @tf.function
    def replace_para(self, target_var, source_var):
        for (a, b) in zip(target_var, source_var):
            a.assign(a * (1 - self.tau) + b * self.tau)

    @tf.function
    def update_gradient(self, s, a, r, s_):
         with tf.GradientTape() as tape:
            q_eval = tf.transpose(self.eval_model(s),perm=[1,0,2])
            q_target=q_eval.numpy().copy()
            # make eval subaction q value to 0 for y
            eval_act_index = a.numpy().astype(int)  # action index of experience
            eval_act_index_mask=tf.one_hot(eval_act_index,self.subaction_num)
            eval_act_index_reverse_mask=tf.one_hot(eval_act_index,self.subaction_num,on_value=0.0,off_value=1.0) # we need reversed one hot matrix to reserve the value of not eval action index
            q_target=tf.multiply(q_target,eval_act_index_reverse_mask)# now the q value of eval_act is zero for us to insert y

            # on policy estimate next value
            q_next_eval = tf.transpose(self.eval_model(s_),perm=[1,0,2]).numpy().copy()
            greedy_next_action=np.argmax(q_next_eval,axis=2) # get greedy action index of next eval
            greedy_next_action_mask=tf.one_hot(greedy_next_action,self.subaction_num)
            if self.td_operator_type=='mean':
                q_target_s_next=tf.transpose(self.target_model(s_),perm=[1,0,2]).numpy().copy()# compute with target network estimate
                masked_q_target_s_next=tf.multiply(q_target_s_next,greedy_next_action_mask)
                q_target_s_next_branch_value=tf.reduce_sum(masked_q_target_s_next,2)
                q_target_s_next_mean=tf.expand_dims(tf.reduce_mean(q_target_s_next_branch_value,1),-1)
                masked_q_target_s_next_mean=tf.multiply(tf.expand_dims(q_target_s_next_mean,-1),eval_act_index_mask)
                masked_r=tf.multiply(tf.expand_dims(r,-1),eval_act_index_mask)
                q_target=q_target+masked_r+self.gamma*masked_q_target_s_next_mean


            # only replace the value of action index in experience
            loss = tf.keras.losses.mean_squared_error(q_eval, tf.convert_to_tensor(q_target))
            self.loss_his.append(np.average(loss.numpy()))

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



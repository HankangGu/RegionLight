import numpy as np
import time

def assign_state(state,itsx_assignment,itsx_state_dim):

    observations = []  # the collection of observations of all agents
    for assignment in itsx_assignment:
        if len(assignment) == 0:
            assignment = [assignment]
        agent_obs = np.zeros((len(assignment),itsx_state_dim))  # observation of current agent
        for id, itsx in enumerate(assignment):
            if itsx != 'dummy':
                # assign the state of one intersection to correct position of agent observation
                # leave the state of imaginary intersection as 0s
                # print(agent_obs)
                # print(state)
                agent_obs[id] = state[itsx]

        observations.append(np.hstack(agent_obs))  #
    return observations
def assign_reward(raw_reward,itsx_assignment):
    """
    build regional reward from raw reward of intersections for each agent according to the intersection assignments
    :param raw_reward: a dictionary {key,value} key(String): intersection id,
                                               value(int): reward of that intersection
    :return:
    """

    rewards = []
    for assignment in itsx_assignment:
        if len(assignment)==0:
            assignment=[assignment]
        agent_reward = 0  # reward of current agent
        for id, itsx in enumerate(assignment):
            if itsx != 'dummy':
                # sum the reward of one intersection
                # leave the reward of imaginary intersection as 0
                agent_reward += raw_reward[itsx]
        rewards.append(agent_reward)
    return rewards

def get_idle_branches(itsx_assignment):
    invalid_indices = []
    for assign in itsx_assignment:
        temp_list = []
        for id, name in enumerate(assign):
            if name == "dummy":
                temp_list.append(id)
        invalid_indices.append(temp_list)
    return invalid_indices

def convert_actions(actions_id,itsx_assignment):
    decoded_actions = {}
    for actions, itsxs in zip(actions_id, itsx_assignment):
        for a, itsx in zip(actions, itsxs):
            if itsx != "dummy":
                decoded_actions[itsx] = int(a) + 1
    return decoded_actions
def pipeline(env,agents,itsx_assignment,EXP_CONFIG,ENV_CONFIG):
    """
    Training pipeline that communicate agent with environment

    :param env:
    :param agents:
    :param itsx_assignment:
    :param EXP_CONFIG:
    :param ENV_CONFIG:
    :return:

    """
    def agent_learn():
        if EXP_CONFIG["TRAINING_PARADIM"] == "CLDE":
            agents[0].learn()
        else:
            for i in range(len(agents)):
                agents[i].learn()
    idle_branches_id=get_idle_branches(itsx_assignment)
    """----init global log-----"""
    global_step=0 # global training step
    episode_intersection_level_rewards=[]
    episode_throughput=[]
    episode_travel_time=[]
    """-------------"""
    for episode in range(EXP_CONFIG["EPISODE"]):
        state=env.reset()
        obs=assign_state(state,itsx_assignment,EXP_CONFIG["ITSX_STATE_DIM"])

        """---init episode log---"""
        episode_start_time=time.time()
        step_itsx_reward=[]
        episode_reward=[]
        """-------"""
        for step in range(int(ENV_CONFIG["SIM_TIMESPAN"]/ENV_CONFIG["ACTION_INTERVAL"])):
            actions_id=[]
            global_step += 1

            for aid,agent in enumerate(agents):
                action_id=agent.choose_action(obs[aid],idle_branches_id[aid])
                actions_id.append(action_id)
            joint_actions=convert_actions(actions_id,itsx_assignment)

            next_states, itsx_rewards, done, log_metric = env.step(joint_actions)

            next_obs = assign_state(next_states,itsx_assignment,EXP_CONFIG["ITSX_STATE_DIM"])
            rewards = assign_reward(itsx_rewards,itsx_assignment)

            if not done:
                # only store experience and learn when not finished
                for aid in range(len(agents)):
                    agents[aid].store_transition(obs[aid], actions_id[aid], rewards[aid], next_obs[aid])
                if global_step % EXP_CONFIG["LEARNING_INTERVAL"] == 0:
                    agent_learn()
            """----update step log---"""
            step_itsx_reward.append([value for _,value in itsx_rewards.items()])
            episode_reward.append(np.average(rewards))
            """-----------------"""
            obs=next_obs
        """----update episode log------"""
        episode_throughput.append(env.get_throughput())
        episode_travel_time.append(env.get_average_travel_time())
        episode_intersection_level_rewards.append(np.array(step_itsx_reward))
        """----------------------------"""
        log_msg={
            "episode":episode,
            "time cost":time.time()-episode_start_time,
            "episode reward":np.sum(episode_reward),
            "average travel time": env.get_average_travel_time(),
            "epsilon":agents[0].epsilon

        }
        print(log_msg)
    sim_log={'reward_log':episode_intersection_level_rewards,
             'throughput':episode_throughput,
             'travel_time':episode_travel_time
             }
    return sim_log
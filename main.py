import tensorflow as tf
import argparse
import numpy as np
import matplotlib.pyplot as plt
from cityflow_env_wrapper import CityflowEnvWrapper
import time
import os
import shutil
from configs import env_config, exp_config,region_config
import copy
from PipeLine import pipeline
# tf.random.set_seed(0)

tf.config.experimental_run_functions_eagerly(True)


# np.random.se
# np.random.seed(0)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netname", type=str, default="Hangzhou")
    parser.add_argument("--netshape", type=str, default="4_4")
    parser.add_argument("--flow", type=str, default="real")
    parser.add_argument("--agent", type=str, default="ABDQ")
    return parser.parse_args()



def init_exp(args):
    """
    Based on arguments and experiment configuration to initialise experiment
    Global working directory
    1.construct environment object

    2.agents object

    :param args:
    :return:
    """
    # retrieve arguments
    # netname = "Manhattan"
    # netshape = "16_3"
    # flow = "real"
    netname = args.netname
    netshape = args.netshape
    flow = args.flow
    agent_type = args.agent
    ENV_CONFIG = copy.deepcopy(env_config.ENV_CONFIG)
    EXP_CONFIG = copy.deepcopy(exp_config.EXP_CONFIG)
    # Construct Cityflow Configuration File and ENV DICT

    roadnet_file = "roadnet_" + netshape + ".json"
    flow_file = netname + "_" + netshape + "_" + flow + ".json"
    net_path = os.path.join(netname, roadnet_file)
    flow_path = os.path.join(netname, flow_file)
    experiment_name = "{0}_{1}_{2}".format(netname, netshape, flow)
    experiment_date = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
    ENV_CONFIG["PATH_TO_WORK_DIRECTORY"] = os.path.join("records", experiment_name + "_" + experiment_date)
    if not os.path.exists(ENV_CONFIG["PATH_TO_WORK_DIRECTORY"]):
        os.makedirs(ENV_CONFIG["PATH_TO_WORK_DIRECTORY"])
    ENV_CONFIG["ROADNET_PATH"] = net_path
    ENV_CONFIG["FLOW_PATH"] = flow_path
    env = CityflowEnvWrapper(ENV_CONFIG)



    # Construct Agent
    # we need state dim , action dim, subaction num
    itsx_state_dim=25 # queue lengh: 12, wave: 12, current phase: 1
    if ENV_CONFIG["ACTION_TYPE"]=="CHOOSE PHASE":
        itsx_action_dim=4
    elif ENV_CONFIG["ACTION_TYPE"]=="SWTICH":
        itsx_action_dim=2
    else:
        raise Exception("Unknow phase control")

    if EXP_CONFIG["REGIONAL"]:
        region_assignment_key=netshape+'_'+EXP_CONFIG["REGION_TYPE"]
        itsx_assignment= region_config.REGION_CONFIG[region_assignment_key]
    else:
        itsx_assignment=[[itsx_id] for itsx_id in env.intersection_ids]
    EXP_CONFIG["AGETN_NUM"]=len(itsx_assignment)
    agent_config={
        "ITSX_STATE_DIM":itsx_state_dim,
        "ACTION_DIM":len(itsx_assignment[0]),
        "ITSX_ACTION_DIM":itsx_action_dim,
    }
    EXP_CONFIG.update(agent_config)

    if EXP_CONFIG["TRAINING_PARADIM"]=="CLDE":

        agent=EXP_CONFIG["AGENT_CLASS_DICT"][agent_type](agent_config)
        agents=[agent for _ in range(EXP_CONFIG["AGETN_NUM"])]
    else:
        agents = [EXP_CONFIG["AGENT_CLASS_DICT"][agent_type](agent_config) for _ in range(EXP_CONFIG["AGETN_NUM"])]
    print(EXP_CONFIG)
    print(ENV_CONFIG)
    return env,agents,itsx_assignment,EXP_CONFIG,ENV_CONFIG

def run_pipeline(env,agents,itsx_assignment,EXP_CONFIG,ENV_CONFIG):
    logs=pipeline(env,agents,itsx_assignment,EXP_CONFIG,ENV_CONFIG)
    np.save(os.path.join(ENV_CONFIG['PATH_TO_WORK_DIRECTORY'],'episode_intersection_reward.npy'),logs['reward_log'])
    np.save(os.path.join(ENV_CONFIG['PATH_TO_WORK_DIRECTORY'],'episode_throughput.npy'), logs['throughput'])
    np.save(os.path.join(ENV_CONFIG['PATH_TO_WORK_DIRECTORY'],'episode_average_travel_time.npy'), logs['travel_time'])


if __name__ == "__main__":
    args = parse_args()
    env,agents,itsx_assignment,EXP_CONFIG,ENV_CONFIG=init_exp(args)
    run_pipeline(env,agents,itsx_assignment,EXP_CONFIG,ENV_CONFIG)
    # for name in dataset_list:
    # main(name)

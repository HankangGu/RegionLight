# Configuration Files
This folder stores configuration parameters
* ``exp_config.py`` Settings for whole experiment(Pipeline) e.g. training episode, training paradigm
    * `EPISODE`: the number of episode to train RL Agents
    * `TRAINING_PARADIM`: Whether agents share parameters and replay buffer. `CLDE`-> share. `DECENTRAL`-> not share
    * `AGENT_CLASS_DICT`: Reference to RL agent obj  
    * `REGIONAL`: whether RL agent control a region of intersection.
        * `REGION_TYPE`: If `REGIONAL==TRUE`, then this parameter further decide which region configuration to use
    * `LEARNING_INTERVAL`: how often RL agent sample and compute loss.     
    In `CLDE`, It is suggested to follow `learning_interval*agent_num<= batch size` to decrease the probability that some transitions are never sampled.
    Therefore, in `Manhattan`, learning interval is suggested to be smaller as the agent number is large.
     
    
* ``env_config.py`` Settings for Cityflow env e.g. running timespan, action type
    * `SIM_TIMESPAN`: seconds each simulation episode lasts
    * `ACTION_INTERVAL`: seconds the simulation steps after it receives actions from agents
    * `ACTION_TYPE`: how the environment interpret the action from agents. If `ACTION_TYPE==SWITCH`, the action of agent should be `0` or `1`
    * `INTERVAL`: 1 second each time environment `step()`. 
    * `SAVEREPLAY`: whether save the replay for each episode
    * `RLTRAFFICLIGHT`: whether to enable traffic light control through python API
    * `STATE` & `REWARD`: a description of state and reward. Modification on these two items have no effect on the actual state and reward during training
    
* ``agent_config.py`` Hyper-parameters for RL learner
    * `AGENT_CONFIG`: General hyper-parameter
        * `BATCH_SIZE`: The number of transitions for learning
        * `MEMORY_SIZE`: the size of replay buffer
        * `MAX_EPSILON`: initial value of epsilon for policy
        * `MIN_EPSILON`: minimum epsilon value 
        * `DECAY_STEPS`: the learning round it takes from MAX eps to MIN eps
        * `NET_REPLACE_TYPE`: 
            * `TAU`: If `NET_REPLACE_TYPE=SOFT`, `TAU` is the weight of new parameter
    * `BDQ_AGENT_CONFIG`:
        * `TD_OPERATEOR`: how temporal difference is computed
        * `LEARNING_RATE`:

* ``region_config.py`` Region Assignment for different scales of roadnets. The naming rule follows roadnet size+region type+ suffix. 
\
Suffix is necessary as one minimum dominant set is possible to generate different region assignment(described in paper)

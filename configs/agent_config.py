AGENT_CONFIG={
    "BATCH_SIZE":32,
    "MEMORY_SIZE":200000,
    "MAX_EPSILON":1,
    "MIN_EPSILON":0.001,
    "DECAY_STEPS":20000,
    "NET_REPLACE_TYPE":"SOFT",
    "TAU":0.001,
    "REPLACE_INTERVAL":200,
    "GAMMA":0.99,#0.9

}

BDQ_AGENT_CONFIG={
    "TD_OPERATOR":"MEAN",
    "LEARNING_RATE":0.0001
}
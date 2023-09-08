from agentpool.AdaptiveBDQ_agent import AdaptiveBDQ_agent
from agentpool.BDQ_agent import BrachingDQ_agent

EXP_CONFIG = {
    "EPISODE": 200,
    "TRAINING_PARADIM": "DECENTRAL",  # "CLDE",  # CLDE: centralised learning but decentralised execution
    "AGENT_CLASS_DICT": {
        "BDQ": BrachingDQ_agent,
        "ABDQ": AdaptiveBDQ_agent
    },
    "REGIONAL": True,
    "REGION_TYPE": "ADJACENCY1",
    "LEARNING_INTERVAL": 10,
}

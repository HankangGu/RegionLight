# agentpool
This folder stores RL agents

## Interface for ABDQ

* ``__init__``: initialize the agent
* ``choose_action(state,idle_id=None)``: evaluate state-action value of ``state`` and choose action according policy. Fill `idle_branch` with `-1`
* ``learn()``: sample from replay buffer and learn
* ``update_gradient(s, a, r, s_)``: tensorflow function, compute loss
* ``replace_para(target_var, source_var)``: update target network(`target_var`) based on source network(`source_var`)
* ``store_transition(s, a, r, s_)``: store transition in replay buffer
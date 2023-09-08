## RegionLight


## Python Requirement
The python version and related packages are listed
* `python=3.7.13`
* `cityflow=0.1`
* `tensorflow=2.4.1`
* `gurobi=10.0.1`
* `networkx=2.6.3`
* `numpy=1.21.5`


## Script
* ``main.py``: takes arguments to initialize agents, cityflow environment and configuration. And pass them to pipeline 
    * `--netname`: dataset name (folder name in `data`)
    * `--netshape`: the size of grid
    * `--flow"`: flow suffix
    * `--agent`: RL agent type `BDQ` or `ABDQ`
* ``PipeLine.py``: takes output of `main.py` and perform simulation-training process
* ``cityflow_env_wrapper.py``: a wrapper of [cityflow](https://cityflow.readthedocs.io/en/latest/introduction.html) package with high level functions such as collecting queue length of one intersection.
\
This script is modified based on [CoLight](https://github.com/wingsweihua/colight/blob/master/anon_env.py).

## Usage

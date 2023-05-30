# RL-MTL
## Model Code
+ layers: stores common network structures
  + critic: critic network
  + esmm: esmm(actor) network, can introduce other MTL models as actor inside slmodels
  + layers: classical Embedding layers and MLP layers
+ slmodels: SL baseline models
+ agents: RL models
+ train: training-related configuration
+ env.py: offline sampling simulation environment
+ RLmain.py: main RL training program
+ SLmain.py: SL training main program

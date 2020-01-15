import numpy as np
from env import Env
from models import PolicyNet, Critic
from utils import one_hot
import torch
from torch.optim import Adam
import time
import os
from datetime import datetime
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#------------------------SET PARAMETERS----------------------------

SEED             = 17
BATCH_SIZE       = 128
N_NODES          = 11
N_DEPOT          = 1
NUM_LAYERS       = 1
CAPACITY         = [20,15,10]
MAX_DEMAND       = 10
N_VEHICLES       = len(CAPACITY)
DIM_STATIC        = 2
DIM_DYNAMIC       = 1 + N_VEHICLES
DIM_LOAD          = N_VEHICLES
DIM_EMBED         = 128
MAX_EP_lEN        = 16
GAMMA             = 0.99
ENTROPY_REG       = 0.01
MAX_GRAD_NORM     = 2
DROPOUT           = 0.1
EMBED_TYPE        = 'conv1d'
LOG_INTERVAL      = 200

#----------------INITIALIZE ENVIROMENT AND POLICIES----------------

env_test = Env(seed = SEED, batch_size = BATCH_SIZE, capacity = CAPACITY,
        n_nodes = N_NODES, n_depot = N_DEPOT, max_demand = MAX_DEMAND, n_agents = N_VEHICLES)

policy = [PolicyNet(batch_size = BATCH_SIZE, n_nodes = N_NODES, n_agents=N_VEHICLES, num_layers = NUM_LAYERS,
                        dim_s = DIM_STATIC, dim_d = DIM_DYNAMIC,
                        dim_embed = DIM_EMBED, n_glimpses = 0,  embeding_type=EMBED_TYPE,
                        dropout = DROPOUT).to(device) for i in range(N_VEHICLES)]

#------------------LOAD TRAINDEL MODEL---------------------------
model_dir = 'weights/model_exp_1.pt'
policy_name = "policy_agent_X"

if os.path.isfile(model_dir):
    checkpoint     = torch.load(model_dir,map_location=device)
else:
    raise ValueError('No model file!')

for agent_id in range(N_VEHICLES):
    p_name = policy_name.replace("X",str(agent_id))
    policy[agent_id].load_state_dict(checkpoint[p_name])


#-----------------RUN TRAINED POLICY----------------
num_epochs = math.ceil(1000/BATCH_SIZE)

total_tests = []
total_times = []
for i in range(num_epochs):
    start = time.time()
    o_t, d_t, r_t = env_test.reset(), False, 0

    actions_ep   = []
    log_probs_ep = []
    rewards_ep   = []
    values_ep    = []
    last_hh_t         = [None]*N_VEHICLES
    for  t in range(int(MAX_EP_lEN) ):
        actions         = []
        actions_one_hot = []
        log_probs       = []
        values          = []
        for agent_id in range(N_VEHICLES) :
            model = policy[agent_id].eval()
            logits, prob , log_p, last_hh_t[agent_id] = model(o_t, last_hh_t[agent_id], agent_id)

            #--------- GREEDY POLICY ------------
            act = torch.argmax(prob, dim =1)  # [ batch size ]
            actions.append(act.detach())

            ot_2, d_t, r_t = env_test.step(act.detach().unsqueeze(1), agent_id)
            o_t = ot_2
            values.append( r_t )

        r_step = torch.stack(values, dim = 1)      #[batch_size, n_agents]
        a      = torch.stack(actions,   dim = 1)   #[batch_size, n_agents]
        actions_ep.append(a)
        rewards_ep.append(r_step)
    end = time.time()
    rewards  = torch.stack(rewards_ep,  dim = 2 ).sum(dim=2).sum(dim=1)  #[batch_size, n_agents, ep_len]
    total_tests.append(rewards)
    total_times.append((end-start)/BATCH_SIZE)

#------------------- SAVE RESULTS -----------------------
rewards_total = torch.stack(total_tests, dim=1).reshape(-1,)
np_results    = rewards_total.numpy()
np.save('vrp_results_RL',np_results)
np_runtimes   = np.array(total_times).reshape(-1,)
np.save('vrp_runtimes_RL',np_runtimes)

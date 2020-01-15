import os
import time
import numpy as np
from env import Env
from utils import one_hot
from models import PolicyNet, Critic
import torch
import torch.nn as nn
from torch.optim import Adam
from datetime import datetime
from tensorboardX import SummaryWriter



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

#------------------------SET LOGS WRITER--------------------------

time_id = datetime.now().strftime("%d_%m_%Y")
filename = "experiment1"
log_dir = os.path.join('tensorboardLogs', filename)
writer = SummaryWriter( log_dir = log_dir)

#----------------INITIALIZE ENVIROMENT AND POLICIES----------------

env = Env(seed = SEED, batch_size = BATCH_SIZE, capacity = CAPACITY,
          n_nodes = N_NODES, n_depot = N_DEPOT, max_demand = MAX_DEMAND, n_agents = N_VEHICLES)

env_test = Env(seed = SEED+2, batch_size = BATCH_SIZE, capacity = CAPACITY,
        n_nodes = N_NODES, n_depot = N_DEPOT, max_demand = MAX_DEMAND, n_agents = N_VEHICLES)

policy = [PolicyNet(batch_size = BATCH_SIZE, n_nodes = N_NODES, n_agents=N_VEHICLES, num_layers = NUM_LAYERS, 
                        dim_s = DIM_STATIC, dim_d = DIM_DYNAMIC,
                        dim_embed = DIM_EMBED, n_glimpses = 0,  embeding_type=EMBED_TYPE, 
                        dropout = DROPOUT).to(device) for i in range(N_VEHICLES)]

value_func = Critic( batch_size = BATCH_SIZE, n_nodes = N_NODES, dim_s = DIM_STATIC,
                        dim_embed = DIM_EMBED, embeding_type = EMBED_TYPE).to(device)

actor_optimizers  = [Adam(i.parameters(), lr=0.0005) for i in policy]
critic_optimizers = Adam(value_func.parameters(), lr=0.0005)

#------------------LOAD TRAINING CHECKPOINT---------------------------


model_dir = 'weights/model_exp_1.pt'
save_dir  = 'weights/model_tmp.pt'
policy_name = "policy_agent_X"
value_name  = "value_func"
actr_op_name = 'actor_opt_agent_X'
crtc_op_name = 'critic_opt'
if os.path.isfile(model_dir):
    print("Loaded params!")
    checkpoint     = torch.load(model_dir,map_location=device)
    trainend_steps = checkpoint['steps']
    value_func.load_state_dict(checkpoint[value_name])
    critic_optimizers.load_state_dict(checkpoint[crtc_op_name])
    for agent_id in range(N_VEHICLES):
        p_name = policy_name.replace("X",str(agent_id))
        a_opt  = actr_op_name.replace("X", str(agent_id))

        policy[agent_id].load_state_dict(checkpoint[p_name])
        actor_optimizers[agent_id].load_state_dict(checkpoint[a_opt])

else:
    trainend_steps = 0
max_steps = 260000
total_steps = max_steps - trainend_steps


#INFO STATE:
#----- o: (position, nodes_locations, demand, load, mask) --- observation
          #env.position:  [batch_size, n_agents_2] --- vehicle position
          #env.input_pnt: [batch_size, n_nodes, 2] --- coordinates nodes
          #env.demand:    [batch_size, n_nodes]    --- demand of each node
          #env.load:      [batch_size, n_agents]   --- load of each agent
          #env.mask:      [batch_size, n_nodes]    --- load of each agent

#GET FIRST STATE
o, d, r = env.reset(), False, 0

print("Training.....")
start_time = time.time()

for epoch in range(total_steps):
    actions_ep   = []
    log_probs_ep = []
    rewards_ep   = []
    values_ep    = []
    last_hh         = [None]*N_VEHICLES 
    for  t in range(int(MAX_EP_lEN) ):
        actions         = []
        actions_one_hot = []
        log_probs       = []
        values          = []
        rewards         = []
        for agent_id in range(N_VEHICLES) :
            model = policy[agent_id].train()
            logits, prob , log_p, last_hh[agent_id] = model(o, last_hh[agent_id], agent_id)

            #--------- STOCHASTIC POLICY ----------
            prob_dist  = torch.distributions.categorical.Categorical( probs = prob)
            act        = prob_dist.sample()  # [ batch size ]

            o2, d, r = env.step(act.detach().unsqueeze(1), agent_id)
            o = o2
            rewards.append( r )
            actions.append(act.detach())
            actions_one_hot.append(one_hot(act.detach(),N_NODES))
            log_probs.append(log_p)

        r_step = torch.stack(rewards, dim = 1)            #[batch_size, n_agents]
        a      = torch.stack(actions,   dim = 1)          #[batch_size, n_agents]
        a_oh   = torch.stack(actions_one_hot , dim =1)    #[batch_size, n_agents, n_nodes]
        lp     = torch.stack(log_probs, dim = 1)          #[batch_size, n_agents, n_nodes]

        actions_ep.append(a_oh)
        log_probs_ep.append(lp)
        rewards_ep.append(r_step)

    values = value_func(o)
    
    actions   = torch.stack(actions_ep,  dim=2).to(device, dtype=torch.float) #[batch_size, n_agents, ep_len, n_nodes]
    log_probs = torch.stack(log_probs_ep,dim=2 )                              #[batch_size, n_agents, ep_len, n_nodes]
    rewards   = torch.stack(rewards_ep,  dim = 2 )                            #[batch_size, n_agents, ep_len]

    if epoch % LOG_INTERVAL == 0 :
        end_time = time.time() - start_time
        total_rewards_agent = torch.sum(rewards,dim=2)
        total_rewards_ep    = torch.sum(total_rewards_agent, dim=1)
        print("--------------- Step: {}, Time: {} -----------------".format( epoch + trainend_steps,  time.strftime("%H:%M:%S", time.gmtime(end_time) ) ) )
        print( "Mean_train_reward: "+str(torch.mean(total_rewards_ep).detach().cpu().numpy()) )
        writer.add_scalar("Mean_train_reward", torch.mean(total_rewards_ep).detach().cpu().numpy(), epoch + trainend_steps)

    for agent_id in range(N_VEHICLES):

        adv = torch.sum(rewards,(2,1)) - values.detach()

        action_log_probs = torch.sum( actions[:,agent_id,:,:]*log_probs[:,agent_id,:,:], dim=2)
        sum_log_probs    = torch.sum(action_log_probs, dim=1)

        #-----------------ACTOR UPDATE---------------------
        actor_loss  = torch.mean( sum_log_probs*adv ).view(-1,)

        actor_optimizers[agent_id].zero_grad()
        actor_loss.backward(retain_graph=True)

        if MAX_GRAD_NORM is not None:
            nn.utils.clip_grad_norm_(policy[agent_id].parameters(), MAX_GRAD_NORM)
        actor_optimizers[agent_id].step()
        #--------------------------------------------------

    #-----------------CRITCI UPDATE---------------------
    critic_loss = nn.MSELoss()(values, torch.sum(rewards,(2,1)))
    critic_optimizers.zero_grad()
    critic_loss.backward(retain_graph=True)

    if MAX_GRAD_NORM is not None:
        nn.utils.clip_grad_norm_(value_func.parameters(), MAX_GRAD_NORM)
    critic_optimizers.step()               
    
    #--------------------------------------------------

    if epoch % LOG_INTERVAL == 0:

        writer.add_scalar("Actor_loss:{}".format(agent_id) , actor_loss.detach().cpu().numpy(), epoch +trainend_steps)
        writer.add_scalar("Critic_loss:{}".format(agent_id) , critic_loss.detach().cpu().numpy(), epoch + trainend_steps )


    o, d, r, ep_ret, ep_len = env.reset(), False, 0, 0 , 0

#-----------------SAVE CHECKPOINT----------------
    if epoch % LOG_INTERVAL == 0:
        save_dict = {}
        save_dict['steps']      = epoch + trainend_steps
        save_dict[value_name]   = value_func.state_dict()
        save_dict[crtc_op_name] = critic_optimizers.state_dict()
        for agent_id in range(N_VEHICLES):
            p_name = policy_name.replace("X",str(agent_id))
            a_opt  = actr_op_name.replace('X',str(agent_id))
            save_dict[p_name] = policy[agent_id].state_dict()
            save_dict[a_opt]  = actor_optimizers[agent_id].state_dict()
        if os.path.isfile(save_dir):
            torch.save(save_dict, save_dir)
        else:
            os.mkdir('weights')
            torch.save(save_dict, save_dir)
#-----------------TEST TRAINED POLICY----------------
    if epoch % (LOG_INTERVAL*2) == 0:
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

            r_step = torch.stack(values, dim = 1)       #[batch_size, n_agents]
            a      = torch.stack(actions,   dim = 1)    #[batch_size, n_agents]
            actions_ep.append(a)
            rewards_ep.append(r_step)

        rewards   = torch.stack(rewards_ep,  dim = 2 ).sum(dim=2).sum(dim=1)  #[batch_size, n_agents, ep_len]
        actions   = torch.stack(actions_ep,  dim = 2 )                        #[batch_size, n_agents, ep_len
        mean_test_reward = torch.mean(rewards).cpu().numpy()

        print("              -------  TESTING:  -------")
        print("Mean_test_reward: "+str(mean_test_reward) )
        print("Unsatisfied demand: ",torch.sum(env_test.demand).item())
        print('*depot is node:',str(N_NODES-1))
        agent_name = "Actions vehicle: N"
        for agent_num in range(N_VEHICLES):
            name_a = agent_name.replace('N',str(agent_num))
            print(name_a,actions[0,agent_num,:].cpu().numpy())
        print("--------------------------------------------------------")
        writer.add_scalar("Mean_test_reward" , mean_test_reward, epoch  + trainend_steps)
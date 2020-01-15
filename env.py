import numpy as np
import torch
from utils import reward_func

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Env(object):
    '''
    Generate a batch of environments of nodes with randomized demands and locations in the [0,1]^2 space
    '''

    def __init__(self,  seed, batch_size = 1, capacity = 20, n_nodes =10, n_depot = 1,
                 n_agents= 3, max_demand = 10, max_decode_len = 20):

        self.batch_size     = batch_size
        self.capacity       = capacity
        self.n_nodes        = n_nodes
        self.n_cust         = n_nodes - n_depot
        self.n_agents       = n_agents
        self.max_demand     = max_demand
        self.rnd            = np.random.RandomState(seed= seed)
        self.positions      = np.zeros((batch_size,n_agents,1))
        self.max_decode_len = max_decode_len


    def reset(self):

        self.n_steps  = 0
        self.done     = 0

        input_pnt = self.rnd.uniform(0,1,
            size=(self.batch_size, self.n_nodes, 2))

        demand       = self.rnd.randint(1,self.max_demand,[self.batch_size, self.n_nodes])
        demand[:,-1] = 0 # demand of depot

        #[coord1, coord2, demand]
        self.input_data = \
        torch.from_numpy(np.concatenate([input_pnt,np.expand_dims(demand,2)],2)).to(device, torch.float)

        self.input_pnt  = torch.from_numpy(input_pnt).to(device, torch.float)
        self.demand     = torch.from_numpy(demand).to(device, torch.float) #[batch_size , n_nodes]

        # load: [batch_size, n_agents]
        self.load = torch.tensor([self.capacity]).repeat(self.batch_size,1).to(device, torch.float)

        # create mask
        self.mask = torch.zeros((self.batch_size, self.n_nodes)).to(device, torch.float)

        # update mask -- mask if customer demand is 0 and depot
        #[batch_size * beam_width , n_nodes]
        self.mask = torch.cat( ( self.demand.eq(0)[:,:-1].type(torch.float),
                                torch.ones(self.batch_size,1).to(device) ), dim = 1)

        #----------mask if demand > load of first agent
        demand_gt  = self.demand.gt(self.load[:,0].unsqueeze(1).repeat(1,self.n_nodes))[:,:-1].float()
        depot_demand = torch.zeros((self.batch_size,1)).to(device,torch.float)
        self.mask += torch.cat((demand_gt, depot_demand),dim=1)

        self.positions  = torch.ones((self.batch_size,self.n_agents)).to(device)*(self.n_nodes-1) #Initial position is last node which is depot

        self.position_coord      = torch.ones((self.batch_size, 1, 2)).to(device) *self.input_pnt[:,-1,:].unsqueeze(1)
        self.position_coord      = self.position_coord.repeat(1,self.n_agents,1)

        return (self.position_coord, self.input_pnt, self.demand, self.load, self.mask)

    def step(self, action, agent):
        '''
        Make action step of an agent
        
        action: [batch_size, 1]
        agent : int
        '''
        self.n_steps += 1

        previous_pos_temp = self.positions[:, agent]
        batched_pos_v = previous_pos_temp.unsqueeze(1).repeat(1,self.n_nodes).unsqueeze(2).repeat(1,1, 2).to(device, torch.long)
        batched_action_v = action.repeat(1,self.n_nodes).unsqueeze(2).repeat(1,1, 2).to(device, torch.long)

        #input_pnt: [batch_size, n_nodes, 2]
        #batched_pos_v : [batch_size,n_nodes,2]
        gather_s_1    = torch.gather(self.input_pnt,
                                        dim=1, index = batched_pos_v) #Initial position coordinates
        gather_s_2    = torch.gather(self.input_pnt,
                                        dim=1, index = batched_action_v) #Next position coordinates
        reward = reward_func([gather_s_1[:,0,:], gather_s_2[:,0,:]]) #Distance between positions
        self.positions[ :, agent ] = action.squeeze(1) #set vehicle new position

        # self.position_coord[:, agent,:]     = gather_s_2[:,0,:]

        ixs = torch.arange(self.n_agents, dtype=torch.int64).to(device).unsqueeze(1)
        selected_pos   = torch.where(ixs[None,:] == agent, gather_s_2[:,0,:].unsqueeze(1).repeat(1,self.n_agents,1), torch.tensor(0.,device=device))
        deselected_pos = torch.where(ixs[None,:] == agent, torch.tensor(0.,device=device), self.position_coord)
        self.position_coord = selected_pos + deselected_pos

        #--------------------- UPDATE DEMAND ----------------------
        # how much the demand is satisfied

        selected_demand = torch.gather(self.demand.float(), dim = 1 , index = action.repeat(1,self.n_nodes).to(device,torch.long))
        agent_id_v  = torch.from_numpy( np.array([agent] )  ).unsqueeze(1).repeat(self.batch_size,self.n_agents).to(device,torch.long)

        selected_load   = torch.gather(self.load.float(), dim = 1 , index = agent_id_v) 

        d_sat = torch.min(selected_demand[:,0], selected_load[:,0] ).to(device) #[batch_size] Demand satiisfied

        d_scatter = torch.zeros((self.demand.shape[0],
                    self.demand.shape[1])).to(device, torch.float).scatter_(dim = 1, index = action, src = d_sat.unsqueeze(1))

        self.demand = self.demand - d_scatter  #[batch_size, n_nodes]


        #---------------------- UPDATE LOAD ----------------------
        #refill the truck if depot=10 -- action: [10,9,10] -> load_flag: [1 0 1]

        load_flag = action.squeeze(1).eq(self.n_cust).type(torch.float) #[batch_size]
        self.load[:, agent] -= d_sat
        self.load[:,agent]  = self.load[:,agent]*(1-load_flag) + load_flag *self.capacity[agent] #[batch_size]


        #--------------------- UPDATE MASK  -------------------------
        
        #----------- mask customers with zero demand
        #[batch_size * beam_width, n_nodes]

        zero_demand  = self.demand.eq(0)[:,:-1].float()
        depot_demand = torch.zeros((self.batch_size,1)).to(device,torch.float)

        self.mask    = torch.cat(( zero_demand, depot_demand ), dim=1).to(device) #[batch_size, n_nodes]

        #---------- mask if load = 0 **FOR NEXT AGENT**
        # mask if in depot and there is still a demand 

        if agent+1 >= self.n_agents:
            next_agent = 0
        else:
            next_agent = agent+1
        zero_load         = self.load[:,next_agent].unsqueeze(1).eq(0).type(torch.float).repeat(1, self.n_cust) #[batch_size, n_cust]
        batch_with_demand = torch.sum(self.demand,1).gt(0).type(torch.float).unsqueeze(1) #[batch_size, 1]
        if_in_depot       = action.eq(self.n_cust).type(torch.float) #[batch_size, 1]

        self.mask += \
        torch.cat( ( zero_load, batch_with_demand*if_in_depot ), dim=1 )

        #----------mask if demand > load **FOR NEXT AGENT**
        demand_gt  = self.demand.gt(self.load[:,next_agent].unsqueeze(1).repeat(1,self.n_nodes))[:,:-1].float()

        self.mask += torch.cat((demand_gt, depot_demand),dim=1)

        if self.n_steps == self.max_decode_len:
            self.done = 1

        return (self.position_coord, self.input_pnt, self.demand, self.load, self.mask), self.done, reward
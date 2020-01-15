import torch
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def reward_func(sample_solution):
    '''
    Distance between two nodes

    sample_solution: list of two batches of dim [batch_size x input_dim]
    '''

    sample_solution = torch.stack(sample_solution,dim=0) # [2 x batch_size x input_dim]

    dist = sample_solution[-1,:,:].unsqueeze(0) - sample_solution[:-1,:,:]

    route_lens_decoded = torch.pow(torch.sum(torch.pow(dist ,2),dim=2),.5).squeeze(0)

    return route_lens_decoded

def one_hot(acts, act_dim):

    '''
    One-hot encoding of actions

    acts: actions to be one-hot encoded
    act_dim: aka number of nodes
    '''

    return torch.zeros((acts.shape[0], act_dim)).to(device).scatter_( dim=1, index=acts.unsqueeze(1), src=torch.ones((acts.shape[0],1)).to(device) )
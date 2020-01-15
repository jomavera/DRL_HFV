import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):


    def __init__(self, dim, dim_embed):
        super(Encoder, self).__init__()

        self.embed = nn.Conv1d(dim, dim_embed, 1)

        return

    def forward(self, input):

        input_2 = input.permute(0,2,1)
        out = self.embed(input_2)

        return out.permute(0,2,1)


class Decoder(nn.Module):


    def __init__(self, dim_embed, num_layers, dropout=0.1):

        super(Decoder, self).__init__()

        self.LSTM_layer = nn.LSTM(dim_embed, dim_embed, num_layers, dropout=dropout)

        return

    def forward(self, input_, h, c):

        input_ = input_.unsqueeze(0)
        output, (h_i, c_i) = self.LSTM_layer(input_, (h, c))

        return output.squeeze(0), (h_i, c_i)      

class Attention(nn.Module):

    def __init__(self, dim_embed, embeding_type = 'conv1d', tanh_exp=0):
        super(Attention, self).__init__()

        self.dim_embed = dim_embed

        if embeding_type == 'conv1d':
            self.proj = Encoder(dim_embed , dim_embed)
            self.w_a  = Encoder(dim_embed*3 , dim_embed)
            self.v_a  = nn.Parameter(torch.randn(dim_embed))#[dim embed]

        else:
            self.proj = nn.Linear(dim_embed , dim_embed)
            self.w_a  = nn.Linear(dim_embed*3, dim_embed)
            self.v_a  = nn.Parameter(torch.randn(dim_embed))
    
        self.tanh_exp = tanh_exp

        return


    def forward(self, encoded_static, encoded_dynamic, decoder_output):

        n_nodes = encoded_static.shape[1]

        x_t = torch.cat( (encoded_static, encoded_dynamic),dim=2)                #[batch_size, n_nodes, dim_embed*2]
        proj_dec  = self.proj(decoder_output.unsqueeze(1)).repeat(1, n_nodes,1)  #[batch_size, n_nodes, dim_embed]
        hidden    = torch.cat( (x_t, proj_dec),dim=2)                            #[batch_size, n_nodes, dim_embed*3]

        u_t = torch.matmul(self.v_a, torch.tanh(self.w_a(hidden) ).permute(0,2,1) ) #[batch-size, n_nodes]

        if self.tanh_exp > 0:

                logits = self.tanh_exp*torch.tanh(u_t)
            
        else:

            logits = u_t

        return logits

class PolicyNet(nn.Module):

    def __init__(self, batch_size, n_nodes, n_agents, num_layers, dim_s, dim_d,
                 dim_embed, n_glimpses, embeding_type = 'conv1d', dropout = 0):

        super(PolicyNet, self).__init__()

        self.batch_size   = batch_size
        self.n_agents     = n_agents
        self.dim_embed    = dim_embed
        self.num_layers   = num_layers

        self.n_nodes    = n_nodes
        if embeding_type == 'conv1d':

            self.enc_s      = Encoder(dim_s, dim_embed)
            self.enc_pos    = Encoder(dim_s, dim_embed)
            self.enc_d      = Encoder(dim_d, dim_embed)
            self.project_s  = Encoder(dim_embed, dim_embed)
            self.project_d  = Encoder(dim_embed*n_agents + dim_embed, dim_embed)

        else:

            self.enc_s      = nn.Linear(dim_s, dim_embed)
            self.enc_pos    = nn.Linear(dim_s, dim_embed)
            self.enc_d      = nn.Linear(dim_d, dim_embed)
            self.project_s  = nn.Linear(dim_embed, dim_embed)
            self.project_d  = nn.Linear(dim_embed*n_agents + dim_embed, dim_embed)

        self.dec        = Decoder(dim_embed, num_layers = num_layers, dropout = dropout)
        self.att        = Attention(dim_embed, embeding_type, tanh_exp = 0)
        self.h          = nn.Parameter(torch.randn( (1,batch_size, dim_embed))) #[batch_size, hidden_dim]
        self.c          = nn.Parameter(torch.randn( (1,batch_size, dim_embed))) #[batch_size, hidden_dim]
        self.max_len    = n_nodes * 2
        self.dim_s      = dim_s

        self.n_glimpses = n_glimpses
        self.glimpses   = []
        for _ in range(self.n_glimpses):
            self.glimpses.append(Attention(dim_embed))

        self.drop_rnn = nn.Dropout(p = dropout)
        self.drop_h   = nn.Dropout(p=dropout)
        self.drop_c   = nn.Dropout(p=dropout)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        return

    def forward(self, o, last_hh=None, agent_id=0):

        positions = o[0].to(device)
        static    = o[1].to(device)
        dynamic   = torch.cat( (o[2].unsqueeze(2), o[3].unsqueeze(1).repeat(1,self.n_nodes,1)) , dim=2 ).to(device)
        mask      = o[4].to(device)

        encoded_s      = self.enc_s( static )

        encoded_pos    = self.enc_pos( positions)                    #[batch_size, n_agents, dim_embed]
        encoded_d      = self.enc_d( dynamic )                     #[batch_size, n_nodes , dim_embed]
        #[batch_size, n_nodes, dim_embed*n_agents + dim_embed ]
        augmented_d    = torch.cat( (encoded_pos.unsqueeze(1).reshape(self.batch_size,1,-1).repeat(1,self.n_nodes,1), encoded_d) ,dim=2 )

        proj_s         = self.project_s( encoded_s )
        proj_d         = self.project_d( augmented_d )

        if last_hh == None:
            last_h = self.h
            last_c = self.c
        else:
            last_h = last_hh[0]
            last_c = last_hh[1]

        encoded_input = encoded_pos[: , agent_id, :]
        dec_output, decoder_state = self.dec(encoded_input, last_h, last_c) 

        dec_output = self.drop_rnn(dec_output)

        if self.num_layers == 1:
            h = self.drop_h(decoder_state[0])
            c = self.drop_c(decoder_state[1])
        
        else:
            h = decoder_state[0]
            c = decoder_state[1]

        h_i = h.squeeze(0) #[batch_size, dim_embed]

        for i in range(self.n_glimpses):

            logits   = self.glimpses[i](proj_s, proj_d, h_i)
            logits  -= 100000*mask
            prob     = torch.softmax(logits, dim=1)
            h_i      = torch.bmm(prob.unsqueeze(1), proj_s).squeeze(1)

        logits   = self.att(proj_s, proj_d, h_i)
        logits  -= 100000* mask
        logprob  = F.log_softmax(logits, dim=1)
        prob     = torch.exp(logprob)

        return logits, prob, logprob, (h,c)

class Critic(nn.Module):

    def __init__(self, batch_size, n_nodes, dim_s,
                 dim_embed, embeding_type = 'conv1d'):
        super(Critic, self).__init__()

        self.dim_embed = dim_embed
        if embeding_type == 'conv1d':

            self.project_s = Encoder(dim_s, dim_embed)
            self.w_a       = Encoder(dim_embed  , dim_embed)
            self.w_c       = Encoder( (dim_embed*2), dim_embed)

            self.v_a = nn.Parameter(torch.randn(dim_embed)) #[dim embed]
            self.v_c = nn.Parameter(torch.randn(dim_embed))

        else:

            self.project_s = nn.Linear(dim_s, dim_embed)
            self.w_a       = nn.Linear(dim_embed  , dim_embed)
            self.w_c       = nn.Linear( (dim_embed*2), dim_embed)

            self.v_a = nn.Parameter(torch.randn(dim_embed)) #[dim embed]
            self.v_c = nn.Parameter(torch.randn(dim_embed))

        self.linear_1  = nn.Linear(dim_embed, dim_embed)
        self.linear_2  = nn.Linear(dim_embed, 1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        return


    def forward(self, o ):

        instance = o[1]

        projected_instance = self.project_s(instance) #[batch_size, n_nodes, dim_embed]
        u_t = torch.matmul(self.v_a, torch.tanh(self.w_a(projected_instance) ).permute(0,2,1) )

        a_t = F.softmax (u_t, dim=1).unsqueeze(2).repeat(1,1, self.dim_embed)     #[batch_size, n_nodes, dim_embed]

        c_t = a_t*projected_instance          #[batch_size, n_nodes, dim_embed]
        hidden_2 = torch.cat((projected_instance,c_t),dim=2) #[batch-size, n_nodes, dim_embed*2]

        u_t_2 = torch.matmul(self.v_c, torch.tanh(self.w_c(hidden_2) ).permute(0,2,1) ) #[batch_size, n_nodes]

        prob   = torch.softmax(u_t_2, dim=1)
        h_i    = torch.bmm(prob.unsqueeze(1), projected_instance).squeeze(1)   

        output_1    = F.relu( self.linear_1(h_i) )
        v           = self.linear_2(output_1).squeeze(1)

        return v
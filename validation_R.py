import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import brain_workcenter_R as brain

'''
load trained parameters in experiment
'''

class DRL_routing(brain.routing_brain):
    def __init__(self, env, job_creator, wc_list, **kwargs):
        # initialize the environment and the workcenter to be controlled
        self.env = env
        self.job_creator = job_creator
        self.wc_list = wc_list
        for wc in self.wc_list:
            wc.job_routing = self.action_by_DRL
        # retrive the data of chosen workcenter
        self.m_per_wc = len(self.wc_list[0].m_list)
        all_m_no = len(self.wc_list[0].m_list)*len(wc_list)
        # state space, eah machine generate 4 types of data, along with the processing time of job
        self.input_size = self.m_per_wc*3 + 3
        # action space, consists of all selectable machines
        self.output_size = self.m_per_wc
        # specify the path to store the model
        self.path = sys.path[0]
        # specify the ANN and state function
        if 'validated' in kwargs and kwargs['validated']:
            if self.m_per_wc == 2:
                self.address_seed = "{}\\routing_models\\validated_2machine_small.pt"
                self.action_NN = brain.build_network_small(self.input_size, self.output_size)
            if self.m_per_wc == 3:
                self.address_seed = "{}\\routing_models\\validated_3machine_medium.pt"
                self.action_NN = brain.build_network_medium(self.input_size, self.output_size)
            if self.m_per_wc == 4:
                self.address_seed = "{}\\routing_models\\validated_4machine_large.pt"
                self.action_NN = brain.build_network_large(self.input_size, self.output_size)
            self.build_state = self.state_deeper
            self.action_NN.load_state_dict(torch.load(self.address_seed.format(sys.path[0])))
            self.action_NN.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            print("---> VALIDATION mode ON <---")
        elif 'TEST' in kwargs and kwargs['TEST']:
            self.address_seed = "{}\\routing_models\\TEST_state_dict.pt"
            self.action_NN = brain.build_network_TEST(self.input_size, self.output_size)
            self.build_state = self.state_deeper
            self.action_NN.load_state_dict(torch.load(self.address_seed.format(sys.path[0])))
            self.action_NN.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            print("---> TEST mode ON <---")

    def action_by_DRL(self, job_idx, routing_data, job_pt, job_slack, wc_idx, *args):
        s_t = self.build_state(routing_data, job_pt, job_slack, wc_idx)
        # input state to policy network, produce the state-action value
        value = self.action_NN.forward(s_t.reshape(1,1,self.input_size),wc_idx)
        # generate the action
        a_t = torch.argmax(value)
        #print(value,a_t)
        #print('Policy NN choose action')
        return a_t

    def check_parameter(self):
        print('------------------ Sequencing Brain Parameter Check ------------------')
        print("Collect from:",self.address_seed)
        print('State function:',self.build_state.__name__)
        print('ANN architecture:',self.action_NN.__class__.__name__)
        print('*** SCENARIO:')
        print("Configuration: {} work centers, {} machines".format(len(self.job_creator.wc_list),len(self.job_creator.m_list)))
        print("PT heterogeneity:",self.job_creator.pt_range)
        print('Due date tightness:',self.job_creator.tightness)
        print('Utilization rate:',self.job_creator.E_utliz)
        print('----------------------------------------------------------------------')

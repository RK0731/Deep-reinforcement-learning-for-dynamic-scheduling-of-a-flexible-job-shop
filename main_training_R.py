import simpy
import sys
sys.path 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import numpy as np
from tabulate import tabulate

import agent_machine
import agent_workcenter
import brain_workcenter_R
import job_creation
import breakdown_creation
import heterogeneity_creation
import validation_S # for co-training

"""
THIS IS THE MODULE FOR ROUTING AGENT TRAINING
"""

class shopfloor:
    def __init__(self, env, span, m_no, wc_no, **kwargs):
        '''STEP 1: create environment instances and specifiy simulation span '''
        self.env=env
        self.span = span
        self.m_no = m_no
        self.m_list = []
        self.wc_no = wc_no
        self.wc_list = []
        m_per_wc = int(self.m_no / self.wc_no)
        '''STEP 2.1: create instances of machines'''
        for i in range(m_no):
            expr1 = '''self.m_{} = agent_machine.machine(env, {}, print = 0)'''.format(i,i) # create machines
            exec(expr1)
            expr2 = '''self.m_list.append(self.m_{})'''.format(i) # add to machine list
            exec(expr2)
        #print(self.m_list)
        '''STEP 2.2: create instances of work centers'''
        cum_m_idx = 0
        for i in range(wc_no):
            x = [self.m_list[m_idx] for m_idx in range(cum_m_idx, cum_m_idx + m_per_wc)]
            #print(x)
            expr1 = '''self.wc_{} = agent_workcenter.workcenter(env, {}, x)'''.format(i,i) # create work centers
            exec(expr1)
            expr2 = '''self.wc_list.append(self.wc_{})'''.format(i) # add to machine list
            exec(expr2)
            cum_m_idx += m_per_wc
        #print(self.wc_list)

        '''STEP 3: initialize the job creator'''
        # env, span, machine_list, workcenter_list, number_of_jobs, pt_range, due_tightness, E_utliz
        self.job_creator = job_creation.creation\
        (self.env, self.span, self.m_list, self.wc_list, [1,50], 3, 0.8, random_seed = True)
        self.job_creator.output()

        '''STEP 4: initialize machines and work centers'''
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)
        for i,m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i/m_per_wc)
            m.initialization(self.m_list,self.wc_list,self.job_creator,self.wc_list[wc_idx])

        '''STEP 5: set up the brains for workcenters'''
        self.routing_brain = brain_workcenter_R.routing_brain(self.env, self.job_creator, \
            self.m_list, self.wc_list, self.span/5, self.span)

        '''STEP 6: run the simulaiton'''
        env.run()
        self.routing_brain.check_parameter()

# create the environment instance for simulation
env = simpy.Environment()
# create the shop floor instance
span = 100000
m_no = 6
wc_no = 3
spf = shopfloor(env, span, m_no, wc_no)

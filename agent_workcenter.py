import simpy
import sys
sys.path
import random
import numpy as np
import torch
from tabulate import tabulate
import sequencing
import routing

class workcenter:
    def __init__(self, env, index, m_list, *args, **kwargs):
        # initialize the environment of simulation
        self.env = env
        self.m_list = m_list
        self.m_no = len(self.m_list)
        self.m_idx_list = [m.m_idx for m in m_list]
        self.wc_idx = index
        # queue list stores the index of jobs
        self.queue = []
        # each workcenter will have an independent storage for each type of job information
        # initialize all job-related information storage as empty lists
        self.sequence_list = [] # sequence of all queuing jobs
        self.pt_list = [] # processing time
        self.remaining_pt_list = [] # remaining processing time
        self.due_list = [] # due for each job
        self.weight_list = [] # importance of each job
        self.routing_data = []
        # print out the information of initial jobs
        self.print_info = True
        # initialize the routing(dispatching) event, for now it's untriggered
        # wait to be activated by job arrivals
        self.routing_event = self.env.event()
        # standby disctionary for storing incomplete experience for learning
        self.build_routing_experience = self.complete_experience_full
        self.incomplete_experience = {}
        # standby list for storing complete experience for learning
        # its content is taken by brain module periodically
        self.rep_memo = []
        # set the routing rule before start of simulation
        if 'rule' in kwargs:
            order = "self.job_routing = routing." + kwargs['rule']
            try:
                exec(order)
                print("workcenter {} uses {} routing rule".format(self.wc_idx, kwargs['rule']))
            except:
                print("Rule assigned to workcenter {} is invalid !".format(self.wc_idx))
                raise Exception
        else:
            # default routing rule is earliest available machine
            self.job_routing = routing.EA

    # this function should be called after __init__ to avoid deadlock
    # after the creation of all machine instances and initial job list for each machine
    # pass the list of machines to all machines so the shopfloor is established
    # the initial jobs are allocated through job_initialization module
    def initialization(self, job_creator):
        self.job_creator = job_creator
        self.dummy_pt = np.ones(self.m_no)*self.job_creator.avg_pt
        # initial information
        if self.print_info:
            print('work center {} contains machine {}'.format(self.wc_idx, self.m_idx_list))
            print('Initial %s jobs at workcenter %s are:'%(len(self.queue), self.wc_idx))
            job_info = [[self.queue[i],self.sequence_list[i], self.pt_list[i], self.due_list[i]] for i in range(len(self.queue))]
            print(tabulate(job_info, headers=['idx.','sqc.','proc.t.','due']))
            print('************************************')
        '''
        When all initial data, control variables, jobs and machines are ready
        PROCESS the routing function
        '''
        # initialize the data (empty lists) of all constituent machines
        for m in self.m_list:
            m.state_update_all()
        # dispatch one initial job to all constituent machines
        for i,m in enumerate(self.m_list):
            # get the first element of remaining pt list -> an array
            remaining_ptl = self.remaining_pt_list.pop(0)
            current_pt = remaining_ptl[0]
            estimated_slack_time = self.due_list[0] - self.env.now - np.sum(remaining_ptl.max(axis=1))
            remaining_ptl = np.delete(remaining_ptl, 0 ,axis=0) # remove the first element (row) of remaining_ptl, because it's now realized
            # add the information of job to machine agent's storage
            self.m_list[i].queue.append(self.queue.pop(0))
            self.m_list[i].sequence_list.append(self.sequence_list.pop(0))
            self.m_list[i].pt_list.append(self.pt_list.pop(0))
            # add slack, remaining.pt and due information to machines's storage
            self.m_list[i].slack_upon_arrival.append(estimated_slack_time)
            self.m_list[i].remaining_pt_list.append(remaining_ptl)
            self.m_list[i].due_list.append(self.due_list.pop(0))
            self.m_list[i].arrival_time_list.append(self.env.now)
            # after the job is routed to a machine, update the MACHINE's state
            self.m_list[i].state_update_after_job_arrival(15)
        # after the initial assignment, process the real routing function
        self.state_update_before_routing()
        self.env.process(self.routing())

    # The main process function that needs to be called in env.process()
    # the function run for one time when the new job / jobs join the queue
    def routing(self):
        while True:
            # program stucks here if event is not triggered
            # the event is triggerd by any job arrivals
            yield self.routing_event
            # could have more than 1 jobs join the queue at same time
            for j in range(len(self.queue)):
                # update current state
                self.state_update_before_routing()
                # always dispatch the first remaining job
                # get the expected processing time of all remaining processes
                remaining_ptl = self.remaining_pt_list.pop(0)
                # get the list of processing time of that job on all constituent machines
                current_pt = remaining_ptl[0]
                # calculate slack time based on expected waiting, processing time
                #estimated_slack_time = self.due_list[0] - self.env.now - self.average_waiting - np.sum(remaining_ptl.max(axis=1))
                estimated_slack_time = self.due_list[0] - self.env.now - self.least_waiting - np.sum(remaining_ptl.max(axis=1))
                # and remove pt of current operation on constituent machines
                remaining_ptl = np.delete(remaining_ptl, 0, axis=0) # remove the first element (row) of remaining_ptl, because it's dispatched to a specific machine
                remaining_pt = remaining_ptl.sum()
                #print(remaining_ptl,remaining_pt)
                # select a machine
                # the returned value is machine's position in self.m_list
                selected_machine_index = self.job_routing(self.queue[0], self.routing_data, current_pt, estimated_slack_time, self.wc_idx, np.sum(remaining_ptl.mean(axis=1)), len(remaining_ptl))
                # after assign this job to machine, the amount that machine's available time will increase by
                increased_available_time = current_pt[selected_machine_index]
                #print(self.queue[0], self.routing_data, self.machine_condition, current_pt)
                #print('ROUTING: Job %s to machine %s at time %s'%(self.queue[0], self.m_list[selected_machine_index].m_idx, self.env.now))
                # simply pop out the information of job, add to machine's storage
                self.m_list[selected_machine_index].queue.append(self.queue.pop(0))
                self.m_list[selected_machine_index].sequence_list.append(self.sequence_list.pop(0))
                self.m_list[selected_machine_index].pt_list.append(self.pt_list.pop(0))
                # add slack, remaining.pt and due information to machines's storage
                self.m_list[selected_machine_index].slack_upon_arrival.append(estimated_slack_time)
                self.m_list[selected_machine_index].remaining_pt_list.append(remaining_ptl)
                self.m_list[selected_machine_index].due_list.append(self.due_list.pop(0))
                self.m_list[selected_machine_index].arrival_time_list.append(max(self.env.now,self.m_list[selected_machine_index].release_time))
                #print('Jutified in time is:',self.m_list[selected_machine_index].release_time)
                # update the information of machine
                self.m_list[selected_machine_index].state_update_after_job_arrival(increased_available_time)
                try:
                    # relase from starvation
                    self.m_list[selected_machine_index].sufficient_stock.succeed()
                except:
                    # if not starving, just pass
                    pass
            # and delete this job from own queue
            # de-activate the routing(dispatching) event, wait to be activated again by new job arrivals
            self.routing_event = self.env.event()

    # this function accumulates data of consitituent machines
    def state_update_before_routing(self):
        # routing data: cumulative pt on machine, available time and que size
        self.routing_data = [machine.routing_data_generation() for machine in self.m_list]
        # get the average waiting time to calculate the slack time
        self.least_waiting = np.min(self.routing_data, axis=0)[1]
        #print(self.routing_data, self.least_waiting)
        avg = np.average(np.array(self.routing_data).clip(0),axis=0)
        self.average_workcontent = avg[0]
        self.average_waiting = avg[1]
        #print(self.wc_idx,self.routing_data,self.average_waiting)
        # update condition of machines
        self.machine_condition = np.array([machine.working_event.triggered*1 for machine in self.m_list])

    # the function of building incomplete experience is in brain module
    # this function is called only if the routing_learning_event of constituent machine is triggered
    # when this function is called upon the completion of an operation
    def complete_experience_full(self, job_idx, slack_change, critical_level_R):
        # first update current state, extract data from constituent machines
        self.state_update_before_routing()
        # get the state at the time of job output
        s_t = self.build_state(self.routing_data, self.dummy_pt, 0, self.wc_idx)
        # calculate reward of action
        # if slack time of job reduces, reward in [-1,0], if slack time increases, in [0,1]
        r_t = torch.tensor(np.clip(slack_change*critical_level_R/20, -1, 1),dtype=torch.float)
        self.job_creator.rt_reward_record.append([self.env.now, r_t])
        #r_t = torch.tensor(np.clip(slack_change/40, -1, 1),dtype=torch.float)
        #print('build complete memory',job_idx,[r_t, s_t],self.env.now)
        # append reward and new state to corresponding incomplete experience
        self.incomplete_experience[job_idx] += [r_t, s_t]
        # and pop out this complete experience, add it to complete replay memory
        self.rep_memo.append(self.incomplete_experience.pop(job_idx))

    def complete_experience_global_reward(self, job_idx, slack_change, critical_level_R):
        self.state_update_before_routing()
        s_t = self.build_state(self.routing_data, self.dummy_pt, 0, self.wc_idx)
        self.incomplete_experience[job_idx] += [s_t]

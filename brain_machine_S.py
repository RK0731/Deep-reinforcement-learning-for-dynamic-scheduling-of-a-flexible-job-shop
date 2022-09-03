import random
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tabulate import tabulate
import sequencing

class sequencing_brain:
    def __init__(self, env, job_creator, all_machines, target_machines, warm_up, span, *args, **kwargs):
        # initialize the environment and the machines to be controlled
        self.env = env
        self.job_creator = job_creator
        # m_list contains all machines on shop floor, we need them to collect data
        self.m_list = all_machines
        self.m_no = len(self.m_list)
        # target_m_list are those be controled by DRL
        self.target_m_list = target_machines # usually all machines on shopfloor
        self.target_m_no = len(self.target_m_list)
        self.warm_up = warm_up
        self.span = span
        # and build dicts that equals number of machines to be controlled in job creator
        self.job_creator.build_sqc_experience_repository(self.target_m_list)
        # activate the sequencing learning event of machines so they will collect data
        # and build dictionary to store the data
        print("+++ Take over all machines, activate learning mode +++")
        for m in self.m_list:
            m.sequencing_learning_event.succeed()
            m.job_sequencing = self.action_default
        print('+++ Take over sequencing / reward function of target machines +++')
        for m in self.target_m_list:
            m.job_sequencing = self.action_warm_up
        # list that contains available rules, and use SPT for the first phase of warmup
        self.func_list = [sequencing.SPT,sequencing.WINQ,sequencing.MS,sequencing.CR]
        self.func_selection = 0
        # action space, consists of all selectable rules
        self.output_size = len(self.func_list)
        '''
        choose the reward function for machines
        '''
        if 'reward_function' in kwargs:
            order = 'm.reward_function = m.get_reward{}'.format(kwargs['reward_function'])
            for m in self.target_m_list:
                exec(order)
        else:
            print('WARNING: reward function is not specified')
            raise Exception
        '''
        chooose the architecture of state space and architecture of ANN
        and specify the address to store the trained state-dict
        needs to be specified in kwargs, otherwise abstract networks + abstract state space
        '''
        if 'TEST' in kwargs and kwargs['TEST']:
            print("---> Normalization mode ON <---")
            self.input_size = len(self.state_normalization(self.m_list[0].sequencing_data_generation()))
            # initialize the policy network and target network
            self.sequencing_action_NN = network_normalization(self.input_size, self.output_size, 3, 9, self.target_m_list)
            # deepcopy the policy NN, as the sequencing NN
            self.sequencing_target_NN = copy.deepcopy(self.sequencing_action_NN)
            # define the name and directory of the model (for saving the model)
            self.address_seed = "{}\\sequencing_models\\Norm_rwd"+str(kwargs['reward_function'])+"_{}.pt"
            # build state with additional scaling factor
            self.build_state = self.state_normalization
            self.action_DRL = self.action_sqc_rule
            for m in self.target_m_list:
                m.build_state = self.state_normalization
        elif 'MR' in kwargs and kwargs['MR']:
            print("---X Minimal repetition (direct) mode ON X---")
            self.input_size =  self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.output_size = 4
            self.sequencing_action_NN = network_value_based(self.input_size, self.output_size)
            self.sequencing_target_NN = copy.deepcopy(self.sequencing_action_NN)
            self.address_seed = "{}\\sequencing_models\\MR_direct_rwd"+str(kwargs['reward_function'])+".pt"
            self.build_state = self.state_direct
            self.train = self.train_Double_DQN
            self.action_DRL = self.action_direct
            for m in self.m_list:
                m.build_state = self.state_direct
        elif 'MC' in kwargs and kwargs['MC']:
            print("---> Multi-Channel (MC) mode ON <---")
            self.input_size = len(self.state_multi_channel(self.m_list[0].sequencing_data_generation()))
            self.sequencing_action_NN = network_validated(self.input_size, self.output_size)
            self.sequencing_target_NN = copy.deepcopy(self.sequencing_action_NN)
            self.address_seed = "{}\\sequencing_models\\MC_rwd"+str(kwargs['reward_function'])+".pt"
            self.build_state = self.state_multi_channel
            self.train = self.train_validated
            self.action_DRL = self.action_sqc_rule
            for m in self.target_m_list:
                m.build_state = self.state_multi_channel
        else:
            print("WARNING: ANN TYPE NOT SPECIFIED !!!!")

        if "trained_parameter" in kwargs: # import trained parameters for better efficiency
            for m in self.target_m_list:
                import_address = "{}\\sequencing_models\\validated_"+kwargs["trained_parameter"]+".pt"
                self.sequencing_action_NN.network.load_state_dict(torch.load(import_address.format(sys.path[0])))
            print("IMPORT FROM:", import_address)

        '''
        specify new address seed for storing the trained parameters
        '''
        if 'store_to' in kwargs:
            self.address_seed = "{}\\sequencing_models\\" + str(kwargs['address_seed']) + ".pt"
            print("New address seed:", self.address_seed)
        # initialize initial replay memory, a dictionary that contains empty lists of replay memory for machines
        self.rep_memo = []
        # some training-related parameters
        self.minibatch_size = 64
        self.rep_memo_size = 1024
        self.sequencing_action_NN_training_interval = 5 # training frequency of updating the policy network
        self.sequencing_action_NN_training_time_record = []
        self.sequencing_target_NN_update_interval = 500  # synchronize the weights of NN every 200 time units
        self.sequencing_target_NN_update_time_record = []
        # Initialize the parameters for learning of DRL
        self.discount_factor = 0.8 # how much agent care long-term rewards
        self.epsilon = 0.15  # chance of exploration
        # initialize the lists of data memory for all target machines
        # record the loss
        self.loss_time_record = []
        self.loss_record = []
        # processes
        if kwargs['IQL'] or kwargs['I_DDQN']:
            self.env.process(self.training_process_independent())
            self.env.process(self.update_rep_memo_independent_process())
            self.rep_memo = {} # replace the list by dict
            for m in self.target_m_list:
                self.rep_memo[m.m_idx] = []
            self.build_initial_rep_memo = self.build_initial_rep_memo_independent
            #self.rep_memo_size /= self.m_no # size for independent replay memory
        else: # default mode is parameter sharing
            self.env.process(self.training_process_parameter_sharing())
            self.env.process(self.update_rep_memo_parameter_sharing_process())
            self.build_initial_rep_memo = self.build_initial_rep_memo_parameter_sharing
        self.env.process(self.warm_up_process())
        self.env.process(self.update_training_setting_process())
        #self.env.process(self.update_learning_rate_process())

    '''
    1. downwards for functions that required for the simulation
       including the warm-up, action functions and multiple sequencing rules
    '''


    def warm_up_process(self):
        '''
        Phase 1.1 : warm-up
        within this phase, agent shift between sequencing rules
        '''
        # take over the target mahcines' sequencing function
        # from FIFO to here
        # first half of warm-up period
        for idx,func in enumerate(self.func_list):
            self.func_selection = idx
            print('set to rule {}'.format(func))
            yield self.env.timeout(self.warm_up/5)
        '''
        Phase 1.2 : random exploration
        within this phase, agents choose random sequencing action, to accumulate experience
        '''
        # change the target machines' sequencing function to random exploration, to try all combinations
        for m in self.target_m_list:
            m.job_sequencing = self.action_random_exploration
        # second half of warm-up period
        print("start random exploration from time {} till time {}".format(self.env.now, self.warm_up))
        yield self.env.timeout(self.warm_up - self.env.now - 1)
        '''
        After the warm up, build initial replay memory
        and hand over to sequencing NN
        '''
        # after the warm up period, build replay memory and start training
        self.build_initial_rep_memo()
        # hand over the target machines' sequencing function to policy network
        for m in self.target_m_list:
            m.job_sequencing = self.action_DRL

    # for those not controlled by brain, use default FIFO rule for sequencing
    def action_default(self, sqc_data):
        m_idx = sqc_data[-1]
        job_position = sequencing.FIFO(sqc_data)
        j_idx = sqc_data[-2][job_position]
        return job_position

    # used in the first half phase of warm up
    def action_warm_up(self, sqc_data):
        s_t = self.build_state(sqc_data)
        m_idx = sqc_data[-1]
        #print('state:',self.env.now,s_t)
        # action is index of rule, NOT index or position of job
        a_t = torch.tensor(self.func_selection)
        # the decision is made by either of the available sequencing rule
        job_position = self.func_list[self.func_selection](sqc_data)
        j_idx = sqc_data[-2][job_position]
        # add the state at sequencing to job creator's corresponding repository
        self.build_experience(j_idx,m_idx,s_t,a_t)
        return job_position

    # used in the second phase of warm up
    def action_random_exploration(self, sqc_data):
        s_t = self.build_state(sqc_data)
        m_idx = sqc_data[-1]
        #print('state:',self.env.now,s_t)
        # action is index of rule, NOT index or position of job
        self.func_selection = np.random.randint(self.output_size)
        a_t = torch.tensor(self.func_selection)
        # the decision is made by either of the available sequencing rule
        job_position = self.func_list[a_t](sqc_data)
        j_idx = sqc_data[-2][job_position]
        # add the state at sequencing to job creator's corresponding repository
        self.build_experience(j_idx,m_idx,s_t,a_t)
        return job_position

    def action_sqc_rule(self, sqc_data):
        s_t = self.build_state(sqc_data)
        m_idx = sqc_data[-1]
        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            a_t = torch.randint(0,self.output_size,[])
            #print('Random Action / By Brain')
        else:
            # input state to policy network, produce the state-action value
            value = self.sequencing_action_NN.forward(s_t.reshape([1,1,self.input_size]), m_idx)
            # greedy policy
            a_t = torch.argmax(value)
            #print("State is:", s_t)
            #print('State-Action Values:', value)
            #print('Sequencing NN, action %s / By Brain'%(a_t))
        # the decision is made by one of the available sequencing rule
        job_position = self.func_list[a_t](sqc_data)
        j_idx = sqc_data[-2][job_position]
        # add the state at sequencing to job creator's corresponding repository
        self.build_experience(j_idx,m_idx,s_t,a_t)
        return job_position

    def action_direct(self, sqc_data): # strategic idleness is prohibitted
        s_t = self.build_state(sqc_data)
        m_idx = sqc_data[-1]
        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            a_t = torch.randint(0,self.output_size,[])
            #print('Random Selection:', a_t)
        else:
            # input state to action network, produce the state-action value
            value = self.sequencing_action_NN.forward(s_t.reshape([1]+self.input_size_as_list),m_idx).squeeze()
            # greedy policy
            a_t = torch.argmax(value)
            #print("State is:", s_t)
            #print('State-Action Values:', value)
            #print('Direct Selection: %s'%(a_t))
        job_position, j_idx = self.action_conversion(a_t)
        # add the state at sequencing to job creator's corresponding repository
        self.build_experience(j_idx,m_idx,s_t,a_t)
        return job_position

    def action_conversion(self, a_t):
        #print(a_t)
        job_position = self.correspondence_pos[a_t]
        j_idx = self.correspondence_idx[a_t]
        #print(self.correspondence_idx)
        #print(self.correspondence_pos)
        #print('selected job idx: %s, position in queue: %s'%(j_idx, job_position))
        return job_position, j_idx

    '''
    2. downwards are functions used for building the state of the experience (replay memory)
    '''
    '''
    local data consists of:
    0            1                 2         3        4
    [current_pt, remaining_job_pt, due_list, env.now, completion_rate,
    5              6      7     8     9        10         11           12/-3   13/-2  14/-1
    time_till_due, slack, winq, avlm, next_pt, rem_no_op, waited_time, wc_idx, queue, m_idx]
    '''

    def state_multi_channel(self, sqc_data):
        # information in job number, global and local
        in_system_job_no = self.job_creator.in_system_job_no
        local_job_no = len(sqc_data[0])
        # the information of coming job (currently being processed by other machines)
        #print('coming jobs:',self.job_creator.next_wc_list, self.job_creator.arriving_job_slack_list, self.job_creator.release_time_list, sqc_data[-3])
        arriving_jobs = np.where(self.job_creator.next_wc_list == sqc_data[-3])[0] # return the index of coming jobs
        arriving_job_no = arriving_jobs.size  # expected arriving job number
        if arriving_job_no: # if there're jobs coming at your workcenter
            arriving_job_time = (self.job_creator.release_time_list[arriving_jobs] - self.env.now).mean() # average time from now when next job arrives at workcenter
            arriving_job_slack = (self.job_creator.arriving_job_slack_list[arriving_jobs]).mean() # what's the average slack time of the arriving job
        else:
            arriving_job_time = 0
            arriving_job_slack = 0
        #print(arriving_jobs, arriving_job_no, arriving_job_time, arriving_job_slack, self.env.now, sqc_data[-3])
        # information of progression of jobs, get from the job creator
        global_comp_rate = self.job_creator.comp_rate
        global_realized_tard_rate = self.job_creator.realized_tard_rate
        global_exp_tard_rate = self.job_creator.exp_tard_rate
        available_time = (self.job_creator.available_time_list - self.env.now).clip(0,None)
        # get the pt of all remaining jobs in system
        rem_pt = []
        # need loop here because remaining_pt have different length
        for m in self.m_list:
            for x in m.remaining_pt_list:
                rem_pt += x.tolist()
        # processing time related data
        pt_share = available_time[sqc_data[-1]] / sum(available_time) # sum of pt / sum of available time
        global_pt_CV = np.std(rem_pt) / np.mean(rem_pt)
        # information of queuing jobs in queue
        local_pt_sum = np.sum(sqc_data[0])
        local_pt_mean = np.mean(sqc_data[0])
        local_pt_min = np.min(sqc_data[0])
        local_pt_CV = np.std(sqc_data[0]) / local_pt_mean
        # information of queuing jobs in remaining processing time
        local_remaining_pt_sum = np.sum(sqc_data[1])
        local_remaining_pt_mean = np.mean(sqc_data[1])
        local_remaining_pt_max = np.max(sqc_data[1])
        local_remaining_pt_CV = np.std(sqc_data[1]) / local_remaining_pt_mean
        # information of WINQ
        avlm_mean = np.mean(sqc_data[8])
        avlm_min = np.min(sqc_data[8])
        avlm_CV = np.std(sqc_data[8]) / avlm_mean
        # time-till-due related data:
        time_till_due = sqc_data[5]
        realized_tard_rate = time_till_due[time_till_due<0].size / local_job_no # ratio of tardy jobs
        ttd_sum = time_till_due.sum()
        ttd_mean = time_till_due.mean()
        ttd_min = time_till_due.min()
        ttd_CV = (time_till_due.std() / ttd_mean).clip(-2,2)
        # slack-related data:
        slack = sqc_data[6]
        exp_tard_rate = slack[slack<0].size / local_job_no # ratio of jobs expect to be tardy
        slack_sum = slack.sum()
        slack_mean = slack.mean()
        slack_min = slack.min()
        slack_CV = (slack.std() / slack_mean).clip(-2,2)
        # use raw data, and leave the magnitude adjustment to normalization layers
        no_info = [in_system_job_no, arriving_job_no, local_job_no] # info in job number
        pt_info = [local_pt_sum, local_pt_mean, local_pt_min] # info in processing time
        remaining_pt_info = [local_remaining_pt_sum, local_remaining_pt_mean, local_remaining_pt_max, avlm_mean, avlm_min] # info in remaining processing time
        ttd_slack_info = [ttd_mean, ttd_min, slack_mean, slack_min, arriving_job_slack] # info in time till due
        progression = [pt_share, global_comp_rate, global_realized_tard_rate, global_exp_tard_rate] # progression info
        heterogeneity = [global_pt_CV, local_pt_CV, ttd_CV, slack_CV, avlm_CV] # heterogeneity
        # concatenate the data input
        s_t = np.nan_to_num(np.concatenate([no_info, pt_info, remaining_pt_info, ttd_slack_info, progression, heterogeneity]),nan=0,posinf=1,neginf=-1)
        # convert to tensor
        s_t = torch.tensor(s_t, dtype=torch.float)
        return s_t

    def state_direct(self, sqc_data): # presenting information of job which satisfies certain criteria
        '''STEP 1: check queuing jobs, if any, clip the sqc data'''
        # number of candidate jobs
        no_candidate_jobs = len(sqc_data[0])
        if no_candidate_jobs == 1: # if there's only one queuing job, simply copy the info of the only job (most common case)
            # original sqc_data contains lots of things that won't be used, create a clipped copy of it
            clipped_data = np.concatenate([sqc_data[0], sqc_data[1], sqc_data[6], sqc_data[8], sqc_data[11]])
            s_t = [clipped_data for i in range(4)]
            # and set the correspondence to the first job in the queue
            self.correspondence_pos = [0 for i in range(4)]
            self.correspondence_idx = [0 for i in range(4)]
        elif no_candidate_jobs == 0 : # if there's no queuing job, create dummy state that all enrties are 0
            s_t = [np.array([0 for i in range(5)]) for i in range(4)]
            # and set the correspondence to dummy value
            self.correspondence_pos = [-1 for i in range(4)]
            self.correspondence_idx = [-1 for i in range(4)]
        else: # if there's multiple jobs, try include them exhaustively
            # empty list of position and index of candidate jobs
            s_t = [] # initialize empty state
            self.correspondence_pos = []
            self.correspondence_idx = []
            clipped_data = np.array([sqc_data[0], sqc_data[1], sqc_data[6], sqc_data[8], sqc_data[11]])
            # copy the lists for exhaustive inclusion
            copied_clipped_data = clipped_data.copy() # jobs would be gradually kicked out from a copy of clipped_data
            exhaust_idx = sqc_data[-2].copy() # also kick out from list of indexes
            exhaust_pos = np.arange(no_candidate_jobs) # also kick out from list of position
            row_number = [0,1,2,3] # spt, lwkr, ms, avlm
            row = 0
            # first try to include all jobs, reduce duplication as possible
            try:
                for i in range(4):
                    #print(copied_clipped_data, exhaust_idx, exhaust_pos, self.env.now)
                    no_duplication_pos = np.argmin(copied_clipped_data[row])
                    job_idx = exhaust_idx[no_duplication_pos]
                    job_pos = exhaust_pos[no_duplication_pos]
                    self.correspondence_idx.append(job_idx)
                    self.correspondence_pos.append(job_pos)
                    s_t.append(copied_clipped_data[:,no_duplication_pos])
                    row += 1
                    # kick out the selected job from exhaust list
                    copied_clipped_data = np.delete(copied_clipped_data, no_duplication_pos, axis=1)
                    exhaust_idx = np.delete(exhaust_idx, no_duplication_pos)
                    exhaust_pos = np.delete(exhaust_pos, no_duplication_pos)
            # if number of candidate job less than 4 (expection raise), then return to normal procedure to complete the state
            except:
                for i in range(row,4):
                    normal_pos = np.argmin(clipped_data[row])
                    normal_idx = sqc_data[-2][normal_pos]
                    self.correspondence_idx.append(normal_idx)
                    self.correspondence_pos.append(normal_pos)
                    s_t.append(clipped_data[:,normal_pos])
                    row += 1
        '''STEP 2: get information of arriving jobs and others'''
        arriving_jobs = np.where(self.job_creator.next_wc_list == sqc_data[-3])[0] # see if there are jobs will arrive
        self.arriving_job_exists = bool(len(arriving_jobs)) # get the bool variable to represent whether arriving job exists
        # get the available time of machine itself
        avlm_self = self.job_creator.available_time_list[sqc_data[-1]] - self.env.now
        #print(self.job_creator.next_machine_list, self.job_creator.release_time_list, self.env.now)
        #print('%s arriving jobs from machine %s'%(self.arriving_job_exists,arriving_jobs))
        if self.arriving_job_exists: # if there are arriving jobs
            # get the exact next job that will arrive at machine out of all arriving jobs
            pos = arriving_jobs[self.job_creator.release_time_list[arriving_jobs].argmin()]
            arriving_j_idx = self.job_creator.current_j_idx_list[pos]
            # and retrive the information of this job
            pt_self = self.job_creator.next_pt_list[pos]
            rem_pt = self.job_creator.arriving_job_rempt_list[pos]
            slack = self.job_creator.arriving_job_slack_list[pos]
            self.strategic_idleness_time = self.job_creator.release_time_list[arriving_jobs].min() - self.env.now # how long to wait if agent decide to wait for arriving job
            arriving_job_info = np.array([pt_self, rem_pt, slack, avlm_self, self.strategic_idleness_time])
        else: # if there is no arriving job
            arriving_j_idx = None
            arriving_job_info = np.array([0, 0, 0, avlm_self, 0])
            self.strategic_idleness_time = 0 # no need to wait for any arriving jobs
        # add position and index of arriving job to correspondence
        self.correspondence_pos.append(len(sqc_data[0]))
        self.correspondence_idx.append(arriving_j_idx)
        s_t.append(arriving_job_info)
        '''STEP 3: finally, convert list to tensor and output it'''
        s_t = torch.FloatTensor(s_t)
        #print('state:',s_t)
        return s_t

    # add the experience to job creator's incomplete experiece memory
    def build_experience(self,j_idx,m_idx,s_t,a_t):
        self.job_creator.incomplete_rep_memo[m_idx][self.env.now] = [s_t,a_t]


    '''
    3. downwards are functions used for building / updating replay memory
    '''


    def build_initial_rep_memo_parameter_sharing(self):
        #print(self.job_creator.rep_memo)
        for m in self.target_m_list:
            # copy the initial memoery from corresponding rep_memo from job creator
            #print('%s complete and %s incomplete experience for machine %s'%(len(self.job_creator.rep_memo[m.m_idx]), len(self.job_creator.incomplete_rep_memo[m.m_idx]), m.m_idx))
            #print(self.job_creator.incomplete_rep_memo[m.m_idx])
            self.rep_memo += self.job_creator.rep_memo[m.m_idx].copy()
            # and clear the replay memory in job creator, keep it updated
            self.job_creator.rep_memo[m.m_idx] = []
        # and the initial dummy TDerror
        self.rep_memo_TDerror = torch.ones(len(self.rep_memo),dtype=torch.float)
        print('INITIALIZATION - replay_memory')
        print(tabulate(self.rep_memo, headers = ['s_t','a_t','s_t+1','r_t']))
        print('INITIALIZATION - size of replay memory:',len(self.rep_memo))
        print('---------------------------initialization accomplished-----------------------------')

    def build_initial_rep_memo_independent(self):
        #print(self.job_creator.rep_memo)
        print('INITIALIZATION - replay_memory')
        for m in self.target_m_list:
            # copy the initial memoery from corresponding rep_memo from job creator
            #print('%s complete and %s incomplete experience for machine %s'%(len(self.job_creator.rep_memo[m.m_idx]), len(self.job_creator.incomplete_rep_memo[m.m_idx]), m.m_idx))
            #print(self.job_creator.incomplete_rep_memo[m.m_idx])
            self.rep_memo[m.m_idx] += self.job_creator.rep_memo[m.m_idx].copy()
            # and clear the replay memory in job creator, keep it updated
            self.job_creator.rep_memo[m.m_idx] = []
            print(tabulate(self.rep_memo[m.m_idx], headers = ['s_t','a_t','s_t+1','r_t']))
            print('INITIALIZATION - size of replay memory:',len(self.rep_memo[m.m_idx]))
        print('---------------------------initialization accomplished-----------------------------')

    # update the replay memory periodically
    def update_rep_memo_parameter_sharing_process(self):
        yield self.env.timeout(self.warm_up)
        while self.env.now < self.span:
            for m in self.m_list:
                # add new memoery from corresponding rep_memo from job creator
                self.rep_memo += self.job_creator.rep_memo[m.m_idx].copy()
                # and assign top priority to new experiences
                self.rep_memo_TDerror = torch.cat([self.rep_memo_TDerror, torch.ones(len(self.job_creator.rep_memo[m.m_idx]),dtype=torch.float)])
                # and clear the replay memory in job creator, keep it updated
                self.job_creator.rep_memo[m.m_idx] = []
            # clear the obsolete experience periodically
            if len(self.rep_memo) > self.rep_memo_size:
                truncation = len(self.rep_memo)-self.rep_memo_size
                self.rep_memo = self.rep_memo[truncation:]
                self.rep_memo_TDerror = self.rep_memo_TDerror[truncation:]
            #print(self.rep_memo_TDerror)
            yield self.env.timeout(self.sequencing_action_NN_training_interval*10)

    def update_rep_memo_independent_process(self):
        yield self.env.timeout(self.warm_up)
        while self.env.now < self.span:
            for m in self.m_list:
                # add new memoery from corresponding rep_memo from job creator
                self.rep_memo[m.m_idx] += self.job_creator.rep_memo[m.m_idx].copy()
                # and assign top priority to new experiences
                #self.rep_memo_TDerror = torch.cat([self.rep_memo_TDerror, torch.ones(len(self.job_creator.rep_memo[m.m_idx]),dtype=torch.float)])
                # and clear the replay memory in job creator, keep it updated
                self.job_creator.rep_memo[m.m_idx] = []
            # clear the obsolete experience periodically
            if len(self.rep_memo[m.m_idx]) > self.rep_memo_size:
                truncation = len(self.rep_memo[m.m_idx])-self.rep_memo_size
                self.rep_memo[m.m_idx] = self.rep_memo[m.m_idx][truncation:]
                #self.rep_memo_TDerror = self.rep_memo_TDerror[truncation:]
            #print(self.rep_memo_TDerror)
            yield self.env.timeout(self.sequencing_action_NN_training_interval*10)


    '''
    4. downwards are functions used in the training of DRL, including the dynamic training process
       dynamic training parameters update
    '''

    # print out the functions and classes used in the training
    def check_parameter(self):
        print('------------- Training Parameter Check -------------')
        print("Address seed:",self.address_seed)
        print('Rwd.Func.:',self.target_m_list[0].reward_function.__name__)
        print('State Func.:',self.build_state.__name__)
        print('ANN:',self.sequencing_action_NN.__class__.__name__)
        print('Discount rate:',self.discount_factor)
        print('*** SCENARIO:')
        print("Configuration: {} work centers, {} machines".format(len(self.job_creator.wc_list),len(self.m_list)))
        print("PT heterogeneity:",self.job_creator.pt_range)
        print('Due date tightness:',self.job_creator.tightness)
        print('Utilization rate:',self.job_creator.E_utliz)
        print('------------------------------------------------------------')

    def loss_record_output(self,**kwargs):
        fig = plt.figure(figsize=(10,5.5))
        loss_record = fig.add_subplot(1,1,1)
        loss_record.set_xlabel('Iterations of training ('+r'$\times 10^3$'+')')
        loss_record.set_ylabel('Loss of training')
        iterations = np.arange(len(self.loss_record))
        loss_record.scatter(iterations, self.loss_record,s=1,color='r', alpha=0.3,zorder=3)
        # moving average
        x = 50
        loss_record.plot(np.arange(x/2,len(self.loss_record)-x/2+1,1),np.convolve(self.loss_record, np.ones(x)/x, mode='valid'),color='k',label='moving average')
        # limits, grids,
        ylim=0.25
        loss_record.set_xlim(0,len(self.loss_record))
        loss_record.set_ylim(0.05,ylim)
        xtick_interval = 1000
        loss_record.set_xticks(np.arange(0,len(self.loss_record)+1,xtick_interval))
        loss_record.set_xticklabels(np.arange(0,len(self.loss_record)/xtick_interval,1).astype(int),rotation=30, ha='right', rotation_mode="anchor", fontsize=8.5)
        loss_record.set_yticks(np.arange(0.05, ylim+0.01, 0.01))
        loss_record.grid(axis='x', which='major', alpha=0.5, zorder=0, )
        loss_record.grid(axis='y', which='major', alpha=0.5, zorder=0, )
        loss_record.legend()
        # dual axis
        ax_time = loss_record.twiny()
        ax_time.set_xlabel('Time in simulation ('+r'$\times 10^3$'+', excluding warm up phase)')
        ax_time.set_xlim(self.warm_up,self.span)
        ax_time.set_xticks(np.arange(self.warm_up,self.span+1,xtick_interval*2))
        ax_time.set_xticklabels(np.arange(self.warm_up/xtick_interval,self.span/xtick_interval+1,2).astype(int),rotation=30, ha='left', rotation_mode="anchor", fontsize=8.5)
        loss_record.set_title("Sequencing Agent Training Loss / {}-operation test".format(len(self.job_creator.wc_list)))
        fig.subplots_adjust(top=0.8, bottom=0.1, right=0.9)
        plt.show()
        # save the loss record to .txt file
        record = open("sequencing_models\\training_loss_record.txt","w")
        record.write(str(self.loss_record))
        record.close()
        # save the figure if required
        if 'save' in kwargs and kwargs['save']:
            address = sys.path[0]+"//experiment_result//SA_loss_{}wc_{}m.png".format(len(self.job_creator.wc_list),len(self.m_list))
            fig.savefig(address, dpi=500, bbox_inches='tight')
            print('figure saved to'+address)
        return

    def reward_record_output(self,**kwargs):
        reward_record = fig.add_subplot(1,1,1)
        reward_record.set_xlabel('Time')
        reward_record.set_ylabel('Reward')
        time = np.array(self.job_creator.sqc_reward_record).transpose()[0]
        rewards = np.array(self.job_creator.sqc_reward_record).transpose()[1]
        #print(time, rewards)
        reward_record.scatter(time, rewards, s=1,color='g', alpha=0.3, zorder=3)
        reward_record.set_xlim(0,self.span)
        reward_record.set_ylim(-1.1,1.1)
        xtick_interval = 2000
        reward_record.set_xticks(np.arange(0,self.span+1,xtick_interval))
        reward_record.set_xticklabels(np.arange(0,self.span+1,xtick_interval),rotation=30, ha='right', rotation_mode="anchor", fontsize=8.5)
        reward_record.set_yticks(np.arange(-1, 1, 0.1))
        reward_record.grid(axis='x', which='major', alpha=0.5, zorder=0, )
        reward_record.grid(axis='y', which='major', alpha=0.5, zorder=0, )
        # moving average
        x = 50
        print(len(time))
        reward_record.plot(time[int(x/2):len(time)-int(x/2)+1],np.convolve(rewards, np.ones(x)/x, mode='valid'),color='k',label="moving average")
        reward_record.legend()
        plt.show()
        # save the figure if required
        fig.subplots_adjust(top=0.5, bottom=0.5, right=0.9)
        if 'save' in kwargs and kwargs['save']:
            fig.savefig(sys.path[0]+"//experiment_result//SA_reward_{}wc_{}m.png".format(len(self.job_creator.wc_list),len(self.m_list)), dpi=500, bbox_inches='tight')
        return

    def training_process_parameter_sharing(self):
        # wait for the warm up
        yield self.env.timeout(self.warm_up)
        # pre-train the policy NN before hand over to it
        for i in range(10):
            self.train()
        # periodic training
        while self.env.now < self.span:
            self.train()
            yield self.env.timeout(self.sequencing_action_NN_training_interval)
        # end the training after span time
        # and store the trained parameters
        print('FINAL- replay_memory')
        print(tabulate(self.rep_memo, headers = ['s_t','a_t','s_t+1','r_t']))
        print('FINAL - size of replay memory:',len(self.rep_memo))
        # specify the address to store the model / state_dict
        address = self.address_seed.format(sys.path[0])
        # save the parameters of policy / action network after training
        torch.save(self.sequencing_action_NN.network.state_dict(), address)
        # after the training, print out the setting of DRL architecture
        print("Training terminated, store trained parameters to: {}".format(self.address_seed))

    def training_process_independent(self):
        # wait for the warm up
        yield self.env.timeout(self.warm_up)
        # pre-train the policy NN before hand over to it
        for m in self.target_m_list:
            for i in range(10):
                self.train(m.m_idx)
        # periodic training
        while self.env.now < self.span:
            for m in self.target_m_list:
                self.train(m.m_idx)
            yield self.env.timeout(self.sequencing_action_NN_training_interval)
        # end the training after span time
        # and store the trained parameters
        for m in self.target_m_list:
            print('FINAL - replay_memory of machine %s is:'%m.m_idx,"size:",len(self.rep_memo[m.m_idx]))
            print(tabulate(self.rep_memo[m.m_idx],headers = ['s_t','a_t','s_t+1','r_t']))
            # specify the address to store the model / state_dict
            address = self.address_seed.format(sys.path[0],str(m.m_idx))
            # save the parameters of policy / action network after training
            torch.save(self.sequencing_action_NN.module_dict[m.m_idx].state_dict(), address)
        # after the training, print out the setting of DRL architecture
        print("Training terminated, address of trained parameters: {}".format(self.address_seed))

    # synchronize the ANN and TNN, and some settings
    def update_training_setting_process(self):
        # one second after the initial training, so we can have a slightly better target network
        yield self.env.timeout(self.warm_up+1)
        while self.env.now < self.span:
            # synchronize the parameter of policy and target network
            self.sequencing_target_NN = copy.deepcopy(self.sequencing_action_NN)
            #self.epsilon -= 0.0015
            print('--------------------------------------------------------')
            print('the target network and epsilion are updated at time %s' % self.env.now)
            print('--------------------------------------------------------')
            yield self.env.timeout(self.sequencing_target_NN_update_interval)

    # reduce the learning rate periodically
    def update_learning_rate_process(self):
        # one second after the initial training
        yield self.env.timeout(self.warm_up)
        reduction = (self.sequencing_action_NN.lr - self.sequencing_action_NN.lr/10)/10
        while self.env.now < self.span:
            yield self.env.timeout((self.span-self.warm_up)/10)
            # reduce the learning rate
            self.sequencing_action_NN.lr -= reduction
            #self.epsilon -= 0.0015
            print('--------------------------------------------------------')
            print('learning rate adjusted to {} at time {}'.format(self.sequencing_action_NN.lr, self.env.now))
            print('--------------------------------------------------------')

    # the function that draws minibatch and trains the policy NN
    def train_I_DDQN(self, m_idx):
        print(".............TRAINING of module %s.............%s"%(m_idx,self.env.now))
        """
        draw the random minibatch to train the network, randomly
        every element in the replay menory is a list [s_0, a_0, s_1, r_0]
        all element of this list are tensors
        """
        size = min(len(self.rep_memo[m_idx]),self.minibatch_size)
        minibatch = random.sample(self.rep_memo[m_idx],size)
        '''
        slice, and stack the 1D tensors to several 3D tensors (batch, channel, vector)
        the "torch.stack" is only applicable when the augment is a list of tensors, or multi-dimensional tensor
        '''
        sample_s0_batch = torch.stack([data[0] for data in minibatch], dim=0).reshape(size,1,self.input_size)
        sample_s1_batch = torch.stack([data[2] for data in minibatch], dim=0).reshape(size,1,self.input_size)
        sample_a0_batch = torch.stack([data[1] for data in minibatch], dim=0).reshape(size,1)
        sample_r0_batch = torch.stack([data[3] for data in minibatch], dim=0).reshape(size,1)
        '''
        the size of these batches:
        sample_s0_batch = sample_s1_batch = minibatch size * 1 * input_size
        sample_a0_batch = sample_r0_batch = minibatch size * m_no
        sample_r0_batch = minibatch size
        '''
        # get the Q value (current value of state-action pair) of s0
        Q_0 = self.sequencing_action_NN.forward(sample_s0_batch, m_idx)
        #print('Q_0 is:\n', Q_0)
        #print('a_0 is:', sample_a0_batch)
        # get the current state-action value of actions that would have been taken
        current_value = Q_0.gather(1, sample_a0_batch)
        #print('current value is:', current_value)
        '''
        get the Q Value of s_1 in both action and target network, to estimate the state value
        architecture is DDQN, NOT DQN !!!
        evaluate the greedy policy according to action network, but using the target network to estimate the value
        '''
        Q_1_action = self.sequencing_action_NN.forward(sample_s1_batch, m_idx)
        Q_1_target = self.sequencing_target_NN.forward(sample_s1_batch, m_idx)
        #print('Q_1_action is:\n', Q_1_action)
        #print('Q_1_target is:\n', Q_1_target)
        '''
        size of Q_0, Q_1_action and Q_1_target = minibatch size * m_no
        they're 2D tensors
        '''
        max_Q_1_action, max_Q_1_action_idx = torch.max(Q_1_action, dim=1) # use action network to get action, rather than max operation
        #print('max value of Q_1_action is:\n', max_Q_1_action)
        max_Q_1_action_idx = max_Q_1_action_idx.reshape([size,1])
        #print('max idx of Q_1_action is:\n', max_Q_1_action_idx)
        # adjust the max_Q of s_0 by the discount factor (refer to Bellman Equation and TD)
        next_state_value = Q_1_target.gather(1, max_Q_1_action_idx)
        #print('estimated value of next state is:\n', next_state_value)
        next_state_value *= self.discount_factor
        #print('discounted next state value is:\n', next_state_value)
        '''
        the sum of reward and discounted max_Q is the target value
        target value is 2D matrix, size = minibatch_size * m_no
        '''
        #print('reward batch is:', sample_r0_batch)
        target_value = (sample_r0_batch + next_state_value)
        #print('target value is:', target_value)
        '''
        the loss is difference between current state-action value and the target value
        '''
        # calculate the loss
        loss = self.sequencing_action_NN.loss_func(current_value, target_value).detach()
        print('loss is:', loss)
        self.loss_record.append(float(loss))
        # first, clear the gradient (old) of parameters
        self.sequencing_action_NN.optimizer_dict[m_idx].zero_grad()
        # second, calculate gradient (new) of parameters
        loss.backward(retain_graph=True)
        '''
        # check the gradient, to avoid exploding/vanishing gradient, very seldom though
        for param in self.sequencing_action_NN.module_dict[m_idx].parameters():
            print(param.grad.norm())
        '''
        # third, update the parameters
        self.sequencing_action_NN.optimizer_dict[m_idx].step()

    def train_Double_DQN(self):
        """
        draw the random minibatch to train the network
        every element in the replay menory is a list [s_0, a_0, s_1, r_0]
        all element of this list are tensors
        """
        size = min(len(self.rep_memo),self.minibatch_size)
        minibatch = random.sample(self.rep_memo,size)
        '''
        slice, and stack the 1D tensors to several 3D tensors (batch, channel, vector)
        the "torch.stack" is only applicable when the augment is a list of tensors, or multi-dimensional tensor
        '''
        sample_s0_batch = torch.stack([data[0] for data in minibatch], dim=0).reshape([size]+self.input_size_as_list)
        sample_s1_batch = torch.stack([data[2] for data in minibatch], dim=0).reshape([size]+self.input_size_as_list)
        sample_a0_batch = torch.stack([data[1] for data in minibatch], dim=0).reshape(size,1)
        sample_r0_batch = torch.stack([data[3] for data in minibatch], dim=0).reshape(size,1)
        '''
        the size of these batches:
        sample_s0_batch = sample_s1_batch = minibatch size * 1 * input_size
        sample_a0_batch = sample_r0_batch = minibatch size * m_no
        sample_r0_batch = minibatch size
        '''
        # get the Q value (current value of state-action pair) of s0
        Q_0 = self.sequencing_action_NN.forward(sample_s0_batch)
        #print('Q_0 is:\n', Q_0)
        #print('a_0 is:', sample_a0_batch)
        # get the current state-action value of actions that would have been taken
        current_value = Q_0.gather(1, sample_a0_batch)
        #print('current value is:', current_value)
        '''
        get the Q Value of s_1 in both action and target network, to estimate the state value
        architecture is DDQN, NOT DQN !!!
        evaluate the greedy policy according to action network, but using the target network to estimate the value
        '''
        Q_1_action = self.sequencing_action_NN.forward(sample_s1_batch).detach()
        Q_1_target = self.sequencing_target_NN.forward(sample_s1_batch).detach()
        #print('Q_1_action is:\n', Q_1_action)
        #print('Q_1_target is:\n', Q_1_target)
        '''
        size of Q_0, Q_1_action and Q_1_target = minibatch size * m_no
        they're 2D tensors
        '''
        max_Q_1_action, max_Q_1_action_idx = torch.max(Q_1_action, dim=1) # use action network to get action, rather than max operation
        #print('max value of Q_1_action is:\n', max_Q_1_action)
        max_Q_1_action_idx = max_Q_1_action_idx.reshape([size,1])
        #print('max idx of Q_1_action is:\n', max_Q_1_action_idx)
        # adjust the max_Q of s_0 by the discount factor (refer to Bellman Equation and TD)
        next_state_value = Q_1_target.gather(1, max_Q_1_action_idx)
        #print('estimated value of next state is:\n', next_state_value)
        next_state_value *= self.discount_factor
        #print('discounted next state value is:\n', next_state_value)
        '''
        the sum of reward and discounted max_Q is the target value
        target value is 2D matrix, size = minibatch_size * m_no
        '''
        #print('reward batch is:', sample_r0_batch)
        target_value = (sample_r0_batch + next_state_value)
        #print('target value is:', target_value)
        #print('TD error:',target_value - current_value)
        # calculate the loss
        loss = self.sequencing_action_NN.loss_func(current_value, target_value).detach()
        self.loss_time_record.append(self.env.now)
        self.loss_record.append(float(loss))
        if not self.env.now%50:
            print('Time: %s, loss: %s:'%(self.env.now, loss))
        # first, clear the gradient (old) of parameters
        self.sequencing_action_NN.optimizer.zero_grad()
        # second, calculate gradient (new) of parameters
        loss.backward(retain_graph=True)
        '''
        # check the gradient, to avoid exploding/vanishing gradient, very seldom though
        for param in self.sequencing_action_NN.module_dict[m_idx].parameters():
            print(param.grad.norm())
        '''
        # third, update the parameters
        self.sequencing_action_NN.optimizer.step()

    def train_validated(self):
        """
        draw the random minibatch to train the network
        every element in the replay menory is a list [s_0, a_0, s_1, r_0]
        all element of this list are tensors
        """
        size = min(len(self.rep_memo),self.minibatch_size)
        minibatch = random.sample(self.rep_memo,size)
        '''
        slice, and stack the 1D tensors to several 3D tensors (batch, channel, vector)
        the "torch.stack" is only applicable when the augment is a list of tensors, or multi-dimensional tensor
        '''
        sample_s0_batch = torch.stack([data[0] for data in minibatch], dim=0).reshape(size,1,self.input_size)
        sample_s1_batch = torch.stack([data[2] for data in minibatch], dim=0).reshape(size,1,self.input_size)
        sample_a0_batch = torch.stack([data[1] for data in minibatch], dim=0).reshape(size,1)
        sample_r0_batch = torch.stack([data[3] for data in minibatch], dim=0).reshape(size,1)
        '''
        the size of these batches:
        sample_s0_batch = sample_s1_batch = minibatch size * 1 * input_size
        sample_a0_batch = sample_r0_batch = minibatch size * m_no
        sample_r0_batch = minibatch size
        '''
        # get the Q value (current value of state-action pair) of s0
        Q_0 = self.sequencing_action_NN.forward(sample_s0_batch)
        #print('Q_0 is:\n', Q_0)
        #print('a_0 is:', sample_a0_batch)
        # get the current state-action value of actions that would have been taken
        current_value = Q_0.gather(1, sample_a0_batch)
        #print('current value is:', current_value)
        '''
        get the Q Value of s_1 in both action and target network, to estimate the state value
        architecture is DDQN, NOT DQN !!!
        evaluate the greedy policy according to action network, but using the target network to estimate the value
        '''
        Q_1_action = self.sequencing_action_NN.forward(sample_s1_batch).detach()
        Q_1_target = self.sequencing_target_NN.forward(sample_s1_batch).detach()
        #print('Q_1_action is:\n', Q_1_action)
        #print('Q_1_target is:\n', Q_1_target)
        '''
        size of Q_0, Q_1_action and Q_1_target = minibatch size * m_no
        they're 2D tensors
        '''
        max_Q_1_action, max_Q_1_action_idx = torch.max(Q_1_action, dim=1) # use action network to get action, rather than max operation
        #print('max value of Q_1_action is:\n', max_Q_1_action)
        max_Q_1_action_idx = max_Q_1_action_idx.reshape([size,1])
        #print('max idx of Q_1_action is:\n', max_Q_1_action_idx)
        # adjust the max_Q of s_0 by the discount factor (refer to Bellman Equation and TD)
        next_state_value = Q_1_target.gather(1, max_Q_1_action_idx)
        #print('estimated value of next state is:\n', next_state_value)
        next_state_value *= self.discount_factor
        #print('discounted next state value is:\n', next_state_value)
        '''
        the sum of reward and discounted max_Q is the target value
        target value is 2D matrix, size = minibatch_size * m_no
        '''
        #print('reward batch is:', sample_r0_batch)
        target_value = (sample_r0_batch + next_state_value)
        #print('target value is:', target_value)
        #print('TD error:',target_value - current_value)
        # calculate the loss
        loss = self.sequencing_action_NN.loss_func(current_value, target_value)
        self.loss_time_record.append(self.env.now)
        self.loss_record.append(float(loss))
        if not self.env.now%50:
            print('Time: %s, loss: %s:'%(self.env.now, loss))
        # first, clear the gradient (old) of parameters
        self.sequencing_action_NN.optimizer.zero_grad()
        # second, calculate gradient (new) of parameters
        loss.backward(retain_graph=True)
        '''
        # check the gradient, to avoid exploding/vanishing gradient, very seldom though
        for param in self.sequencing_action_NN.module_dict[m_idx].parameters():
            print(param.grad.norm())
        '''
        # third, update the parameters
        self.sequencing_action_NN.optimizer.step()

'''
classes of ANN (along with optimzers), total 3 types:
1. normalization (normalize data of same type in one channel)
2. abstract (use the reciprocal of input data, no normalization layers)
3. multi-channel (normalize data in multiple channels)
'''


# normalize data of same type in one channel
class network_normalization(nn.Module):
    def __init__(self, input_size, output_size, number_size, time_size, target_m_list):
        super(network_normalization, self).__init__()
        # first get the list of machine index for all target machines
        m_idx_list = [m.m_idx for m in target_m_list]
        print("*** Initialize policy network (Normalization mode) for target machines ***")
        print("*** Target machines' indexes are: {} ***".format(m_idx_list))
        self.input_size = input_size
        self.output_size = output_size
        self.norm_number = number_size
        self.norm_time = time_size + number_size
        # FCNN parameters
        layer_1 = input_size
        layer_2 = input_size*3
        layer_3 = input_size*2
        layer_4 = input_size
        layer_5 = layer_4
        layer_6 = layer_5
        layer_7 = layer_6
        layer_8 = layer_7
        # normalization modules
        self.normlayer_number = nn.Sequential(
                                nn.InstanceNorm1d(self.norm_number),
                                nn.Flatten()
                                )
        self.normlayer_time = nn.Sequential(
                                nn.InstanceNorm1d(self.norm_time),
                                nn.Flatten()
                                )
        # shared layers of machines
        self.subsequent_module = nn.Sequential(
                                nn.Linear(self.input_size, layer_1),
                                nn.Tanh(),
                                nn.Linear(layer_1, layer_2),
                                nn.Tanh(),
                                nn.Linear(layer_2, layer_3),
                                nn.Tanh(),
                                nn.Linear(layer_3, layer_4),
                                nn.Tanh(),
                                nn.Linear(layer_4, layer_5),
                                nn.Tanh(),
                                nn.Linear(layer_5, layer_6),
                                nn.Tanh(),
                                nn.Linear(layer_6, output_size)
                                )
        # Huber loss function
        self.loss_func = F.smooth_l1_loss
        # the dictionary that stores ModuleList for each sequencing agent
        self.module_dict = {}
        self.optimizer_dict = {}
        for i in m_idx_list:
            # for each agent, its module list contains shared and independent layers
            self.module_dict[i] = nn.ModuleList(\
            [self.normlayer_number, self.normlayer_time, self.subsequent_module])
            # accompanied by an independent optimizer
            self.optimizer_dict[i] = optim.Adam(self.module_dict[i].parameters(), lr=0.01)
        #print("All modules:\n",self.module_dict)
        #print("All optimizers:\n",self.optimizer_dict)

    def forward(self, x, idx):
        #print('original',x)
        # slice the data
        x_norm_number = x[:,:,:self.norm_number]
        x_norm_time = x[:,:,self.norm_number:self.norm_time]
        x_rest = x[:,:,self.norm_time:].squeeze(1)
        #print(x_norm_number)
        #print(x_norm_time)
        #print(x_rest)
        # normalize data in two 'channels'
        x_normed_number = self.module_dict[idx][0](x_norm_number)
        x_normed_time = self.module_dict[idx][1](x_norm_time)
        #print('normalized',x_normed_number)
        #print('normalized',x_normed_time)
        # concatenate normalized data and scaling factor
        x = torch.cat([x_normed_number, x_normed_time, x_rest], dim=1)
        #print('combined',x)
        # the last, independent part of module
        x = self.module_dict[idx][2](x)
        #print('output',x)
        return x

# normalize data in multiple channels
class network_multi_channel(nn.Module):
    def __init__(self, input_size, output_size, target_m_list):
        super(network_multi_channel, self).__init__()
        # first get the list of machine index for all target machines
        m_idx_list = [m.m_idx for m in target_m_list]
        print("*** Initialize policy network (Normalization mode) for target machines ***")
        print("*** Target machines' indexes are: {} ***".format(m_idx_list))
        self.lr = 0.001
        self.input_size = input_size
        self.output_size = output_size
        # for slicing the data
        self.no_size = 3
        self.pt_size = 6
        self.remaining_pt_size = 11
        self.ttd_slack_size = 16
        # FCNN parameters
        layer_1 = 48
        layer_2 = 36
        layer_3 = 36
        layer_4 = 24
        layer_5 = 24
        layer_6 = 12
        # normalization modules
        self.normlayer_no = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_pt = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_remaining_pt = nn.Sequential(
                                nn.InstanceNorm1d(5),
                                nn.Flatten()
                                )
        self.normlayer_ttd_slack = nn.Sequential(
                                nn.InstanceNorm1d(5),
                                nn.Flatten()
                                )
        # shared layers of machines
        self.subsequent_module = nn.Sequential(
                                nn.Linear(self.input_size, layer_1),
                                nn.Tanh(),
                                nn.Linear(layer_1, layer_2),
                                nn.Tanh(),
                                nn.Linear(layer_2, layer_3),
                                nn.Tanh(),
                                nn.Linear(layer_3, layer_4),
                                nn.Tanh(),
                                nn.Linear(layer_4, layer_5),
                                nn.Tanh(),
                                nn.Linear(layer_5, layer_6),
                                nn.Tanh(),
                                nn.Linear(layer_6, output_size)
                                )
        # Huber loss function
        self.loss_func = F.smooth_l1_loss
        # the dictionary that stores ModuleList for each sequencing agent
        self.module_dict = {}
        self.optimizer_dict = {}
        for i in m_idx_list:
            # for each agent, its module list contains shared and independent layers
            self.module_dict[i] = nn.ModuleList(\
            [self.normlayer_no, self.normlayer_pt, self.normlayer_remaining_pt, self.normlayer_ttd_slack, self.subsequent_module])
            # accompanied by an independent optimizer
            self.optimizer_dict[i] = optim.SGD(self.module_dict[i].parameters(), lr=self.lr, momentum = 0.9)
        #print("All modules:\n",self.module_dict)
        #print("All optimizers:\n",self.optimizer_dict)

    def forward(self, x, idx):
        #print('original',x)
        # slice the data
        x_no = x[:,:, : self.no_size]
        x_pt = x[:,:, self.no_size : self.pt_size]
        x_remaining_pt = x[:,:, self.pt_size : self.remaining_pt_size]
        x_ttd_slack = x[:,:, self.remaining_pt_size : self.ttd_slack_size]
        x_rest = x[:,:, self.ttd_slack_size :].squeeze(1)
        # normalize data in multiple channels
        x_normed_no = self.module_dict[idx][0](x_no)
        x_normed_pt = self.module_dict[idx][1](x_pt)
        x_normed_remaining_pt = self.module_dict[idx][2](x_remaining_pt)
        x_normed_ttd_slack = self.module_dict[idx][3](x_ttd_slack)
        #print('normalized',x_normed_no)
        # concatenate all data
        #print(x_normed_no, x_normed_pt, x_normed_remaining_pt, x_normed_ttd, x_normed_slack, x_rest)
        x = torch.cat([x_normed_no, x_normed_pt, x_normed_remaining_pt, x_normed_ttd_slack, x_rest], dim=1)
        #print('combined',x)
        # the last, independent part of module
        x = self.module_dict[idx][4](x)
        #print('output',x)
        return x

class network_value_based(nn.Module):
    def __init__(self, input_size, output_size):
        super(network_value_based, self).__init__()
        self.lr = 0.001
        self.input_size = input_size
        self.output_size = output_size
        # for slicing the data
        self.no_size = 3
        self.pt_size = 6
        self.remaining_pt_size = 11
        self.ttd_slack_size = 16
        # FCNN parameters
        layer_1 = 48
        layer_2 = 36
        layer_3 = 36
        layer_4 = 24
        layer_5 = 24
        layer_6 = 12
        # normalization modules
        self.normlayer_no = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_pt = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_remaining_pt = nn.Sequential(
                                nn.InstanceNorm1d(5),
                                nn.Flatten()
                                )
        self.normlayer_ttd_slack = nn.Sequential(
                                nn.InstanceNorm1d(5),
                                nn.Flatten()
                                )
        # shared layers of machines
        self.subsequent_module = nn.Sequential(
                                nn.Linear(self.input_size, layer_1),
                                nn.Tanh(),
                                nn.Linear(layer_1, layer_2),
                                nn.Tanh(),
                                nn.Linear(layer_2, layer_3),
                                nn.Tanh(),
                                nn.Linear(layer_3, layer_4),
                                nn.Tanh(),
                                nn.Linear(layer_4, layer_5),
                                nn.Tanh(),
                                nn.Linear(layer_5, layer_6),
                                nn.Tanh(),
                                nn.Linear(layer_6, output_size)
                                )
        # Huber loss function
        self.loss_func = F.smooth_l1_loss
        # the universal network for all scheudling agents
        self.network = nn.ModuleList([self.normlayer_no, self.normlayer_pt, self.normlayer_remaining_pt, self.normlayer_ttd_slack, self.subsequent_module])
        # accompanied by optimizer
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum = 0.9)

    def forward(self, x, *args):
        #print('original',x)
        # slice the data
        x_no = x[:,:, : self.no_size]
        x_pt = x[:,:, self.no_size : self.pt_size]
        x_remaining_pt = x[:,:, self.pt_size : self.remaining_pt_size]
        x_ttd_slack = x[:,:, self.remaining_pt_size : self.ttd_slack_size]
        x_rest = x[:,:, self.ttd_slack_size :].squeeze(1)
        # normalize data in multiple channels
        x_normed_no = self.network[0](x_no)
        x_normed_pt = self.network[1](x_pt)
        x_normed_remaining_pt = self.network[2](x_remaining_pt)
        x_normed_ttd_slack = self.network[3](x_ttd_slack)
        #print('normalized',x_normed_no)
        # concatenate all data
        #print(x_normed_no, x_normed_pt, x_normed_remaining_pt, x_normed_ttd, x_normed_slack, x_rest)
        x = torch.cat([x_normed_no, x_normed_pt, x_normed_remaining_pt, x_normed_ttd_slack, x_rest], dim=1)
        #print('combined',x)
        # the last, independent part of module
        x = self.network[4](x)
        #print('output',x)
        return x

class network_validated(nn.Module):
    def __init__(self, input_size, output_size):
        super(network_validated, self).__init__()
        self.lr = 0.001
        self.input_size = input_size
        self.output_size = output_size
        # for slicing the data
        self.no_size = 3
        self.pt_size = 6
        self.remaining_pt_size = 11
        self.ttd_slack_size = 16
        # FCNN parameters
        layer_1 = 48
        layer_2 = 36
        layer_3 = 36
        layer_4 = 24
        layer_5 = 24
        layer_6 = 12
        # normalization modules
        self.normlayer_no = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_pt = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_remaining_pt = nn.Sequential(
                                nn.InstanceNorm1d(5),
                                nn.Flatten()
                                )
        self.normlayer_ttd_slack = nn.Sequential(
                                nn.InstanceNorm1d(5),
                                nn.Flatten()
                                )
        # shared layers of machines
        self.subsequent_module = nn.Sequential(
                                nn.Linear(self.input_size, layer_1),
                                nn.Tanh(),
                                nn.Linear(layer_1, layer_2),
                                nn.Tanh(),
                                nn.Linear(layer_2, layer_3),
                                nn.Tanh(),
                                nn.Linear(layer_3, layer_4),
                                nn.Tanh(),
                                nn.Linear(layer_4, layer_5),
                                nn.Tanh(),
                                nn.Linear(layer_5, layer_6),
                                nn.Tanh(),
                                nn.Linear(layer_6, output_size)
                                )
        # Huber loss function
        self.loss_func = F.smooth_l1_loss
        # the universal network for all scheudling agents
        self.network = nn.ModuleList([self.normlayer_no, self.normlayer_pt, self.normlayer_remaining_pt, self.normlayer_ttd_slack, self.subsequent_module])
        # accompanied by optimizer
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum = 0.9)

    def forward(self, x, *args):
        #print('original',x)
        # slice the data
        x_no = x[:,:, : self.no_size]
        x_pt = x[:,:, self.no_size : self.pt_size]
        x_remaining_pt = x[:,:, self.pt_size : self.remaining_pt_size]
        x_ttd_slack = x[:,:, self.remaining_pt_size : self.ttd_slack_size]
        x_rest = x[:,:, self.ttd_slack_size :].squeeze(1)
        # normalize data in multiple channels
        x_normed_no = self.network[0](x_no)
        x_normed_pt = self.network[1](x_pt)
        x_normed_remaining_pt = self.network[2](x_remaining_pt)
        x_normed_ttd_slack = self.network[3](x_ttd_slack)
        #print('normalized',x_normed_no)
        # concatenate all data
        #print(x_normed_no, x_normed_pt, x_normed_remaining_pt, x_normed_ttd, x_normed_slack, x_rest)
        x = torch.cat([x_normed_no, x_normed_pt, x_normed_remaining_pt, x_normed_ttd_slack, x_rest], dim=1)
        #print('combined',x)
        # the last, independent part of module
        x = self.network[4](x)
        #print('output',x)
        return x

class network_value_based(nn.Module):
    def __init__(self, input_size, output_size):
        super(network_value_based, self).__init__()
        self.lr = 0.005
        self.input_size = input_size
        self.output_size = output_size
        self.flattened_input_size = torch.tensor(self.input_size).prod()
        # FCNN parameters
        layer_1 = 64
        layer_2 = 48
        layer_3 = 48
        layer_4 = 36
        layer_5 = 24
        layer_6 = 12
        # normalization modules
        self.norm_layer = nn.Sequential(
                                nn.LayerNorm(self.input_size),
                                nn.Flatten()
                                )
        # shared layers of machines
        self.FC_layers = nn.Sequential(
                                nn.Linear(self.flattened_input_size, layer_1),
                                nn.Tanh(),
                                nn.Linear(layer_1, layer_2),
                                nn.Tanh(),
                                nn.Linear(layer_2, layer_3),
                                nn.Tanh(),
                                nn.Linear(layer_3, layer_4),
                                nn.Tanh(),
                                nn.Linear(layer_4, layer_5),
                                nn.Tanh(),
                                nn.Linear(layer_5, layer_6),
                                nn.Tanh(),
                                nn.Linear(layer_6, output_size)
                                )
        # Huber loss function
        self.loss_func = F.smooth_l1_loss
        # the universal network for all scheudling agents
        self.network = nn.ModuleList([self.norm_layer, self.FC_layers])
        # accompanied by optimizer
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum = 0.9)

    def forward(self, x, *args):
        #print('original',x)
        x = self.network[0](x)
        x = self.network[1](x)
        #print('output',x)
        return x

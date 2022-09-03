import simpy
import sys
sys.path
import random
import numpy as np
import torch
from tabulate import tabulate
import sequencing
import routing

class machine:
    def __init__(self, env, index, *args, **kwargs):
        # initialize the environment of simulation
        self.env = env
        self.m_idx = index

        # each machine will have an independent storage for each type of job information
        # initialize all job-related information storage as empty lists
        self.queue = []
        self.sequence_list = [] # sequence of all queuing jobs
        self.pt_list = [] # processing time
        self.remaining_pt_list = [] # average processing time
        self.due_list = [] # due for each job
        self.arrival_time_list = [] # time that job join the queue
        self.waited_time = [] # time that job stayed in the queue
        self.slack_upon_arrival = [] # slack record of queuing jobs
        self.no_jobs_record = []
        # the time that agent do current and next decision
        self.decision_point = 0
        self.release_time = 0
        # track the utilization
        self.cumulative_run_time = 0
        self.global_exp_tard_rate = 0

        # Initialize the possible events during production
        self.sufficient_stock = self.env.event()
        # working condition in shut down and breakdowns
        self.working_event = self.env.event()
        # this is the time that machine needs to recover from breakdown
        # initial value is 0, later will be changed by "breakdown_creation" module
        self.restart_time = 0
        self.count = 0
        self.count2 = 0
        # Initialize the events'states
        # if the queue is not empty
        if not len(self.queue):
            self.sufficient_stock.succeed()
        # no shutdown, no breakdown at beginning
        self.working_event.succeed()
        # print out the information of initial jobs
        self.print_info = True
        self.routing_global_reward = False
        # initialize the data for learning and recordiing
        self.breakdown_record = []
        # use exponential moving average to measure slack and tardiness
        self.EMA_slack_change = 0
        self.EMA_realized_tardiness = 0
        self.EMA_alpha = 0.1
        # set the sequencing rule before start of simulation
        if 'rule' in kwargs:
            order = "self.job_sequencing = sequencing." + kwargs['rule']
            try:
                exec(order)
                print("machine {} uses {} sequencing rule".format(self.m_idx, kwargs['rule']))
            except:
                print("Rule assigned to machine {} is invalid !".format(self.m_idx))
                raise Exception
        else:
            # default sequencing rule is FIFO
            self.job_sequencing = sequencing.FIFO
        # record extra data for learning, initially not activated, can be activated by brains
        self.sequencing_learning_event = self.env.event()
        self.routing_learning_event = self.env.event()


    '''
    1. downwards are functions that perform the simulation
       including production, starvation and breakdown
    '''


    # this function should be called after __init__ to avoid deadlock
    # after the creation of all machines, initial jobs and work centers
    # pass the list of work centers to all machines so the shopfloor is established
    # the initial jobs are allocated through job_creation module
    def initialization(self, machine_list, workcenter_list, job_creator, assigned_wc):
        # knowing other machines, workcenters, and the job creator
        # so the machine agent can manipulate other agents'variables
        self.m_list = machine_list
        self.m_no = len(self.m_list)
        self.wc_list = workcenter_list
        self.wc = assigned_wc
        self.wc_idx = assigned_wc.wc_idx
        self.no_ops = len(self.wc_list)
        self.job_creator = job_creator
        # initial information
        if self.print_info:
            print('machine {} belongs to work center {}'.format(self.m_idx,assigned_wc.wc_idx))
            print('Initial %s jobs at machine %s are:'%(len(self.queue), self.m_idx))
            job_info = [[self.queue[i],self.sequence_list[i], self.pt_list[i], self.slack_upon_arrival[i], self.due_list[i]] for i in range(len(self.queue))]
            print(tabulate(job_info, headers=['idx.','sqc.','proc.t.','slack','due']))
            print('************************************')
        self.state_update_all()
        self.update_global_info_progression()
        self.env.process(self.production())

    # The main function, simulates the production
    def production(self):
        # first check the initial queue/stock level, if none, starvation begines
        if not len(self.queue):
            # triggered the starvation
            yield self.env.process(self.starvation())
        # update information of queuing jobs at the end of initial phase
        self.state_update_all()
        # the loop that will run till the ned of simulation
        while True:
            # record the time of the sequencing decision (select a job to process), used as the index of produciton record in job creator
            self.decision_point = self.env.now
            self.no_jobs_record.append(len(self.queue))
            # if we have more than one queuing jobs, sequencing is required
            if len(self.queue)-1:
                # determine the next job to be processed
                # the returned value is selected job's self.position in queue
                self.position = self.job_sequencing(self.sequencing_data_generation())
                self.job_idx = self.queue[self.position]
                self.before_operation()
                self.count += 1
                if len(self.queue)-2:
                    self.count2 += 1
                #print("Sequencing: Machine %s choose job %s at time %s"%(self.m_idx,self.job_idx,self.env.now))
            # otherwise simply select the first(only) one
            else:
                self.position = 0
                self.job_idx = self.queue[self.position]
                #print("One queue: Machine %s process job %s at time %s"%(self.m_idx,self.job_idx,self.env.now))
            # retrive the information of job
            pt = self.pt_list[self.position][self.m_idx] # processing time of the selected job
            wait = self.env.now - self.arrival_time_list[self.position] # time that job waited before being selected
            # after determined the next job to be processed, update a bunch of data
            self.update_global_info_progression()
            self.update_global_info_anticipation(pt)
            self.record_production(pt, wait) # record these information
            # The production process (yield the processing time of operation)
            yield self.env.timeout(pt)
            self.cumulative_run_time += pt
            #print("completion: Job %s leave machine %s at time %s"%(self.queue[self.position],self.m_idx,self.env.now))
            # transfer job to next workcenter or delete it, and update information
            self.after_operation()
            # check if routing learning mode is on, if yes, call the function of WORKCENTER, NOT ITSELF!!!
            # examine whether the scheduled shutdown is triggered
            if not self.working_event.triggered:
                yield self.env.process(self.breakdown())
                # after restart, update information of queuing jobs
                self.state_update_all()
            # check the queue/stock level, if none, starvation begines
            if not len(self.queue):
                # triggered the starvation
                yield self.env.process(self.starvation())
                # after replenishement, update information of queuing jobs
                self.state_update_all()

    def starvation(self):
        #print('STARVATION *BEGIN*: machine %s at time %s' %(self.m_idx, self.env.now))
        # set the self.sufficient_stock event to untriggered
        self.sufficient_stock = self.env.event()
        # proceed only if the sufficient_stock event is triggered by new job arrival
        yield self.sufficient_stock
        # examine whether the scheduled shutdown is triggered
        if not self.working_event.triggered:
            yield self.env.process(self.breakdown())
        #print('STARVATION *END*: machine %s at time: %s'%(self.m_idx, self.env.now))

    def breakdown(self):
        print('********', self.m_idx, "breakdown at time", self.env.now, '********')
        start = self.env.now
        # simply update the available time of that machines
        self.available_time = self.restart_time + self.cumulative_pt
        # suspend the production here, untill the working_event is triggered
        yield self.working_event
        self.breakdown_record.append([(start, self.env.now-start), self.m_idx])
        print('********', self.m_idx, 'brekdown ended, restart production at time', self.env.now, '********')


    '''
    2. downwards are functions the called before and after each operation
       to maintain some record, and transit the finished job to next workcenter or out of system
    '''


    # update lots information that will be used for calculating the rewards
    def before_operation(self):
        # number of jobs that to be sequenced, and their ttd and slack
        self.waiting_jobs = len(self.queue)
        time_till_due = np.array(self.due_list) - self.env.now
        self.before_op_ttd = time_till_due
        self.before_op_ttd_chosen = self.before_op_ttd[self.position]
        self.before_op_ttd_loser = np.delete(self.before_op_ttd, self.position)
        tardy_jobs = len(time_till_due[time_till_due<0])
        #self.before_op_realized_tard_rate =tardy_jobs/len(self.queue)
        #print('before realized tard rate: ', self.before_op_realized_tard_rate)
        initial_slack = self.slack_upon_arrival.copy()
        self.before_op_remaining_pt = self.remaining_job_pt + self.current_pt
        self.before_op_remaining_pt_chosen = self.before_op_remaining_pt[self.position]
        self.before_op_remaining_pt_loser = np.delete(self.before_op_remaining_pt, self.position)
        current_slack = time_till_due - self.before_op_remaining_pt
        exp_tardy_jobs = len(current_slack[current_slack<0])
        # get information of all jobs before operation
        self.before_op_exp_tard = current_slack[current_slack<0]
        self.before_op_sum_exp_tard = self.before_op_exp_tard.sum()
        self.before_op_slack = current_slack
        self.before_op_sum_slack = self.before_op_slack.sum()
        # calculate the critical level  of all queuing jobs
        self.critical_level = 1 - current_slack / 100
        self.critical_level_chosen  = self.critical_level[self.position]
        #print(current_slack, self.critical_level,self.critical_level_chosen)
        # get the information of the selected job
        self.pt_chosen = self.current_pt[self.position]
        self.initial_slack_chosen = initial_slack[self.position]
        self.before_op_slack_chosen = current_slack[self.position]
        self.before_op_exp_tard_chosen = min(0,self.before_op_slack_chosen)
        self.before_op_winq_chosen = self.winq[self.position]
        # get the information of jobs that haven't been selected (loser)
        self.before_op_slack_loser = np.delete(current_slack, self.position) # those haven't been selected
        self.critical_level_loser = np.delete(self.critical_level, self.position)
        self.before_op_sum_exp_tard_loser = self.before_op_slack_loser[self.before_op_slack_loser<0].sum()
        self.before_op_sum_slack_loser = self.before_op_slack_loser.sum()
        self.before_op_winq_loser = np.delete(self.winq, self.position)
        #print('before',self.m_idx,self.env.now,slack,slack_loser,self.before_op_exp_tard,self.current_pt,self.position)
        #self.before_op_avg_slack = slack.sum()/len(self.queue)
        #self.before_op_expected_tard_rate = exp_tardy_jobs/len(self.queue)
        #print('before expected tard rate: ', self.before_op_expected_tard_rate)

    # transfer unfinished job to next workcenter, or delete finished job from record
    # and update the data of queuing jobs, EMA_tardiness etc.
    def after_operation(self):
        # check if this is the last operation of job
        # if the sequence is not empty, any value > 0 is True
        if len(self.sequence_list[self.position]):
            #print('OPERATION: Job %s output from machine %s at time %s'%(self.queue[self.position], self.m_idx, self.env.now))
            next_wc = self.sequence_list[self.position][0]
            # add the job to next work center's queue
            self.wc_list[next_wc].queue.append(self.queue.pop(self.position))
            # add the information of this job to next work center's storage
            self.wc_list[next_wc].sequence_list.append(np.delete(self.sequence_list.pop(self.position),0))
            self.wc_list[next_wc].pt_list.append(self.pt_list.pop(self.position))
            # get the expected processing time of remaining processes
            remaining_ptl = self.remaining_pt_list.pop(self.position)
            self.wc_list[next_wc].remaining_pt_list.append(remaining_ptl)
            # get old and current_slack time of the job, meanwhile add due to next wc's storage
            current_slack = self.due_list[self.position] - self.env.now - np.sum(remaining_ptl.max(axis=1))
            self.wc_list[next_wc].due_list.append(self.due_list.pop(self.position))
            estimated_slack_time = self.slack_upon_arrival.pop(self.position)
            del self.arrival_time_list[self.position]
            # calculate slack gain/loss
            self.slack_change = current_slack - estimated_slack_time
            self.critical_level_R = 1 - estimated_slack_time / 100
            # record the slack change
            self.record_slack_tardiness()
            # calculate the EMA_slack_change
            self.EMA_slack_change += self.EMA_alpha * (self.slack_change - self.EMA_slack_change)
            # and activate the dispatching of next work center
            try:
                self.wc_list[next_wc].routing_event.succeed()
            except:
                pass
            # after transfered the job, update information of queuing jobs
            self.state_update_all()
            # clear some global information
            self.update_global_info_after_operation()
            # check if sequencing learning mode is on, and queue is not 0
            if self.routing_learning_event.triggered:
                try:
                    self.wc.build_routing_experience(self.job_idx,self.slack_change, self.critical_level_R)
                except:
                    pass
            if self.sequencing_learning_event.triggered:
                self.complete_experience()
        # if this is the last process, then simply delete job information
        else:
            #print('**FINISHED: Job %s from machine %s at time %s'%(self.queue[self.position], self.m_idx, self.env.now))
            # calculate tardiness of job, and update EMA_realized_tardiness
            self.tardiness = np.max([0, self.env.now - self.due_list[self.position]])
            #print("realized tardiness is:", tardiness)
            self.EMA_realized_tardiness += self.EMA_alpha * (self.tardiness - self.EMA_realized_tardiness)
            #print(self.m_idx,self.EMA_realized_tardiness)
            # delete this job from queue
            del self.queue[self.position]
            # delete the information of this job
            del self.sequence_list[self.position]
            del self.pt_list[self.position]
            del self.remaining_pt_list[self.position]
            # get old and current_slack time of the job
            current_slack = self.due_list[self.position] - self.env.now # there's no more operations for this job
            del self.due_list[self.position]
            estimated_slack_time = self.slack_upon_arrival.pop(self.position)
            del self.arrival_time_list[self.position]
            # kick the job out of system
            self.job_creator.record_job_departure()
            #print(self.job_creator.in_system_job_no)
            # calculate slack gain/loss
            self.slack_change = current_slack - estimated_slack_time
            self.critical_level_R = 1 - estimated_slack_time / 100
            #print(current_slack, estimated_slack_time, self.critical_level_R)
            # record the slack change
            self.record_slack_tardiness(self.tardiness)
            #print("estimated_slack_time: %s / current_slack: %s"%(estimated_slack_time, current_slack))
            # calculate the EMA_slack_change
            self.EMA_slack_change += self.EMA_alpha * (self.slack_change - self.EMA_slack_change)
            # after transfered the job, update information of queuing jobs
            self.state_update_all()
            # clear some global information
            self.update_global_info_after_operation()
            # check if sequencing learning mode is on, and queue is not 0
            # if yes, since the job is finished and tardiness is realized, construct complete experience
            if self.routing_learning_event.triggered:
                try:
                    self.wc.build_routing_experience(self.job_idx,self.slack_change, self.critical_level_R)
                except:
                    pass
            if self.sequencing_learning_event.triggered:
                self.complete_experience()
            if self.routing_global_reward:
                self.add_global_reward_RA()

    '''
    3. downwards are functions that related to information update and exchange
       especially the information that will be used by other agents on shop floor
    '''


    def record_production(self, pt, wait):
        # add the details of operation to job_creator's repository
        self.job_creator.production_record[self.job_idx][0].append((self.env.now,pt))
        self.job_creator.production_record[self.job_idx][1].append(self.m_idx)
        self.job_creator.production_record[self.job_idx][2].append(wait)

    def record_slack_tardiness(self, *args):
        self.job_creator.production_record[self.job_idx][4].append(self.slack_change)
        if len(args):
            self.job_creator.production_record[self.job_idx].append((self.env.now,args[0]))

    # call this function after the completion of operation
    def state_update_all(self):
        # processing time of current process of each queuing job
        self.current_pt = np.array([x[self.m_idx] for x in self.pt_list])
        # cumultive processing time of all queuing jobs on this machine
        self.cumulative_pt = self.current_pt.sum()
        # the time the machine will be available (become idle or breakdown ends)
        self.available_time = self.env.now + self.cumulative_pt
        # expected cumulative processing time (worst possible) of all unfinished processes for each queuing job
        self.remaining_job_pt = np.array([sum(x.mean(axis=1)) for x in self.remaining_pt_list])
        self.remaining_no_op = np.array([len(x) for x in self.remaining_pt_list])
        self.next_pt = np.array([x[0].mean() if len(x) else 0 for x in self.remaining_pt_list])
        # the completion rate of all queuing jobs
        self.completion_rate = np.array([(self.no_ops-len(x)-1)/self.no_ops for x in self.remaining_pt_list])
        # number of queuing jobs
        self.que_size = len(self.queue)
        # time till due and slack time of jobs
        self.time_till_due = np.array(self.due_list) - self.env.now
        self.slack = self.time_till_due - self.current_pt - self.remaining_job_pt
        # time that job spent in the queue
        self.waited_time = self.env.now - np.array(self.arrival_time_list)
        # WINQ
        self.winq = np.array([self.wc_list[x[0]].average_workcontent if len(x) else 0 for x in self.sequence_list])
        self.avlm = np.array([self.wc_list[x[0]].average_waiting if len(x) else 0 for x in self.sequence_list])
        #print(self.sequence_list, self.winq)

    # available timeis a bit tricky, jobs may come when the operation is ongoing
    # or when the machine is already in starvation (availble time is earlier than now)
    # hence we can't simply let available time = now + cumulative_pt
    def state_update_after_job_arrival(self, increased_available_time):
        self.current_pt = np.array([x[self.m_idx] for x in self.pt_list])
        self.cumulative_pt = self.current_pt.sum()
        # add the new job's pt to current time / current available time
        self.available_time = max(self.available_time, self.env.now) + increased_available_time
        self.que_size = len(self.queue)

    # update the information of progression, eralized and expected tardiness to JOB_CREATOR !!!
    def update_global_info_progression(self):
        # realized: 0 if already tardy; exp: 0 is slack time is negative
        realized = self.time_till_due.clip(0,1)
        exp = self.slack.clip(0,1)
        # update the machine's corresponding record in job creator, and several rates
        self.job_creator.comp_rate_list[self.m_idx] = self.completion_rate
        self.job_creator.comp_rate = np.concatenate(self.job_creator.comp_rate_list).mean()
        self.job_creator.realized_tard_list[self.m_idx] = realized
        self.job_creator.realized_tard_rate = 1 - np.concatenate(self.job_creator.realized_tard_list).mean()
        self.job_creator.exp_tard_list[self.m_idx] = exp
        self.job_creator.exp_tard_rate = 1 - np.concatenate(self.job_creator.exp_tard_list).mean()
        self.job_creator.available_time_list[self.m_idx] = self.available_time

    # update the information of the job that being processed to JOB_CREATOR !!!
    def update_global_info_anticipation(self,pt):
        current_j_idx = self.queue[self.position]
        self.job_creator.current_j_idx_list[self.m_idx] = current_j_idx
        next_wc = self.sequence_list[self.position][0] if len(self.sequence_list[self.position]) else -1 # next workcenter of the job
        self.job_creator.next_wc_list[self.m_idx] = next_wc # update the next wc info (hold by job creator)
        self.release_time = self.env.now + pt
        self.job_creator.release_time_list[self.m_idx] = self.release_time # update the time of completion of current operation
        job_rempt = self.remaining_job_pt[self.position].sum() - pt
        self.job_creator.arriving_job_rempt_list[self.m_idx] = job_rempt # update the remaining pt of job under processing
        job_slack = self.slack[self.position]
        self.job_creator.arriving_job_slack_list[self.m_idx] = job_slack # update the slack time of processing job (hold by job creator)

    # must call this after operation otherwise the record persists, lead to error
    def update_global_info_after_operation(self):
        self.job_creator.next_wc_list[self.m_idx] = -1 # after each operation, clear the record in job creator

    # give out the information related to routing decision
    def routing_data_generation(self):
        # note that we subtract current time from available_time
        # becasue state_update_all function may be called at a different time
        self.routing_data = [self.cumulative_pt, max(0,self.available_time-self.env.now), self.que_size, self.cumulative_run_time]
        return self.routing_data

    # give ou the information related to sequencing decision
    def sequencing_data_generation(self):
        self.sequencing_data = \
        [self.current_pt, self.remaining_job_pt, np.array(self.due_list), self.env.now, self.completion_rate, \
        self.time_till_due, self.slack, self.winq, self.avlm, self.next_pt, self.remaining_no_op, self.waited_time, \
        self.wc_idx, self.queue, self.m_idx]
        #print(self.sequencing_data)
        return self.sequencing_data


    '''
    4. downwards are functions related to the calculation of reward and construction of state
       only be called if the sequencing learning mode is activated
       the options of reward function are listed at bottom
    '''


    # this function is called only if self.sequencing_learning_event is triggered
    # when this function is called upon the completion of an operation
    # it add received data to corresponding record in job creator's incomplete_rep_memo
    def complete_experience(self):
        # it's possible that not all machines keep memory for learning
        # machine that needs to keep memory don't keep record for all jobs
        # only when they have to choose from several queuing jobs
        try:
            # check whether corresponding experience exists, if not, ends at this line
            self.job_creator.incomplete_rep_memo[self.m_idx][self.decision_point]
            #print('PARAMETERS',self.m_idx,self.decision_point,self.env.now)
            #print('BEFORE\n',self.job_creator.incomplete_rep_memo[self.m_idx][self.decision_point])
            # if yes, get the global state
            local_data = self.sequencing_data_generation()
            s_t = self.build_state(local_data)
            #print(self.m_idx,s_t)
            r_t = self.reward_function() # can change the reward function, by sepecifying before the training
            #print(self.env.now, r_t)
            self.job_creator.sqc_reward_record.append([self.env.now, r_t])
            self.job_creator.incomplete_rep_memo[self.m_idx][self.decision_point] += [s_t, r_t]
            #print(self.job_creator.incomplete_rep_memo[self.m_idx])
            #print(self.job_creator.incomplete_rep_memo[self.m_idx][self.decision_point])
            complete_exp = self.job_creator.incomplete_rep_memo[self.m_idx].pop(self.decision_point)
            # and add it to rep_memo
            self.job_creator.rep_memo[self.m_idx].append(complete_exp)
            #print(self.job_creator.rep_memo[self.m_idx])
            #print('AFTER\n',self.job_creator.incomplete_rep_memo[self.m_idx][self.decision_point])
            #print(self.m_idx,self.env.now,'state: ',s_t,'reward: ',r_t)
        except:
            pass

    # testing reward function, check if the agent learns, this function encourages using SPT
    def get_reward0(self):
        if self.pt_chosen <= self.current_pt[:self.waiting_jobs-1].mean():
            r_t = 1
        else:
            r_t = 0
        r_t = torch.tensor(r_t, dtype=torch.float)
        return r_t

    # those functions are called only if self.sequencing_learning_event is triggered
    # called only upon the completion of all operations of a job
    # it calculates the reward for all machines that job went through
    # hence a complete experience is constructed for learning

    def get_reward1(self):
        slack = self.before_op_slack
        critical_level = 1 - slack / (np.absolute(slack) + 50)
        # get critical level for jobs, chosen and loser, respectively
        critical_level_chosen = critical_level[self.position]
        critical_level_loser = np.delete(critical_level, self.position) # could be a vector or scalar
        # calculate adjusted earned slack for the chosen job
        earned_slack_chosen = np.mean(self.current_pt[:self.waiting_jobs-1])
        earned_slack_chosen *= critical_level_chosen
        # calculate the AVERAGE adjusted slack consumption for jobs that not been chosen
        consumed_slack_loser = self.pt_chosen*critical_level_loser.mean()
        # slack reward
        rwd_slack = earned_slack_chosen - consumed_slack_loser
        # WINQ reward
        rwd_winq = (self.before_op_winq_loser.mean() - self.before_op_winq_chosen) * 0.2
        # calculate the reward
        #print(rwd_slack, rwd_winq)
        rwd = ((rwd_slack + rwd_winq)/20).clip(-1,1)
        # optional printout
        #print(self.env.now,'slack and pt:', slack, critical_level, self.position, self.pt_chosen, self.current_pt[:self.waiting_jobs-1])
        #print(self.env.now,'winq and reward:',self.before_op_winq_chosen, self.before_op_winq_loser, earned_slack_chosen, consumed_slack_loser)
        #print(self.env.now,'reward:',rwd)
        r_t = torch.tensor(rwd , dtype=torch.float)
        return r_t

    def add_global_reward_RA(self): # BASELINE RULE !!!
        job_record = self.job_creator.production_record[self.job_idx]
        path = job_record[1]
        queued_time = np.array(job_record[2])
        # if tardiness is non-zero and waiting time exists, machines in path get punishment
        if self.tardiness and queued_time.sum():
            global_reward = - np.clip(self.tardiness / 64,0,1)
            reward = torch.ones(len(queued_time),dtype=torch.float)*global_reward
        else:
            reward = torch.ones(len(queued_time),dtype=torch.float)*0
        #print(queued_time)
        #print(self.tardiness,reward)
        for i,m_idx in enumerate(path):
            r_t = reward[i]
            wc_idx = self.m_list[m_idx].wc_idx
            try:
                self.wc_list[wc_idx].incomplete_experience[self.job_idx].insert(2,r_t)
                self.wc_list[wc_idx].rep_memo.append(self.wc_list[wc_idx].incomplete_experience.pop(self.job_idx))
            except:
                pass

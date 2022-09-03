import numpy as np
import random
import sys
from tabulate import tabulate
import matplotlib.pyplot as plt

'''
this module simulates the dynamic arrival of jobs
the feature and timing of new jobs are determined by three types of factors:
the range of processing time, due date tightness, and the expected utilization rate of system
'''

class creation:
    def __init__ (self, env, span, machine_list, workcenter_list, pt_range, due_tightness, E_utliz, **kwargs):
        if 'seed' in kwargs:
            np.random.seed(kwargs['seed'])
            print("Random seed of job creation is fixed, seed: {}".format(kwargs['seed']))
        # environemnt and simulation span
        self.env = env
        self.span = span
        # all machines and workcenters
        self.m_list = machine_list
        self.wc_list = workcenter_list
        # the longest operaiton sequence passes all workcenters
        self.no_wcs = len(self.wc_list)
        self.no_machines = len(self.m_list)
        self.m_per_wc = int(self.no_machines / self.no_wcs)
        # the dictionary that records the details of operation and tardiness
        self.production_record = {}
        self.tardiness_record = {}
        # the reward record
        self.sqc_reward_record = []
        self.rt_reward_record = []
        # range of processing time
        self.pt_range = pt_range
        # calulate the average processing time of a single operation
        self.avg_pt = np.average(self.pt_range) - 0.5
        # tightness factor of jobs
        self.tightness = due_tightness
        # expected utlization rate of machines
        self.E_utliz = E_utliz
        # generate a upscending seed for generating initial sequence, start from 0
        self.sequence_seed = np.arange(self.no_wcs)
        # set a variable to track the number of in-system number of jobs
        self.in_system_job_no = 0
        self.in_system_job_no_dict = {}
        self.index_jobs = 0
        # set lists to track the completion rate, realized and expected tardy jobs in system
        self.comp_rate_list = [[] for m in self.m_list]
        self.comp_rate = 0
        self.realized_tard_list = [[] for m in self.m_list]
        self.realized_tard_rate = 0
        self.exp_tard_list = [[] for m in self.m_list]
        self.exp_tard_rate = 0
        # initialize the information associated with jobs that are being processed
        self.available_time_list = np.array([0 for m in self.m_list])
        self.release_time_list = np.array([self.avg_pt for m in self.m_list])
        self.current_j_idx_list = np.arange(self.no_machines)
        self.next_wc_list = np.array([-1 for m in self.m_list])
        self.next_pt_list = np.array([self.avg_pt for m in self.m_list])
        self.arriving_job_rempt_list = np.array([0 for m in self.m_list])
        self.next_ttd_list = np.array([self.avg_pt*self.no_wcs for m in self.m_list])
        self.arriving_job_slack_list = np.array([0 for m in self.m_list])
        # and create an empty, initial array of sequence
        self.sequence_list = []
        self.pt_list = []
        self.remaining_pt_list = []
        self.create_time = []
        self.due_list = []
        # record the arrival and departure information
        self.arrival_dict = {}
        self.departure_dict = {}
        self.mean_dict = {}
        self.std_dict = {}
        self.expected_tardiness_dict = {}
        # decide the feature of new job arrivals
        # beta is the average time interval between job arrivals
        # let beta equals half of the average time of single operation
        self.beta = self.avg_pt / (self.m_per_wc * self.E_utliz)
        # number of new jobs arrive during the simulation
        self.total_no = np.round(self.span/self.beta).astype(int)
        # the interval between job arrivals follows the exponential distribution
        self.arrival_interval = np.random.exponential(self.beta, self.total_no).round()
        # settings
        if 'realistic_var' in kwargs and kwargs['realistic_var']:
            self.ptl_generation = self.ptl_generation_realistic
            self.realistic_var = kwargs['realistic_var']
        else:
            self.ptl_generation = self.ptl_generation_random
        if 'random_seed' in kwargs and kwargs['random_seed']:
            interval = self.span/50
            self.env.process(self.dynamic_seed_change(interval))
        if 'hetero_len' in kwargs and kwargs['hetero_len']:
            pass
        if 'even' in kwargs and kwargs['even']:
            print("EVEN mode ON")
            #print(self.arrival_interval)
            self.arrival_interval = np.ones(self.arrival_interval.size)*self.arrival_interval.mean()
            #print(self.arrival_interval)
        self.initial_job_assignment()
        # start the new job arrival
        self.env.process(self.new_job_arrival())

    def initial_job_assignment(self):
        sqc_seed = np.arange(self.no_wcs) # array of idx of workcenters
        for wc_idx,wc in enumerate(self.wc_list): # for every work center
            np.random.shuffle(sqc_seed)
            sqc = np.concatenate([np.array([wc_idx]),sqc_seed[sqc_seed!=wc_idx]])
            for m_idx,m in enumerate(wc.m_list): # for every machine inside this workcenter
                # allocate the job index to corrsponding workcenter's queue
                self.sequence_list.append(sqc)
                # produce processing time of job, get corresponding remaining_pt_list
                ptl = self.ptl_generation()
                self.pt_list.append(ptl)
                self.record_job_feature(self.index_jobs,ptl)
                # reshape and rearrange the order of ptl to get remaining pt list
                remaining_ptl = np.reshape(ptl,[self.no_wcs,self.m_per_wc])[sqc]
                #print(self.remaining_pt_list, remaining_ptl)
                self.remaining_pt_list.append(remaining_ptl)
                # produce due date for job
                avg_pt = ptl.mean()
                due = np.round(avg_pt*self.no_wcs*np.random.uniform(1, self.tightness))
                # record the creation time
                self.create_time.append(0)
                # add due date to due list, cannot specify axis
                self.due_list.append(due)
                # update the in-system-job number
                self.record_job_arrival()
                # operation record, path, wait time, decision points, slack change
                self.production_record[self.index_jobs] = [[],[],[],{},[]]
                #print("**ARRIVAL: Job %s, time:%s, sqc:%s, pt:%s, due:%s"%(self.index_jobs, self.env.now, self.sequence_seed ,ptl, due))
                '''after creation of new job, add it to workcernter'''
                # add job to system and create the data repository for job
                wc.queue.append(self.index_jobs)
                # allocate the sequence of that job to corresponding workcenter's storage
                # the added sequence is the one without first element, coz it's been dispatched
                wc.sequence_list.append(np.delete(self.sequence_list[self.index_jobs],0))
                # allocate the processing time of that job to corresponding workcenter's storage
                wc.pt_list.append(self.pt_list[self.index_jobs])
                wc.remaining_pt_list.append(self.remaining_pt_list[self.index_jobs])
                # allocate the due of that job to corresponding workcenter's storage
                wc.due_list.append(self.due_list[self.index_jobs])
                self.index_jobs += 1
            # after assigned the initial job to workcenter, activate its routing behavior
            wc.routing_event.succeed()

    def new_job_arrival(self):
        # main process
        while self.index_jobs < self.total_no:
            # draw the time interval betwen job arrivals from exponential distribution
            # The mean of an exp random variable X with rate parameter λ is given by:
            # 1/λ (which equals the term "beta" in np exp function)
            time_interval = self.arrival_interval[self.index_jobs]
            yield self.env.timeout(time_interval)
            # produce sequence of job, first shuffle the sequence seed
            np.random.shuffle(self.sequence_seed)
            self.sequence_list.append(np.copy(self.sequence_seed))
            # produce processing time of job, get corresponding remaining_pt_list
            ptl = self.ptl_generation()
            self.pt_list.append(ptl)
            self.record_job_feature(self.index_jobs,ptl)
            # reshape and rearrange the order of ptl to get remaining pt list
            remaining_ptl = np.reshape(ptl,[self.no_wcs,self.m_per_wc])[self.sequence_seed]
            #print(self.remaining_pt_list, remaining_ptl)
            self.remaining_pt_list.append(remaining_ptl)
            # produce due date for job
            avg_pt = ptl.mean()
            due = np.round(avg_pt*self.no_wcs*np.random.uniform(1, self.tightness) + self.env.now)
            # record the creation time
            self.create_time.append(self.env.now)
            # add due date to due list, cannot specify axis
            self.due_list.append(due)
            #print("**ARRIVAL: Job %s, time:%s, sqc:%s, pt:%s, due:%s"%(self.index_jobs, self.env.now, self.sequence_seed ,ptl, due))
            '''after creation of new job, add it to workcernter'''
            # first workcenter of that job
            first_workcenter = self.sequence_seed[0]
            # add job to system and create the data repository for job
            self.record_job_arrival()
            # operation record, path, wait time, decision points, slack change
            self.production_record[self.index_jobs] = [[],[],[],{},[]]
            # add job to workcenter
            self.wc_list[first_workcenter].queue.append(self.index_jobs)
            # add sequence list to workcenter's storage
            self.wc_list[first_workcenter].sequence_list.append(np.delete(self.sequence_list[self.index_jobs],0))
            # allocate the processing time of that job to corresponding workcenter's storage
            self.wc_list[first_workcenter].pt_list.append(self.pt_list[self.index_jobs])
            self.wc_list[first_workcenter].remaining_pt_list.append(self.remaining_pt_list[self.index_jobs])
            # allocate the due of that job to corresponding workcenter's storage
            self.wc_list[first_workcenter].due_list.append(self.due_list[self.index_jobs])
            # update index for next new job
            self.index_jobs += 1
            # and activate the dispatching of the work center
            try:
                self.wc_list[first_workcenter].routing_event.succeed()
            except:
                pass

    def ptl_generation_random(self):
        ptl = np.random.randint(self.pt_range[0], self.pt_range[1], size = [self.no_machines])
        return ptl

    def ptl_generation_realistic(self):
        base = np.random.randint(self.pt_range[0], self.pt_range[1], [self.no_wcs,1]) * np.ones([self.no_wcs, self.m_per_wc])
        variation = np.random.randint(-self.realistic_var,self.realistic_var,[self.no_wcs, self.m_per_wc])
        #print(base,variation)
        ptl = (base + variation).clip(self.pt_range[0], self.pt_range[1])
        ptl = np.concatenate(ptl)
        return ptl

    def dynamic_seed_change(self, interval):
        while self.env.now < self.span:
            yield self.env.timeout(interval)
            seed = np.random.randint(2000000000)
            np.random.seed(seed)
            print('change random seed to {} at time {}'.format(seed,self.env.now))

    def change_setting(self,pt_range):
        print('Heterogenity changed at time',self.env.now)
        self.pt_range = pt_range
        self.avg_pt = np.average(self.pt_range)-0.5
        self.beta = self.avg_pt / (2*self.E_utliz)

    def get_global_exp_tard_rate(self):
        x = []
        for m in self.m_list:
            x = np.append(x, m.slack)
        rate = x[x<0].size / x.size
        return rate

    # this fucntion record the time and number of new job arrivals
    def record_job_arrival(self):
        self.in_system_job_no += 1
        self.in_system_job_no_dict[self.env.now] = self.in_system_job_no
        try:
            self.arrival_dict[self.env.now] += 1
        except:
            self.arrival_dict[self.env.now] = 1

    # this function is called upon the completion of a job, by machine agent
    def record_job_departure(self):
        self.in_system_job_no -= 1
        self.in_system_job_no_dict[self.env.now] = self.in_system_job_no
        try:
            self.departure_dict[self.env.now] += 1
        except:
            self.departure_dict[self.env.now] = 1

    def record_job_feature(self,idx,ptl):
        self.mean_dict[idx] = (self.env.now, ptl.mean())
        self.std_dict[idx] = (self.env.now, ptl.std())

    # the sum of remaining processing time of all jobs in system
    # divided by the total number of machines on shop floor
    # is the estimation of waiting time for a new arrived job
    def get_expected_tardiness(self, ptl, due):
        sum_remaining_pt = sum([m.remaining_job_pt.sum() for m in self.m_list])
        expected_waiting_time = sum_remaining_pt / self.no_machines
        expected_processing_time = ptl.mean() * self.no_wcs
        expected_tardiness = expected_processing_time + expected_waiting_time + self.env.now - due
        #print('exp tard. of job {}: '.format(self.index_jobs),due, max(0, expected_tardiness))
        self.expected_tardiness_dict[self.index_jobs] = max(0, expected_tardiness)

    def build_sqc_experience_repository(self,m_list):
        # "grand" dictionary for replay memory
        self.incomplete_rep_memo = {}
        self.rep_memo = {}
        # for each machine to be controlled, build a sub-dictionary
        # because incomplete experience must be indexed by job's index
        for m in m_list:
            self.incomplete_rep_memo[m.m_idx] = {}
            self.rep_memo[m.m_idx] = []

    def output(self):
        print('job information are as follows:')
        job_info = [[i,self.sequence_list[i], self.pt_list[i], \
        self.create_time[i], self.due_list[i]] for i in range(self.index_jobs)]
        print(tabulate(job_info, headers=['idx.','sqc.','proc.t.','in','due']))
        print('--------------------------------------')
        return job_info

    def final_output(self):
        # information of job output time and realized tardiness
        output_info = []
        for item in self.production_record:
            output_info.append(self.production_record[item][5])
        job_info = [[i,self.sequence_list[i], self.pt_list[i], self.create_time[i],\
        self.due_list[i], output_info[i][0], output_info[i][1]] for i in range(self.index_jobs)]
        print(tabulate(job_info, headers=['idx.','sqc.','proc.t.','in','due','out','tard.']))
        realized = np.array(output_info)[:,1].sum()
        exp_tard = sum(self.expected_tardiness_dict.values())

    def tardiness_output(self):
        # information of job output time and realized tardiness
        tard_info = []
        #print(self.production_record)
        for item in self.production_record:
            #print(item,self.production_record[item])
            tard_info.append(self.production_record[item][5])
        # now tard_info is an ndarray of objects, cannot be sliced. need covert to common np array
        # if it's a simple ndarray, can't sort by index
        dt = np.dtype([('output', float),('tardiness', float)])
        tard_info = np.array(tard_info, dtype = dt)
        tard_info = np.sort(tard_info, order = 'output')
        # now tard_info is an ndarray of objects, cannot be sliced, need covert to common np array
        tard_info = np.array(tard_info.tolist())
        tard_info = np.array(tard_info)
        output_time = tard_info[:,0]
        tard = np.absolute(tard_info[:,1])
        cumulative_tard = np.cumsum(tard)
        tard_max = np.max(tard)
        tard_mean = np.cumsum(tard) / np.arange(1,len(cumulative_tard)+1)
        tard_rate = tard.clip(0,1).sum() / tard.size
        #print(output_time, cumulative_tard, tard_mean)
        return output_time, cumulative_tard, tard_mean, tard_max, tard_rate

    def record_printout(self):
        print(self.production_record)

    def timing_output(self):
        return self.arrival_dict, self.departure_dict, self.in_system_job_no_dict

    def feature_output(self):
        return self.mean_dict, self.std_dict

    def all_tardiness(self):
        # information of job output time and realized tardiness
        tard = []
        #print(self.production_record)
        for item in self.production_record:
            #print(item,self.production_record[item])
            tard.append(self.production_record[item][5][1])
        #print(tard)
        tard = np.array(tard)
        mean_tardiness = tard.mean()
        #print(self.production_record)
        #print(tard)
        tardy_rate = tard.clip(0,1).sum() / tard.size
        #print(output_time, cumulative_tard, tard_mean)
        return mean_tardiness, tardy_rate

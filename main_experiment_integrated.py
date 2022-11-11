import simpy
import sys
sys.path #sometimes need this to refresh the path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import numpy as np
from tabulate import tabulate
import pandas as pd
from pandas import DataFrame

import agent_machine
import agent_workcenter
import sequencing
import routing
import job_creation
import breakdown_creation
import heterogeneity_creation
import validation_S
import validation_R

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
        if 'seed' in kwargs:
            self.job_creator = job_creation.creation\
            (self.env, self.span, self.m_list, self.wc_list, kwargs['pt_range'], kwargs['tightness'], 0.9, seed=kwargs['seed'])
            #self.job_creator.output()
        else:
            print("WARNING: seed is not fixed !!")
            raise Exception

        '''STEP 4: initialize machines and work centers'''
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)
        for i,m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i/m_per_wc)
            m.initialization(self.m_list,self.wc_list,self.job_creator,self.wc_list[wc_idx])

        '''STEP 5: set sequencing or routing rules, and DRL'''
        if 'sequencing_rule' in kwargs:
            #print("Taking over: machines use {} sequencing rule".format(kwargs['sequencing_rule']))
            for m in self.m_list:
                order = "m.job_sequencing = sequencing." + kwargs['sequencing_rule']
                try:
                    exec(order)
                except:
                    print("Rule assigned to machine {} is invalid !".format(m.m_label))
                    raise Exception

        # check if need to reset routing rule
        if 'routing_rule' in kwargs:
            #print("Taking over: workcenters use {} routing rule".format(kwargs['routing_rule']))
            for wc in self.wc_list:
                order = "wc.job_routing = routing." + kwargs['routing_rule']
                try:
                    exec(order)
                except:
                    print("Rule assigned to workcenter {} is invalid !".format(wc.wc_idx))
                    raise Exception
        # specify the architecture of DRL
        if 'DRL_AS' in kwargs and kwargs['DRL_AS']:
            print("---> Overall DRL_SA mode ON <---")
            kwargs['DRL_R'] = True
            kwargs['AS'] = True
        # specify the sequencing agents
        if 'AS' in kwargs and kwargs['AS']:
            print("---> AS mode ON <---")
            self.sequencing_brain = validation_S.DRL_sequencing(self.env, self.m_list, self.job_creator, self.span, \
            show = 0, validated=1, reward_function = '')
        # specify the routing agents
        if 'DRL_R' in kwargs and kwargs['DRL_R']:
            print("---> DRL Routing mode ON <---")
            self.routing_brain = validation_R.DRL_routing(self.env, self.job_creator, self.wc_list, \
            validated = 1)
        # specify the routing agents
        if 'Lang2020' in kwargs and kwargs['Lang2020']:
            print("---> Lang Routing mode ON <---")
            self.routing_brain = validation_R.DRL_routing(self.env, self.job_creator, self.wc_list, \
            Lang2020 = 1)
        # specify the routing agents
        if 'Luo2020' in kwargs and kwargs['Luo2020']:
            print("---> Luo Routing mode ON <---")
            self.routing_brain = validation_R.DRL_routing(self.env, self.job_creator, self.wc_list, \
            Luo2020 = 1)

    def simulation(self):
        self.env.run()

# dictionary to store shopfloors and production record
spf_dict = {}
production_record = {}
# list of experiments


benchmark_HH = [['FIFO','EA'],['AVPRO','CT'],['SPT','CT'],['LWKRSPT','CT'],['PTWINQ','CT'],['DPTWINQNPT','CT'],['AVPRO','ET'],['SPT','ET'],['LWKRSPT','ET'],['PTWINQ','ET'],['DPTWINQNPT','ET'],['GP_S1',"EA"],['GP_S2',"EA"]]
benchmark_HL = [['FIFO','EA'],['ATC','CT'],['CR','CT'],['EDD','CT'],['MOD','CT'],['DPTLWKRS','CT'],['ATC','ET'],['CR','ET'],['EDD','ET'],['MOD','ET'],['DPTLWKRS','ET'],['GP_S1',"EA"],['GP_S2',"EA"]]
benchmark_LH = [['FIFO','EA'],['EDD','CT'],['MOD','CT'],['MS','CT'],['PTWINQS','CT'],['DPTLWKRS','CT'],['EDD','EA'],['MOD','EA'],['MS','EA'],['PTWINQS','EA'],['DPTLWKRS','EA'],['GP_S1',"EA"],['GP_S2',"EA"]]
benchmark_LL = [['FIFO','EA'],['CR','CT'],['MOD','CT'],['MS','CT'],['PTWINQS','CT'],['DPTLWKRS','CT'],['CR','EA'],['MOD','EA'],['MS','EA'],['PTWINQS','EA'],['DPTLWKRS','EA'],['GP_S1',"EA"],['GP_S2',"EA"]]

pt_range = [5,26]
tightness = 3

if pt_range[1]/pt_range[0] > 2.5:
    if tightness==2:
        benchmark = benchmark_HH
    else:
        benchmark = benchmark_HL
else:
    if tightness==2:
        benchmark = benchmark_LH
    else:
        benchmark = benchmark_LL

#title = [x[0]+'+'+x[1] for x in benchmark] + ['deep MARL-AS',"deep MARL-MR","Integrated SA+RA"]
span = 1000
m_no = 6
wc_no = 3

DRLs = ['Lang2020','Luo2020','DRL_R',"AS",'DRL_AS']
title = [x[0]+'+'+x[1] for x in benchmark[:-2]] + ['GP1','GP2'] + ['Lang2020','Luo2020','RA alone','SA alone',"Integrated DRL"]
print(title)

sum_record = []
benchmark_record = []
max_record = []
rate_record = []
iteration = 10
# dont mess with above one
export_result = 0

for run in range(iteration):
    print('******************* ITERATION-{} *******************'.format(run))
    sum_record.append([])
    benchmark_record.append([])
    max_record.append([])
    rate_record.append([])
    seed = np.random.randint(2000000000)
    # run simulation with different rules
    for idx,rule in enumerate(benchmark):
        # create the environment instance for simulation
        env = simpy.Environment()
        spf = shopfloor(env, span, m_no, wc_no, pt_range = pt_range, tightness = tightness, sequencing_rule = rule[0], routing_rule = rule[1], seed = seed)
        spf.simulation()
        output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
        sum_record[run].append(cumulative_tard[-1])
        benchmark_record[run].append(cumulative_tard[-1])
        max_record[run].append(tard_max)
        rate_record[run].append(tard_rate)
    for idx,x in enumerate(DRLs):
        # and extra run with DRL
        env = simpy.Environment()
        exec('spf = shopfloor(env, span, m_no, wc_no, pt_range = pt_range, tightness = tightness, {} = True, seed = seed)'.format(x))
        spf.simulation()
        output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
        sum_record[run].append(cumulative_tard[-1])
        max_record[run].append(tard_max)
        rate_record[run].append(tard_rate)

print('-------------- Complete Record --------------')
print(tabulate(sum_record, headers=title))
print('-------------- Average Performance --------------')

# get the performnce without DRL
avg_b = np.mean(benchmark_record,axis=0)
ratio_b = np.around(avg_b/avg_b.max()*100,2)
winning_rate_b = np.zeros(len(title))
for idx in np.argmin(benchmark_record,axis=1):
    winning_rate_b[idx] += 1
winning_rate_b = np.around(winning_rate_b/iteration*100,2)

# get the overall performance (include DRL)
avg = np.mean(sum_record,axis=0)
max = np.mean(max_record,axis=0)
tardy_rate = np.around(np.mean(rate_record,axis=0)*100,2)
ratio = np.around(avg/avg.min()*100,2)
rank = np.argsort(ratio)
winning_rate = np.zeros(len(title))
for idx in np.argmin(sum_record,axis=1):
    winning_rate[idx] += 1
winning_rate = np.around(winning_rate/iteration*100,2)
for rank,rule in enumerate(rank):
    print("{}, avg.: {} | max: {} | %: {}% | tardy %: {}% | winning rate: {}/{}%"\
    .format(title[rule],avg[rule],max[rule],ratio[rule],tardy_rate[rule],winning_rate_b[rule],winning_rate[rule]))

if export_result:
    df_win_rate = DataFrame([winning_rate], columns=title)
    #print(df_win_rate)
    df_sum = DataFrame(sum_record, columns=title)
    #print(df_sum)
    df_tardy_rate = DataFrame(rate_record, columns=title)
    #print(df_tardy_rate)
    df_max = DataFrame(max_record, columns=title)
    #print(df_max)
    df_before_win_rate = DataFrame([winning_rate_b], columns=title)
    address = sys.path[0]+'\\experiment_result\\RAW_integrated_experiment.xlsx'
    Excelwriter = pd.ExcelWriter(address,engine="xlsxwriter")
    dflist = [df_win_rate, df_sum, df_tardy_rate, df_max, df_before_win_rate]
    sheetname = ['win rate','sum', 'tardy rate', 'maximum','before win rate']

    for i,df in enumerate(dflist):
        df.to_excel(Excelwriter, sheet_name=sheetname[i], index=False)
    Excelwriter.save()
    print('export to {}'.format(address))

# check the parameter and scenario setting
spf.routing_brain.check_parameter()
spf.sequencing_brain.check_parameter()

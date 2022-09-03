import simpy
import random
import numpy as np

'''
this module contains the job sequencing rules used for comparison
'''

# Benchmark, as the worst possible case
def random_sequencing(data):
    job_position = np.random.randint(len(data[0]))
    return job_position

def SPT(data): # shortest processing time
    job_position = np.argmin(data[0])
    return job_position

def LPT(data): # longest processing time
    job_position = np.argmax(data[0])
    return job_position

def LRO(data): # least remaining operations / highest completion rate
    job_position = np.argmax(data[10])
    return job_position

def LWKR(data): # least work remaining
    job_position = np.argmin(data[0] + data[1])
    return job_position

def LWKRSPT(data): # remaining work + SPT
    job_position = np.argmin(data[0]*2 + data[1])
    return job_position

def LWKRMOD(data): # remaining work + MOD
    due = data[2]
    operational_finish = data[0] + data[3]
    MOD = np.max([due,operational_finish],axis=0)
    job_position = np.argmin(data[0] + data[1] + MOD)
    return job_position

def EDD(data):
    # choose the job with earlist due date
    job_position = np.argmin(data[2])
    return job_position

def COVERT(data): # cost over time
    average_pt = data[0].mean()
    cost = (data[2] - data[3] - data[0]).clip(0,None)
    priority = (1 - cost / (0.05*average_pt)).clip(0,None) / data[0]
    job_position = priority.argmax()
    return job_position

def CR(data):
    time_till_due = data[5]
    CR = time_till_due / (data[0] + data[1])
    job_position = CR.argmin()
    return job_position

def CRSPT(data): # CR+SPT
    CRSPT = data[5] / (data[0] + data[1]) + data[0]
    job_position = CRSPT.argmin()
    return job_position

def MS(data):
    slack = data[6]
    job_position = slack.argmin()
    return job_position

def MDD(data): # The modified due date is a job's original due date or its early finish time, whichever is larger
    due = data[2]
    finish = data[1] + data[3]
    MDD = np.max([due,finish],axis=0)
    job_position = MDD.argmin()
    return job_position

def MON(data):
    # Montagne's heuristic, this rule combines SPT with additional slack factor
    due_over_pt = np.array(data[2])/np.sum(data[0])
    priority = due_over_pt/np.array(data[0])
    job_position = priority.argmax()
    return job_position

def MOD(data): # The modified operational due date
    due = data[2]
    operational_finish = data[0] + data[3]
    MOD = np.max([due,operational_finish],axis=0)
    job_position = MOD.argmin()
    return job_position

def NPT(data): # next processing time
    job_position = np.argmin(data[9])
    return job_position

def ATC(data): # http://www.growingscience.com/ijiec/Vol7/IJIEC_2015_23.pdf
    #print(data)
    average_pt = data[0].mean()
    cost = (data[2] - data[3] - data[0]).clip(0,None)
    #print(average_pt, AT)
    priority = np.exp( - cost / (0.05*average_pt)) / data[0]
    #print(priority)
    job_position = priority.argmax()
    return job_position

def AVPRO(data): # average processing time per operation
    AVPRO = (data[0] + data[1]) / (data[10] + 1)
    job_position = AVPRO.argmin()
    return job_position

def SRMWK(data): # slack per remaining work, identical to CR
    SRMWK = data[6] / (data[0] + data[1])
    job_position = SRMWK.argmin()
    return job_position

def SRMWKSPT(data): # slack per remaining work + SPT, identical to CR+SPT
    SRMWKSPT = data[6] / (data[0] + data[1]) + data[0]
    job_position = SRMWKSPT.argmin()
    return job_position

def WINQ(data): # WINQ
    job_position = data[7].argmin()
    return job_position

def PTWINQ(data): # PT + WINQ
    sum = data[0] + data[7]
    job_position = sum.argmin()
    return job_position

def PTWINQS(data): # PT + WINQ + Slack
    sum = data[0] + data[6] + data[7]
    job_position = sum.argmin()
    return job_position

def DPTWINQNPT(data): # 2PT + WINQ + NPT
    sum = data[0]*2 + data[7] + data[9]
    job_position = sum.argmin()
    return job_position

def DPTLWKR(data): # 2PT + LWKR
    sum = data[0]*2 + data[1]
    job_position = sum.argmin()
    return job_position

def DPTLWKRS(data): # 2PT + LWKR + slack
    sum = data[0]*2 + data[1] + data[6]
    job_position = sum.argmin()
    return job_position

def FIFO(dummy): # first in, first out, data is not needed
    job_position = 0
    return job_position

def GP_S1(data): # genetic programming rule 1
    sec1 = data[0] + data[1]
    sec2 = (data[7]*2-1) / data[0]
    sec3 = (data[7] + data[1] + (data[0]+data[1])/(data[7]-data[1])) / data[0]
    sum = sec1-sec2-sec3
    job_position = sum.argmin()
    return job_position

def GP_S2(data): # genetic programming rule 2
    NIQ = len(data[0])
    sec1 = NIQ * (data[0]-1)
    sec2 = data[0] + data[1] * np.max([data[0],data[7]],axis=0)
    sec3 = np.max([data[7],NIQ+data[7]],axis=0)
    sec4 = (data[8]+1+np.max([data[1],np.ones_like(data[1])*(NIQ-1)],axis=0)) * np.max([data[7],data[1]],axis=0)
    sum = sec1 * sec2 + sec3 * sec4
    job_position = sum.argmin()
    return job_position

def GP_S3(data): # genetic programming rule 1
    sec1 = data[0] + data[1]
    sec2 = (data[7]*2-1) / data[0]
    sec3 = (data[7] + data[1] + (data[0]+data[1])/(data[7]-data[1])) / data[0]
    sum = sec1-sec2-sec3
    job_position = sum.argmin()
    return job_position

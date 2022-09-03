import numpy as np

class creation:
    def __init__(self, env, target, event_intervals, pt_range_list, **kwargs):
        self.env = env
        self.target = target
        self.event_intervals = list(event_intervals)
        self.pt_range_list = list(pt_range_list)
        # the list of start and end time of events
        print('Durations and pt:',event_intervals, pt_range_list)
        print("--------------------------------------")
        # see if number of events is correct
        if len(self.event_intervals) - len(self.pt_range_list):
            print('Unmatching size of events')
            raise KeyError
        # main process
        self.env.process(self.manipulation())

    def manipulation(self):
        while len(self.event_intervals):
            print("Time {}, change the heterogenity of arriving jobs to: {}".format(self.env.now, self.pt_range_list[0]))
            self.target.change_setting(self.pt_range_list.pop(0))
            yield self.env.timeout(self.event_intervals.pop(0))

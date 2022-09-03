import numpy as np

class creation:
    def __init__(self, env, machine_list, target_index, event_intervals, duration,**kwargs):
        self.env = env
        self.m_list = machine_list
        self.target_index = target_index
        self.event_intervals = event_intervals
        self.duration = duration
        # the list of start and end time of events
        self.event_start_time = np.cumsum(self.event_intervals)
        self.event_end_time = self.event_start_time + duration
        # convert to list, so can use .pop()
        self.event_start_time = self.event_start_time.tolist()
        self.event_end_time = self.event_end_time.tolist()
        print("The event start time: %s\nThe event end time: %s"%(self.event_start_time, self.event_end_time))
        print("--------------------------------------")
        self.event_number = len(target_index)
        # see if number of events is correct
        if len(target_index) - len(event_intervals):
            print('Unmatching size of events')
            raise KeyError
        # main process
        self.env.process(self.manipulation())

    def manipulation(self):
        for i in range(self.event_number):
            yield self.env.timeout(self.event_intervals.pop(0))
            idx = self.target_index.pop(0)
            duration = self.duration.pop(0)
            restart_time = self.event_end_time.pop(0)
            self.env.process(self.event_process(idx, duration, restart_time))

    def event_process(self, idx, duration, restart_time):
        # shut down the machine
        self.m_list[idx].working_event = self.env.event()
        # change the available time for machine
        self.m_list[idx].restart_time = restart_time
        yield self.env.timeout(duration)
        self.m_list[idx].restart_time = 0
        self.m_list[idx].working_event.succeed()

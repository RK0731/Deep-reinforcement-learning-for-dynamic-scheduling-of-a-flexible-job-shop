import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

'''
saving trained parameters
'''

sys.path
print(sys.path)


# copy the old file
from_address = "{}\\validated_3machine_state_dict.pt".format(sys.path[0])
parameters = torch.load(from_address)
print("from:",from_address)
# to new file
to_address = "{}\\validated_3machine_medium.pt".format(sys.path[0])
print("to:",to_address)


torch.save(parameters, to_address)

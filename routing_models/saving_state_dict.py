import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path
print(sys.path)

'''
# copy the old file
from_address = "{}\\Deeper_state_dict.pt".format(sys.path[0])
parameters = torch.load(from_address)
print("from:",from_address)
# to new file
to_address = "{}\\0123_state_dict.pt".format(sys.path[0])
print("to:",to_address)
'''

'''
# copy the old file
from_address = "{}\\Extended_state_dict.pt".format(sys.path[0])
parameters = torch.load(from_address)
print("from:",from_address)
# to new file
to_address = "{}\\validated_5machine_state_dict2.pt".format(sys.path[0])
print("to:",to_address)
'''

# copy the old file
from_address = "{}\\TEST_state_dict.pt".format(sys.path[0])
parameters = torch.load(from_address)
print("from:",from_address)
# to new file
to_address = "{}\\validated_4machine_large.pt".format(sys.path[0])
print("to:",to_address)
'''
# copy the old file
from_address = "{}\\validated_4machine_large.pt".format(sys.path[0])
parameters = torch.load(from_address)
print("from:",from_address)
# to new file
to_address = "{}\\validated_4machine_large2.pt".format(sys.path[0])
print("to:",to_address)
'''

torch.save(parameters, to_address)

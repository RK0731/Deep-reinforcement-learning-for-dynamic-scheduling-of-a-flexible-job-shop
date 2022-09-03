import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path
print(sys.path)
for i in range(6):
    # copy the old file
    from_address = "{}\\validated_HH_ext6.pt".format(sys.path[0])
    parameters = torch.load(from_address)
    print("from:",from_address)
    # to new file
    to_address = "{}\\validated_HH_ext9.pt".format(sys.path[0])
    print("to:",to_address)
    torch.save(parameters, to_address)

'''

# copy the old file
from_address = "{}\\MR_validated_5ops.pt".format(sys.path[0],0)
parameters = torch.load(from_address)
print("from:",from_address)
# to new file
to_address = "{}\\MR_validated_5ops_1.pt".format(sys.path[0])
print("to:",to_address)
torch.save(parameters, to_address)
'''
'''

# copy the old file
from_address = "{}\\MR_direct_rwd13.pt".format(sys.path[0],0)
parameters = torch.load(from_address)
print("from:",from_address)
# to new file
to_address = "{}\\MR_validated_5ops.pt".format(sys.path[0])
print("to:",to_address)
torch.save(parameters, to_address)
'''

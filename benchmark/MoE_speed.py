from FusedOneMoE import FusedOneMoE
from MoE_modeling import MoE
import time
import torch

model1 = MoE(128, 512, 32, 8)
model2 = FusedOneMoE(128, 512, 32, True, 8)
x = torch.randn(1, 512, 128)

time1 = time.time()
for _ in range(100):
    y = model1(x)
time2 = time.time()
for _ in range(100):
    y = model2(x)
time3 = time.time()

print('func1: {:.3f} sec'.format(time2 - time1))
print('func2: {:.3f} sec'.format(time3 - time2))

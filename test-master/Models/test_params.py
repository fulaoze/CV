from Models.networks import *
import torch
import time
#from thop import profile
from ptflops import get_model_complexity_info
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

################修改batch size
batch_size = 2
model = InceptionResnetV1()
# model.fc = nn.Sequential(nn.Linear(2048, 2048),
#                                          nn.ReLU(),
#                                          model.fc)
model = model.cuda()
print(model)
x = torch.randn(batch_size,3,128,128).cuda()
# per_normal = [[x for x in range(7)] for i in range(batch_size)]
# per_normal = torch.tensor(per_normal).cuda(0)

##########################
model.eval()
with torch.no_grad():
    output = model(x)

start = time.time()
with torch.no_grad():
    for i in range(10):
        output = model(x)
end = time.time()
print('cost time:', start, end, (end-start)/10)


flops, params = get_model_complexity_info(model, (3, 128, 128), as_strings=True, print_per_layer_stat=True) #不用写batch_size大小，默认batch_size=1
print('Flops:  ' + flops)
print('Params: ' + params)
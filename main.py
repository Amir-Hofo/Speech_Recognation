from model import *


# config
phase= 'train' # 'inference'

waveforms, targets= next(iter(train_loader))
print(waveforms.shape, targets.shape, '\n')

model= CustomModel(fs= 22050)
print(model(waveforms).shape)


# print("train batch size:",train_loader.batch_size,
#      ", num of batch:", len(train_loader))
# print("valid batch size:",valid_loader.batch_size,
#      ", num of batch:", len(valid_loader))
# print("test batch size:",test_loader.batch_size,
#      ", num of batch:", len(test_loader))
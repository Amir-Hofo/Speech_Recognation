from training import *

# config
phase= 'train' # 'inference'

waveforms, targets= next(iter(train_loader))
print(waveforms.shape, targets.shape, '\n')


print(model(waveforms, targets).shape)
print(num_trainable_params(model))


# print("train batch size:",train_loader.batch_size,
#      ", num of batch:", len(train_loader))
# print("valid batch size:",valid_loader.batch_size,
#      ", num of batch:", len(valid_loader))
# print("test batch size:",test_loader.batch_size,
#      ", num of batch:", len(test_loader))
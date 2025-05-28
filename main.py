from model import *
warnings.filterwarnings("ignore")

# config
phase= 'train' # 'inference'

waveforms, targets= next(iter(train_loader))
print(waveforms.shape, targets.shape, '\n')

model= CustomModel(fs= 22050, d_model= 1280, n_head= 4, 
                   num_encoders= 4, num_decoders= 1,
                   dim_feedforward= 640* 2, planes= 64)
print(model(waveforms, targets).shape)


# print("train batch size:",train_loader.batch_size,
#      ", num of batch:", len(train_loader))
# print("valid batch size:",valid_loader.batch_size,
#      ", num of batch:", len(valid_loader))
# print("test batch size:",test_loader.batch_size,
#      ", num of batch:", len(test_loader))
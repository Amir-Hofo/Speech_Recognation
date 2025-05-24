from preprocess import *

class CustomModel(nn.Module):
    def __init__(self, fs, n_mels= 80, n_fft= 400, 
                 inplanes= 32, planes= 64):
        super().__init__()
        # transform
        self.transforms= nn.Sequential(T.Resample(orig_freq= fs, new_freq= 16000),
                                       T.MelSpectrogram(n_mels= n_mels, n_fft= n_fft)
                                       ).requires_grad_(False)
        
        # feature embedding
        self.cnn= CNN2DFeatureExtractor(inplanes= inplanes, planes= planes)
        
    def forward(self, src):
        with torch.no_grad():
            src= self.transforms(src)
        src= self.cnn(src)
        return src


class CNN2DFeatureExtractor(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        self.conv1= nn.Conv2d(1, inplanes, kernel_size= 11,
                              stride= 1, padding= 5, bias= False)
        self.bn1= nn.BatchNorm2d(inplanes)
        self.relu= nn.ReLU(inplace= True)
        self.maxpool= nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)

        self.conv2= nn.Conv2d(inplanes, planes, kernel_size= 11,
                              stride= 1, padding= 5, bias= False)
        self.bn2= nn.BatchNorm2d(planes)

        self.conv3= nn.Conv2d(planes, planes, kernel_size= 11,
                              stride= 1, padding= 5, bias= False)
        self.bn3= nn.BatchNorm2d(planes)
        self.maxpool2= nn.MaxPool2d(kernel_size= 3, stride= (2,1), padding= 1)

    def forward(self, x):
        x= self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x= self.relu(self.bn2(self.conv2(x)))
        x= self.maxpool2(self.relu(self.bn3(self.conv3(x))))
        return x

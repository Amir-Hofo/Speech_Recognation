from .transformer_model import *

class CustomModel(nn.Module):
    def __init__(self, fs, d_model, n_head, num_encoders, num_decoders,
                 dim_feedforward, dropout=0.1, activation= F.relu, 
                 n_mels= 80, n_fft= 400, feature_extractor_model= 'Resnet', 
                 inplanes= 32, planes= 64):
        super().__init__()
        
        # transform
        self.transforms= nn.Sequential(T.Resample(orig_freq= fs, new_freq= 16000),
                                       T.MelSpectrogram(n_mels= n_mels, n_fft= n_fft)
                                       ).requires_grad_(False)
        
        # feature embedding
        if feature_extractor_model == 'CNN2D':
            self.cnn= CNN2DFeatureExtractor(inplanes= inplanes, planes= planes)
        elif feature_extractor_model == 'Resnet':
            self.cnn= ResnetFeatureExtractor()

        # transformer
        self.transformers= TransformerModel(d_model, n_head, num_decoders,
                                            num_decoders, dim_feedforward,
                                            dropout, activation)
        
        # classifier
        self.cls= nn.Linear(d_model, len(vocab))
        self.init_weights()

    def init_weights(self) -> None:
        initrange= 0.1
        self.cls.bias.data.zero_()
        self.cls.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src, tgt):
        with torch.no_grad():
            src= self.transforms(src)
        src= self.cnn(src)
        src= src.reshape(src.shape[0], -1, src.shape[-1])
        out= self.transformers(src.permute(1, 0, 2), tgt)
        out= self.cls(out)
        return out
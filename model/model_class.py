from .transformer_model import *

class CustomModel(nn.Module):
    def __init__(self, fs, d_model, n_head, num_encoders, num_decoders,
                 dim_feedforward, dropout= 0.1, activation= F.relu, 
                 n_mels= 80, n_fft= 400, feature_extractor_model= 'Resnet', 
                 inplanes= 32, planes= 64):
        super().__init__()
        
        # transform
        self.transforms= nn.Sequential(T.Resample(orig_freq= fs, new_freq= 16000),
                                       T.MelSpectrogram(n_mels= n_mels, n_fft= n_fft)
                                       ).requires_grad_(False)
        ''' 
        in:  src= [batch_size, audio_channel, audio_length]
        out: src= [batch_size, audio_channel, n_mels, mel_time_length]
        '''
        
        # feature embedding
        if feature_extractor_model == 'CNN2D':
            self.cnn= CNN2DFeatureExtractor(inplanes= inplanes, planes= planes)
        elif feature_extractor_model == 'Resnet':
            self.cnn= ResnetFeatureExtractor()
        ''' 
        in:         src= [batch_size, audio_channel, n_mels, mel_time_length]
        out Resnet: src= [batch_size, 64, n_mels/ 4, mel_time_length/ 4]
        out CNN2D:  src= [batch_size, planes, n_mels/ 4, mel_time_length/ 2]
        '''

        # change dimension
        ''' 
        Resnet:
        in:  src= [batch_size, 64, n_mels/ 4, mel_time_length/ 4]
        out: src= [mel_time_length/ 4, batch_size, 64*(n_mels/4)]
        CNN2D:
        in:  src= [batch_size, planes, n_mels/ 4, mel_time_length/ 2]
        out: src= [mel_time_length/ 2, batch_size, planes*(n_mels/4)]
        '''

        # transformer
        self.transformers= TransformerModel(d_model, n_head, num_encoders,
                                            num_decoders, dim_feedforward,
                                            dropout, activation)
        ''' 
        Resnet:
        in:  src= [mel_time_length/ 4, batch_size, 16* n_mels]
             tgt= [batch_size, text_length]
        out: [mel_time_length/ 4, batch_size, 64*(n_mels/4)]
        CNN2D:
        in:  src= [mel_time_length/ 2, batch_size, planes*(n_mels/4)]
             tgt= [batch_size, text_length]
        out: [mel_time_length/ 2, batch_size, planes*(n_mels/4)]
        '''
        
        # classifier
        self.cls= nn.Linear(d_model, len(vocab))
        self.init_weights()
        '''
        Resnet:
        in=  [mel_time_length/ 4, batch_size, 64*(n_mels/4)]
        out= [mel_time_length/ 4, batch_size, len(vocab)]    
        CNN2D:
        in:  [mel_time_length/ 2, batch_size, planes*(n_mels/4)]
        out: [mel_time_length/ 2, batch_size, len(vocab)]
        '''

    def init_weights(self) -> None:
        initrange= 0.1
        self.cls.bias.data.zero_()
        self.cls.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src, tgt):
        with torch.no_grad():
            src= self.transforms(src)
        src= self.cnn(src)

        batch_size, _, _, seq_len= src.shape
        src= src.reshape(batch_size, -1, seq_len)
        src= src.permute(2, 0, 1)

        out= self.transformers(src, tgt)
        src= src.permute(1, 0, 2)
        out= self.cls(out)
        return out
    
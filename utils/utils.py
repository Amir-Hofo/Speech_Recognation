from packages import *

# warning
warnings.filterwarnings("ignore")

# device
device= 'cuda' if torch.cuda.is_available() else 'cpu'
pin_memory= (device == 'cuda')

# system
def is_running_on_colab():
    return 'COLAB_GPU' in os.environ or 'COLAB_RELEASE_TAG' in os.environ
if is_running_on_colab():
    system= "colab"
else:
    system= "local"

# project path
if system == "local":
    project_path= r'./'
    dataset_path= r'./dataset/LJSpeech-1.1/'
elif system == "colab":
    project_path= r'/content/drive/MyDrive/speech_recognation/'
    dataset_path= fr'{project_path}dataset/LJSpeech-1.1/'
else:
  raise ValueError("Invalid system")

# colab mount
if system == "colab":
    from google.colab import drive
    drive.mount('/content/drive')

# number of params
def num_trainable_params(model):
  nums= sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
  return nums

# average meter
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val= 0
        self.avg= 0
        self.sum= 0
        self.count= 0

    def update(self, val, n= 1):
        self.val= val
        self.sum += val* n
        self.count+= n
        self.avg= self.sum/ self.count

# plot spectrogram
def plot_specgram(waveform, sample_rate, title= "Spectrogram"):
    waveform= waveform.numpy()
    num_channels, num_frames= waveform.shape
    figure, axes= plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes= [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs= sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
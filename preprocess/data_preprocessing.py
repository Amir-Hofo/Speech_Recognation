from utils import *
ram_status= False
batch_size= 32

metadata= pd.read_csv(fr'{dataset_path}/metadata.csv',
                      sep= '|', quoting= 3, header= None, 
                      names= ['id', 'text', 'norm_text'])

train, valid, test= random_split(metadata, lengths= [0.75, 0.1, 0.15])

train= metadata.iloc[train.indices]
valid= metadata.iloc[valid.indices]
test= metadata.iloc[test.indices]

vocab= build_vocab_from_iterator(train.norm_text.apply(lambda x: x.lower()),
                                 min_freq= 10, specials= ['=', '#', '<', '>'])
vocab.set_default_index(1)

torch.save(vocab, fr'{project_path}output/vocab.pt')
test.to_csv(fr'{project_path}output/test.csv', sep= '|', index= False)


class CustomDataset(Dataset):
    def __init__(self, dataframe, ram_status):
        self.dataframe, self.ram_status= dataframe, ram_status
        self.vocab= torch.load(fr'{project_path}output/vocab.pt')
        self.sos, self.eos= self.vocab(['<']), self.vocab(['>'])

    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, idx):
        if self.ram_status:
            audio_pack, text_pack= [], []
            for id in self.dataframe['id']:
                audio, _= torchaudio.load(fr'{dataset_path}/wavs/{id}.wav')
                text= self.dataframe.iloc[idx][2]
                text= self.vocab(list(text.lower()))
                text= torch.LongTensor(self.sos+ text+ self.eos)
                audio_pack.append(audio.squeeze())
                text_pack.append(text)
            return audio_pack, text_pack
        else:
            id= self.dataframe.iloc[idx][0]
            audio, _= torchaudio.load(fr'{dataset_path}/wavs/{id}.wav')
            text= self.dataframe.iloc[idx][2]
            text= self.vocab(list(text.lower()))
            text= torch.LongTensor(self.sos+ text+ self.eos)
            return audio.squeeze(), text


trainset= CustomDataset(train, ram_status)
validset= CustomDataset(valid, ram_status)
testset= CustomDataset(test, ram_status)

def collate_fn(batch):
    x, y= zip(*batch)
    x= pad_sequence(x, batch_first= True, padding_value= 0).unsqueeze(1)
    y= pad_sequence(y, batch_first= True, padding_value= 0)
    return x, y

train_loader= DataLoader(trainset, batch_size= batch_size, 
                         shuffle= True, collate_fn= collate_fn)
valid_loader= DataLoader(validset, batch_size= batch_size, 
                         shuffle= False, collate_fn= collate_fn)
test_loader= DataLoader(testset, batch_size= batch_size, 
                        shuffle= False, collate_fn= collate_fn)
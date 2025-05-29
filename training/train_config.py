from model import *

loss_fn= nn.CrossEntropyLoss(ignore_index= 0)
metric= WER().to(device)

model= CustomModel(fs= 22050, d_model= 1280, n_head= 2, 
                   num_encoders= 4, num_decoders= 1,
                   dim_feedforward= 640, planes= 32).to(device)

 
optimizer= optim.SGD(model.parameters(), lr= 0.001, 
                     weight_decay= 1e-4, momentum= 0.9, nesterov= True)


def postprocess(outputs, targets):
    generates, transcripts= [], []
    for output, target in zip(outputs, targets):
        g= ''.join(vocab.lookup_tokens(output.argmax(dim= -1).tolist()))
        generates.append(g)
        t= ''.join(vocab.lookup_tokens(target.tolist()))
        transcripts.append(t)
    return generates, transcripts
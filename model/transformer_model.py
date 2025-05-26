from .feature_extractor import *

# PositionalEncoding
class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, dropout: float = 0.1, 
               max_len: int = 5000):
    super().__init__()
    self.dropout= nn.Dropout(p= dropout)

    position= torch.arange(max_len).unsqueeze(1)
    div_term= torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe= torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2]= torch.sin(position* div_term)
    pe[:, 0, 1::2]= torch.cos(position* div_term)
    self.register_buffer('pe', pe)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Arguments:
        x: Tensor, shape ``[seq_len, batch_size, embedding_dim]`` """
    x= x + self.pe[:x.size(0)]
    return self.dropout(x)
  

# TransformerModel  
class TransformerModel(nn.Module):
    def __init__(self, d_model, n_head, num_encoders, num_decoders,
                 dim_feedforward, dropout, activation):
        super().__init__()
        self.d_model= d_model

        self.embedding= nn.Embedding(len(vocab), 
                                     embedding_dim= self.d_model, 
                                     padding_idx= 0)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

        self.pos_encoder= PositionalEncoding(d_model= d_model)

        self.transformer= nn.Transformer(d_model= d_model, nhead= n_head,
                                         num_encoder_layers= num_encoders,
                                         num_decoder_layers= num_decoders,
                                         dim_feedforward= dim_feedforward,
                                         dropout= dropout, activation= activation)
        

    def forward(self, src, tgt):
        tgt= self.embedding(tgt) * math.sqrt(self.d_model)
        tgt= self.pos_encoder(tgt.permute(1, 0, 2))
        tgt_mask= nn.Transformer.generate_square_subsequent_mask(len(tgt))
        out= self.transformer(src, tgt, tgt_mask= tgt_mask)
        return out
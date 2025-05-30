import torch
import torch.nn as nn
import math
from config import get_config
import warnings
from pathlib import Path

# Questions
# - What is inside embedded vector?
# - What does it mean when I split embedded vecoter?


# input embedding
# - senctence to vector
# - numbers that presented word in vocabulary

class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

# position encoding
# - PE(pos, 2i)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len: int, dropout: float) -> None:
        super().__init__() 
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create PE
        # Create a matrix of shape (sen_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (Seq_Len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)

# Add and Normalize
# - Seperately Normalize each Batch item using Layer Normalization
# - Move mean to 0, std to 1

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps:float = 10e-6): 
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # Multiplied
        self.bias  = nn.Parameter(torch.zeros(features)) # Added
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1,  keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

# Feed Forward Block

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1, B1
        self.dropout  = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2, B2
    
    def forward(self, x):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

# Multi Head Attention
# - Input clone into 3 tensor, Q K V
# - Q K V multiply by seperate weight d_model * d_model
#   - d_model: size of embedding vector (512 ?)
#   - seq: sequence length
# - Q K V split into h amount of tensor as head
#   - h: amount of head
#   - d_k/d_v: dimention of each head
#   - d_k = d_model / h
# - Multi head calculate attention
#   - Multi Head Attention Formular
# - Multi head concat back to dimision before splited (seq, h * d_v) = (seq, d_model)
#   - Multiply a weight d_model * d_model
#   - output
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, " d_model does not divide by h"

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    # Attention calculation
    # Formular : softmax( Q @ transport(K) / sqrt(d_k) ) @ V
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout2d):
        d_k = query.shape[-1]
        # (Batch, h, Seq_Len, d_k) --> (Batch, h, Seq_Len, Seq_Len)
        attention_scores = (query @ key.transpose(-2, -1) / math.sqrt(d_k))
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        # A Sequence by Sequence Matrix, value of word related to other words
        # softmax( Q @ transport(K) / sqrt(d_k) )
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, seq_Len, seq_Len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # Value matrix multiply by attention_scores
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # Note, meaning of Q, K, V in Database's Vector Similarity Search
        # Query : search target, search vector
        # Key   : vector saved in database, calculate with Query to find Similarity
        # Value : Orignal data that related to Key
        
        # Multi Head Attention = Word Meaning Relationship between "one word in the sequence" and "all other word in the sequence"
        # - Relationship can but different across all heads (grammer relation, position relation, )
        # - Using 1 head will lower the preformance significanty (Experiment by orignal paper)

        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        query = self.w_q(q)
        key   = self.w_k(k)
        value = self.w_v(v)

        # Each head contains full sequence information, but each word only contains part of embedding information (split into h groups)
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, h, d_k) --> (Batch, h, Seq_Len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key   = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, Seq_Len, d_k) --> (Batch, Seq_Len, h, d_k) --> (Batch, Seq_Len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, h, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

# Building block of Encoder
class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_conection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        # src_mask mask word itself
        x = self.residual_conection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_conection[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# Building Block of Decoder
class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocal_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocal_size)
    
    def forward(self, x):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, vocab_size)
        #return torch.log_softmax(self.proj(x), dim = -1)
        return self.proj(x)

# Transformer Model
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_embed: InputEmbedding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)
    
# build a transformer for traslate task
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048):
    # Create Embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block  = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create Projection Layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, src_pos, tgt_embed, tgt_pos, projection_layer)

    # Initialize in parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

def test_model_forward():
    # build data
    config = get_config()
    B, seq_len, d_model = 2, config['seq_len'], config['d_model']
    src_vocab_size = 100
    tgt_vocab_size = 120

    # build model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True) # Parents: create all parent folder, Exist OK: ignore error when folder is already existed
    model = build_transformer(src_vocab_size, tgt_vocab_size, seq_len, seq_len, d_model).to(device)

    encoder_input = torch.tensor([1] * seq_len).to(device)
    encoder_mask  = torch.tensor([1] * seq_len).to(device)
    model.encode(encoder_input, encoder_mask)

if __name__ == "__main__":
    test_model_forward()

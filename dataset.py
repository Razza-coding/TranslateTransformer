import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from tokenizer_symbol import *

from pathlib import Path
import ast

from preview import preview_raw_dataset
from config import get_config

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len  = seq_len

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id(BOS_TOKEN)], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id(EOS_TOKEN)], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id(PAD_TOKEN)], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index: any) -> any:
        # Get src and tgt data pair from dataset
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Tokenizer encoding
        enc_input_tokens = self.tokenizer_src.encode(src_text, add_special_tokens=False)
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text, add_special_tokens=False)

        # Fills up sentence to reach fixed sequence length
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # leave 2 space for [SOS] [EOS]
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # leave 1 space for [SOS], Transformer decoder already have [EOS]

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")
        
        # Build input and label tensor
        # Add SOS EOS to the sorce text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype= torch.int64)
            ]
        )

        # Add SOS to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype= torch.int64)
            ]
        )

        # Add EOS to the label (what is expected as output from encoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype= torch.int64)
            ]
        )        

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        data_item = {
            "encoder_input": encoder_input, # (Seq_Len)
            "decoder_input": decoder_input, # (Seq_Len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, Seq_Len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, Seq_Len) & (1, Seq_Len, Seq_Len)
            "label":    label,    # (Seq_Len)
            "src_text": src_text, #
            "tgt_text": tgt_text  #
        }

        #print(data_item)

        return data_item
    
def causal_mask(size):
    mask = torch.triu( torch.ones(1, size, size), diagonal=1 ).type(torch.int)
    return mask == 0

def get_or_build_tokenizer(config, ds, lang):
    sub_folder = "{0}-{1}".format(config['lang_src'], config['lang_tgt'])
    tokenizer_path = Path(config['tokenizer']['tokenizer_file'].format(sub_folder, lang))
    
    if config['tokenizer']['use_common_tokenizer'] is not None:
        # use a common pretrained tokenizer           
        tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")           
        tokenizer.src_lang = "en_XX"
        tokenizer.tgt_lang = "zh_CN"

        # link to function
        tokenizer.token_to_id = tokenizer.convert_tokens_to_ids
        tokenizer.get_vocab_size = lambda: tokenizer.vocab_size

        tokenizer.save_pretrained(str(tokenizer_path))
    elif not Path.exists(tokenizer_path):
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        # Create a word level tokenizer
        # build tokenizer ourself
        # Whitespace as seperater, and one word per token
        # Replace word can not find in vocabulary as [UNK]
        # [SOS] [EOS] start and end of sentence
        # [PAD] fill up sentence to reach fixed input length in training stage
        # Minimum Frequency is 2, word must appear 2 times before adding in vocabulary
        tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOKEN)) # unknown
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=[UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN], min_requency=2)
        # Make generator
        def get_all_sentences(ds, lang):
            for item in ds:
                yield item['translation'][lang]
        # Get item from dataset, train tokenizer
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        # save tokenizer
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def load_opus_books(config):
    '''
    Load Opus Books dataset form hugging face
    '''
    return load_dataset(config["datasource"], f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

def load_LLaMAX_Ted(config):
    '''
    Load LLaMAX/BenchMAX_General_Translation Ted dataset form hugging face
    '''
    src_data = load_dataset(config["datasource"], f'ted_{config["lang_src"]}')['train']
    tgt_data = load_dataset(config["datasource"], f'ted_{config["lang_tgt"]}')['train']

    # transform format
    format_data = []
    for idx in range(len(src_data)):
        s = src_data[idx]
        t = tgt_data[idx]

        id_s = s['id']
        id_t = t['id']
        assert id_s == id_t, "Dataset item id doesn't match"

        text_s = s['text']
        text_t = t['text']
        format_item = {
            "id" : id_s,
            "translation":{
                config["lang_src"] : text_s,
                config["lang_tgt"] : text_t
            }
        }
        format_data.append(format_item)
    return format_data

def load_wmt20_en_zh(config):
    '''
    Load yezhengli9/wmt20-en-zh from hugging face
    '''
    ds = load_dataset(config["datasource"], split="train")

    assert len(ds.column_names) >= 2

    old_id_key = ds.column_names[0]
    old_translation_key = ds.column_names[1]

    ds = ds.rename_column(old_id_key, 'id')
    ds = ds.rename_column(old_translation_key, 'translation')

    format_dataset = []
    for item in ds:
        id = item['id']
        translate = eval(item['translation'])
        format_dataset.append({
            'id': id,
            'translation': translate
        })
        
    return format_dataset

def get_ds(config):
    '''
    Make Dataset Loader and Tokenizer
    '''
    # Download dataset 
    if 'opus_books' == config["datasource"]:
        ds_raw = load_opus_books(config)
    elif 'LLaMAX/BenchMAX_General_Translation' == config["datasource"]:
        ds_raw = load_LLaMAX_Ted(config)
    elif 'yezhengli9/wmt20-en-zh' == config["datasource"]:
        ds_raw = load_wmt20_en_zh(config)
    else:
        ds_raw = None
    
    assert ds_raw != None, "No Dataset Loaded"

    # Preview Raw Data
    preview_raw_dataset(ds_raw, 5)

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # 90% train 10% validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds   = BilingualDataset(val_ds_raw,   tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']])
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']])
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(r"Max Src Len: {}", max_len_src)
    print(r"Max Tgt Len: {}", max_len_tgt)

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader   = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

if __name__ == "__main__":
    device = 'cpu'
    config = get_config()
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    for idx, batch in enumerate(train_dataloader):
        if idx == 10:
            break

        encoder_text = batch["src_text"][0]
        decoder_text = batch["tgt_text"][0]
        label = batch["label"][0]

        encoder_input = batch['encoder_input'][0].to(device)
        decoder_input = batch['decoder_input'][0].to(device)

        encoder_mask = batch['encoder_mask'][0].to(device) # (B, 1, 1, Seq_Len)
        decoder_mask = batch['decoder_mask'][0].to(device) # (B, 1, Seq_Len, Seq_Len)        

        encoder_ids = encoder_input.tolist()
        decoder_ids = decoder_input.tolist()

        encoder_tokens = tokenizer_src.convert_ids_to_tokens(encoder_ids, skip_special_tokens=False)
        decoder_tokens = tokenizer_tgt.convert_ids_to_tokens(decoder_ids, skip_special_tokens=False)

        encoder_decoded = tokenizer_src.decode(encoder_ids, skip_special_tokens=True)
        decoder_decoded = tokenizer_tgt.decode(decoder_ids, skip_special_tokens=True)
        label_decoded   = tokenizer_tgt.decode(label, skip_special_tokens=True)

        # 檢查是否有 [UNK]
        unk_src = "[UNK]" in encoder_tokens
        unk_tgt = "[UNK]" in decoder_tokens

        print("-" * 40)
        print(f"[Batch {idx}]")
        print(f"Encoder Input Org  Text : {encoder_text}")
        #print(f"Encoder Input Token IDs : {encoder_ids}")
        #print(f"Encoder Tokens          : {encoder_tokens}")
        print(f"Encoder Decoded Text    : {encoder_decoded}")
        print(f"Encoder Mask            : {encoder_mask}")
        #print(f"❗ [UNK] in Encoder?     : {unk_src}")
        print()

        print(f"Decoder Input Org  Text : {decoder_text}")
        #print(f"Decoder Input Token IDs : {decoder_ids}")
        #print(f"Decoder Tokens          : {decoder_tokens}")
        print(f"Decoder Decoded Text    : {decoder_decoded}")
        print(f"Decoder Mask            : {decoder_mask}")
        #print(f"❗ [UNK] in Decoder?     : {unk_tgt}")
        print()

        print(f"Label Decode Text       : {label_decoded}")
        print("-" * 40)
from pathlib import Path
from config import get_config, get_latest_weight_filepath, get_weight_file_path
from model import build_transformer
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset import BilingualDataset, get_or_build_tokenizer
from tokenizer_symbol import *
import torch
import sys

def translate():
    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config = get_config()
    # tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    # tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    tokenizer_src = tokenizer_tgt = get_or_build_tokenizer(config)
    model = build_transformer(tokenizer_src.vocab_size, tokenizer_tgt.vocab_size, config["seq_len"], config['seq_len'], d_model=config['d_model']).to(device)

    # Load the pretrained weights
    if config["preload"] != "latest":
        model_filename = get_weight_file_path(config, config["preload"])
    else:
        model_filename = get_latest_weight_filepath(config)
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])
    seq_len = config['seq_len']

    # translate the sentence
    print()
    print()
    print("Enter your sentence")
    model.eval()
    with torch.no_grad():
        while True:
            sentence = input(">> ").strip()
            if sentence.lower() == "exit":
                print("再見！")
                break
            elif sentence == "":
                continue
            else:
                # Precompute the encoder output and reuse it for every generation step
                source = tokenizer_src.encode(sentence, add_special_tokens=False)
                source = torch.cat([
                    torch.tensor([tokenizer_src.convert_tokens_to_ids(BOS_TOKEN)], dtype=torch.int64), 
                    torch.tensor(source, dtype=torch.int64),
                    torch.tensor([tokenizer_src.convert_tokens_to_ids(EOS_TOKEN)], dtype=torch.int64),
                    torch.tensor([tokenizer_src.convert_tokens_to_ids(PAD_TOKEN)] * (seq_len - len(source) - 2), dtype=torch.int64)
                ], dim=0).to(device).unsqueeze(0)
                source_mask = (source != tokenizer_src.convert_tokens_to_ids(PAD_TOKEN)).unsqueeze(0).unsqueeze(0).int().to(device)
                encoder_output = model.encode(source, source_mask)

                # Initialize the decoder input with the sos token
                decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.convert_tokens_to_ids(BOS_TOKEN)).type_as(source).to(device)

                # Print the source sentence and target start prompt
                print(f"{f'SOURCE : ':>12}{sentence}")

                # Generate the translation word by word
                every_decode_word = []
                while decoder_input.size(1) < seq_len:
                    # build mask for target and calculate output
                    decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
                    out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

                    # project next token
                    prob = model.project(out[:, -1])
                    _, next_word = torch.max(prob, dim=1)
                    decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

                    # print the translated word
                    every_decode_word.append(f"{tokenizer_tgt.decode([next_word.item()])}")

                    # break if we predict the end of sentence token
                    if next_word == tokenizer_tgt.convert_tokens_to_ids(EOS_TOKEN):
                        break
                respond = tokenizer_tgt.decode(decoder_input[0].tolist(), skip_special_tokens=True)
                print(f"{f'PREDICTED : ':>12}{respond}")
                print(f"{f'SINGLE WORD : ':>12}{every_decode_word}")
                
                

    # convert ids to tokens
    return 

if __name__ == "__main__":
    translate()
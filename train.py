import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from tokenizers import Tokenizer
import torchmetrics

from pathlib import Path
from tqdm import tqdm
import warnings

from model import build_transformer
from dataset import BilingualDataset, causal_mask, get_ds
from config import get_config, get_weight_folder, get_weight_file_path, get_latest_weight_filepath
from tokenizer_symbol import *

def greedy_decode(model, source, source_mask, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, max_len, device):
    sos_idx = tokenizer_tgt.convert_tokens_to_ids(BOS_TOKEN)
    eos_idx = tokenizer_tgt.convert_tokens_to_ids(EOS_TOKEN)

    # Precompute encoder output and reuse it every iteration of decoding
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder input with sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        # Build mask for the target (encoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token
        prob = model.project(out[:,-1])

        # Select the token with probability (greedy method)
        _, next_word = torch.max(prob, dim=1)
        decoder_input= torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, max_len, device, print_msg, global_step, writer: SummaryWriter, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    # Size of the control window
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask  = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            output_ids  = model_output.detach().cpu().numpy()
            model_out_text = tokenizer_tgt.decode(output_ids, skip_special_tokens=True)

            label   = batch['label'][0]
            # label_decoded   = tokenizer_tgt.decode(label, skip_special_tokens=True)

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print to console
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE    : ':>12}{source_text}")
            print_msg(f"{f'TARGET    : ':>12}{target_text}")
            print_msg(f"{f'PREDICTED : ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()
        pass

# Build a transformer according to the size of input
def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device {device}')

    Path(get_weight_folder(config)).mkdir(parents=True, exist_ok=True) # Parents: create all parent folder, Exist OK: ignore error when folder is already existed

    # get tokenizer and create model
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.vocab_size, tokenizer_tgt.vocab_size).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    # load previous progress
    preload_model_path = None if config['preload'] is None else get_latest_weight_filepath(config) if config['preload'] == "latest" else get_weight_file_path(config, config['preload'])
    if preload_model_path:
        print(f'Preloading model: {preload_model_path}')
        state = torch.load(preload_model_path)
        model.load_state_dict(state_dict=state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print("No preload model set, starting from epoch 0")
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.convert_tokens_to_ids(PAD_TOKEN), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epoch']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch: 02d}')
        for batch in batch_iterator:
            torch.cuda.empty_cache()
            encoder_input = batch['encoder_input'].to(device) # (B, Seq_Len)
            decoder_input = batch['decoder_input'].to(device) # (B, Seq_Len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, Seq_Len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, Seq_Len, Seq_Len)

            # Run Tensor through Transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, Seq_Len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, Seq_Len, d_model)
            proj_output = model.project(decoder_output) # (B, Seq_Len, tgt_vocab_size)

            label = batch['label'].to(device) # (B, Seq_Len)

            # (B, Seq_Len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.vocab_size), label.view(-1))
            batch_iterator.set_postfix({f'loss': f'{loss.item(): 0.2f}'})

            # log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            # Update the weight
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            # free memory
            del loss, proj_output, decoder_output, encoder_output
            torch.cuda.empty_cache()           
        
        # Validation every epoch
        with torch.no_grad():
            run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        
        # Save the model at the end of every epoch
        if epoch % int(config['save_every_n_epoch']) == 0:
            model_filename = get_weight_file_path(config, f'{epoch:03d}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
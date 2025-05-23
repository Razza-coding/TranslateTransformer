import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset

from torch.utils.tensorboard import SummaryWriter

from model import build_transformer
from dataset import BilingualDataset, causal_mask

from config import get_config, get_weight_folder, get_weight_file_path, get_latest_weight_filepath

from datasets import load_dataset # Hugging Face
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import torchmetrics

from pathlib import Path
from tqdm import tqdm

import warnings

# This python contains code for preview data
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def preview_raw_dataset(data_source: str, category: str, split_set=None) -> None:
    # Download dataset form hugging face
    if split_set:
        ds_raw = load_dataset(data_source, category, split_set)
    else:
        ds_raw = load_dataset(data_source, category)['train']
    #ds_raw = load_dataset(config["datasource"], f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    # 90% train 10% validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = Subset(ds_raw, range(train_ds_size)), Subset(ds_raw, range(train_ds_size, train_ds_size + val_ds_size))

    print("Train Set:")
    for idx, item in enumerate(train_ds_raw):
        if idx + 1 > 3:
            break
        print(f"[ Item {idx + 1} ]")
        for k in item:
            print(f"\t{k:<10} : {item[k]}")
        
    print("Validation Set:")
    for idx, item in enumerate(val_ds_raw):
        if idx + 1 > 3:
            break
        print(f"[ Item {idx + 1} ]")
        for k in item:
            print(f"\t{k:<10} : {item[k]}")


# Build a transformer according to the size of input
def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

if __name__ == "__main__":
    warnings.filterwarnings('ignore')   

    preview_raw_dataset("LLaMAX/BenchMAX_General_Translation", "ted_en")
    preview_raw_dataset("LLaMAX/BenchMAX_General_Translation", "ted_zh")
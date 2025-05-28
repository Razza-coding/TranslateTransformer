from pathlib import Path
import os

def get_config():
    return {
        "batch_size": 4,
        "num_epoch": 50,
        "save_every_n_epoch": 10,
        "lr": 10**-4,
        "seq_len": 300,
        "d_model": 256,
        "datasource": "yezhengli9/wmt20-en-zh",
        "lang_src": "en",
        "lang_tgt": "zh",
        "preload": "50",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "tokenizer":{
            "use_common_tokenizer" : "facebook/mbart-large-50-many-to-many-mmt",
            "tokenizer_file": "tokenizer/{0}"
        },
        "experiment_name": "runs/tmodel"
    }

def get_weight_folder(config):
    return os.path.join(os.path.abspath("."), config["model_folder"], config["datasource"])

def get_weight_file_path(config, epoch: str):
    model_weight_folder = get_weight_folder(config)
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return os.path.join(Path(model_weight_folder), model_filename)

def get_latest_weight_filepath(config):
    model_weight_folder = get_weight_folder(config)
    model_basename = config['model_basename']
    weight_files_list = list(Path(model_weight_folder).glob(f'{model_basename}*.pt'))
    if len(weight_files_list) == 0:
        return None
    else:
        weight_files_list.sort()
        return weight_files_list[-1]

def test_config_function():
    print("[ Test Config Function ]")
    config = get_config()
    epoch  = 7

    print(f"Get weight folder: {get_weight_folder(config)}")
    print(f"Get preload weight path: {get_weight_file_path(config, f'{epoch:02d}')}")
    print(f"Get latest  weight path: {get_latest_weight_filepath(config)}")    

if __name__ == "__main__":
    test_config_function()
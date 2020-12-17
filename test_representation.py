from run_script import seed_everything
import pytorch_lightning as pl 
from setup import process_config
import os
import random, torch, numpy
from model import BertEthicsFinetuner
from data_module import EthicsDataModule
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from lin_eval import EthicsEval

def run(checkpoint, config_path, gpu_device=None):
    config = process_config(config_path)
    if not gpu_device:
        gpu_device = config.gpu_device
    seed_everything(config.seed, use_cuda=config.cuda)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        save_top_k=-1, # could be 5, and that would work fine
        period=1,
    )
    
    num_epochs = config.num_epochs
        
    wandb_logger = pl.loggers.WandbLogger(name="testing " + checkpoint, config=config, project=config.exp_name)
    # if we are at 100 epochs pretraining, run for 100 epochs 
    # we are using the same transformations as we did in the pretraining task, but this time for regularization etc   
    dm = EthicsDataModule(batch_size=config.optim_params.batch_size, num_workers=config.num_workers, only_short=config.only_short)     
    model = EthicsEval(config.optim_params.learning_rate, config.checkpoint_path)

    trainer = pl.Trainer(
        default_root_dir=config.exp_dir,
        gpus=[gpu_device],
        max_epochs=num_epochs,
        checkpoint_callback=ckpt_callback,
        resume_from_checkpoint=config.continue_from_checkpoint,
        logger=wandb_logger
    )

    trainer.fit(model, dm)
    results = trainer.test()
    print(results[0]['test_acc'])
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='lineval.json')
    parser.add_argument('checkpoint', type=str, default=None)
    parser.add_argument('--gpu-device', type=int, default=0)
    args = parser.parse_args()
    run(
        checkpoint=args.checkpoint, 
        config_path=args.config, 
        gpu_device=args.gpu_device, 
    )
import os
import random, torch, numpy
from setup import process_config
import pytorch_lightning as pl
from model import BertEthicsFinetuner
from data_module import EthicsDataModule
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from lin_eval import EthicsEval
# import subprocess not for now at least, there's no pretraining here
import subprocess

class RealTimeEvalCallback(pl.Callback):
    def __init__(self, checkpoint_dir, downstream_task_config, vocab_sz=100, parent_config=None, d2=None):
        self.checkpoint_dir = checkpoint_dir
        self.downstream_task_config = downstream_task_config
        self.vocab_sz = str(vocab_sz)
        self.parent_config = parent_config
        self.commands1 = lambda cur_checkpoint: ["python", "test_representation.py", self.downstream_task_config, cur_checkpoint, 
                        "--gpu-device", "4"]

    def on_fit_end(self, trainer, pl_module):
        cur = trainer.current_epoch
        cur_checkpoint = os.path.join(self.checkpoint_dir, "epoch="+str(cur)+".ckpt")
        commands = self.commands1(cur_checkpoint) + ["-l"]
        subprocess.Popen(commands)

    def on_epoch_end(self, trainer, pl_module):
        cur = trainer.current_epoch
        if cur % 10 == 1:
            cur_checkpoint = os.path.join(self.checkpoint_dir, "epoch="+str(cur)+".ckpt")
            subprocess.Popen(self.commands1(cur_checkpoint))

def run(config_path, gpu_device=None):
    config = process_config(config_path)
    seed_everything(config.seed, use_cuda=config.cuda)
    dm = EthicsDataModule(batch_size=config.optim_params.batch_size, num_workers=config.num_workers, only_short=config.only_short)     
    wandb_logger = pl.loggers.WandbLogger(name=config.run_name, project=config.exp_name)
            
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        save_top_k=-1, # could be 5, and that would work fine
        period=1,
    )
    model = BertEthicsFinetuner(config.optim_params.learning_rate)
    eval_realtime_callback = RealTimeEvalCallback(checkpoint_dir=config.checkpoint_dir, downstream_task_config=config.downstream_task_config, parent_config=config_path) #mtype refers to final task

    trainer = pl.Trainer(
        default_root_dir=config.exp_dir,
        gpus=[gpu_device],
        max_epochs=config.num_epochs,
        callbacks=[eval_realtime_callback],
        checkpoint_callback=ckpt_callback,
        resume_from_checkpoint=config.continue_from_checkpoint,
        logger=wandb_logger
    )
    trainer.fit(model, dm)

    trainer.test()


def seed_everything(seed, use_cuda=True):
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda: torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='config/jeopardy_model.json')
    parser.add_argument('--gpu-device', type=int, default=None)
    args = parser.parse_args()
    run(args.config, gpu_device=args.gpu_device)


# class RealTimeEvalCallback(pl.Callback):
#     def __init__(self, checkpoint_dir, downstream_task_config, vocab_sz=100, parent_config=None, d2=None):
#         self.checkpoint_dir = checkpoint_dir
#         self.downstream_task_config = downstream_task_config
#         self.vocab_sz = str(vocab_sz)
#         self.parent_config = parent_config
#         self.downstream2=d2
#         self.commands1 = lambda cur_checkpoint: ["python", "test_representation.py", self.downstream_task_config, cur_checkpoint, 
#                         self.vocab_sz, self.parent_config, "--gpu-device", "1"]
#         self.commands2 = lambda cur_checkpoint: ["python", "test_representation.py", self.downstream2, cur_checkpoint,
#                         self.vocab_sz, self.parent_config, "--gpu-device", "2"]    

#     def on_fit_end(self, trainer, pl_module):
#         cur = trainer.current_epoch
#         cur_checkpoint = os.path.join(self.checkpoint_dir, "epoch="+str(cur)+".ckpt")
#         commands = self.commands1(cur_checkpoint) + ["-l"]
#         subprocess.Popen(commands)

#     def on_epoch_end(self, trainer, pl_module):
#         cur = trainer.current_epoch
#         if cur % 10 == 1:
#             cur_checkpoint = os.path.join(self.checkpoint_dir, "epoch="+str(cur)+".ckpt")
#             subprocess.Popen(self.commands1(cur_checkpoint))
#             if self.downstream2:
#                 subprocess.Popen(self.commands2(cur_checkpoint))
                
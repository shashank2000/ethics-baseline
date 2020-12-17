import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import csv


TRAIN_PATH = 'ethics/commonsense/cm_train.csv'
VAL_PATH = 'ethics/commonsense/cm_test.csv'
TEST_PATH = 'ethics/commonsense/cm_test_hard.csv'

class EthicsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, num_workers=8, only_short=False):
        super().__init__()

        # we have the csv files we want, all we need to do is convert from the csv format to an array consisting of (sentence, label) tuples
        self.train_dataset, self.test_dataset, self.val_dataset = self.get_ds_from_csv(TRAIN_PATH, only_short), self.get_ds_from_csv(TEST_PATH, only_short), self.get_ds_from_csv(VAL_PATH, only_short)

    def get_ds_from_csv(self, file_path, only_short=False):
        final_ds = []
        with open (file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if only_short:
                    if row['is_short']:
                        final_ds.append((row['input'], int(row['label'])))
                else:
                    # always append
                    final_ds.append((row['input'], int(row['label'])))
        return final_ds

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=16, num_workers=8, pin_memory=True, drop_last=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=16, num_workers=8, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=16, num_workers=8, pin_memory=True, drop_last=True)
import torch, os, multiprocessing
import torch.nn.functional as F
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as plm

from torch.utils.data import DataLoader 

from data import collate_batch

class LMFinetuner(pl.LightningModule):

    def __init__(self, model, tokenizer, learning_rate, batch_size, train_dataset, val_datasets, data_args, freeze_backend):
        super(LMFinetuner, self).__init__()
        
        self.lm = model
        if freeze_backend:
            self.freeze_backend()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_datasets = val_datasets
        self.data_args = data_args

    def freeze_backend(self):
        for param in self.lm.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        # fwd
        return self.lm(**inputs)

    def training_step(self, batch, batch_nb):        
        # fwd
        loss = self.forward(batch)[0]
        # logs
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, val_batch, batch_idx, dataset_idx):
        # fwd
        values = self.forward(val_batch)
        # gather values
        loss = values[0]
        y_hat = values[1]
        label = val_batch['labels']
        # f1
        a, y_hat = torch.max(y_hat, dim=1)
        dataset_name = self.val_datasets[dataset_idx].name

        return {'val_loss': loss, f'{dataset_name}-val_y_hat': y_hat, f'{dataset_name}-label': label}

    def validation_end(self, outputs):
        scores = {}
        for dataset_outputs in outputs:
            # Assuming all val steps returned the same keys
            keys = dataset_outputs[0].keys()
            y_hat_key = list(filter(lambda x: 'y_hat' in x, keys))[0]
            label_key = list(filter(lambda x: 'label' in x, keys))[0]
            dataset_name = y_hat_key.split('-')[0]
            
            y_hats = torch.cat([x[y_hat_key] for x in dataset_outputs if y_hat_key in x])
            labels = torch.cat([x[label_key] for x in dataset_outputs if label_key in x])
            scores[f'avg _{dataset_name}_f1'] = plm.f1_score(y_hats, labels)

        avg_val_loss = torch.stack([x['val_loss'] for el in outputs for x in el]).mean()
        scores['avg_val_loss'] = avg_val_loss
        return {'log': scores}
    
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.learning_rate)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=self.data_args.shuffle, num_workers=(self.data_args.num_workers or multiprocessing.cpu_count()), batch_size=self.batch_size, collate_fn=collate_batch, pin_memory=True if torch.cuda.is_available and torch.cuda.device_count() > 1)
    
    def val_dataloader(self):
        return [              
            DataLoader(val_dataset, num_workers=(self.data_args.num_workers or multiprocessing.cpu_count()), batch_size=self.batch_size, collate_fn=collate_batch, pin_memory=True if torch.cuda.is_available() and torch.cuda.device_count() > 1) for val_dataset in self.val_datasets
        ]

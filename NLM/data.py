import os, torch, tqdm
import numpy as np

def collate_batch(batch):
    # batch same keys assuming all batches have the same keys   
    return {key : torch.tensor([example[key] for example in batch]) for key in batch[0].keys()}

class ParaphraseDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, tokenizer, task_name=None, name="spinbot_wiki_train"):
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.name = name
        self.path_list = self.prepare_links(data_dir)
        
    def prepare_links(self, data_dir):
        path_list = []

        for ext in ['og', 'mg']:
            for filename in os.listdir(os.path.join(data_dir, ext)):
                path_list.append(os.path.join(data_dir, ext, filename))
                    
        return path_list

    def __getitem__(self, idx):
        path = self.path_list[idx]
        if 'SPUN' in path:
            label = 1
        elif 'ORIG' in path:
            label = 0
        else:
            raise ValueError('No Label for Example')
        with open(path, "r", encoding="'iso-8859-1'") as f:
            text = f.read()
        
        tokens = self.tokenizer(
            text,
            max_length=self.tokenizer.max_len,
            padding="max_length",
            truncation=True,
        )
        
        tokens['labels'] = torch.tensor(label)
        return tokens

    def __len__(self):
        return len(self.path_list)
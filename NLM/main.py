import os, sys, argparse, logging, torch
import pytorch_lightning as pl

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, HfArgumentParser

from data import ParaphraseDetectionDataset
from args import ModelArguments, ParaphraseDataTrainingArguments
from model import LMFinetuner

from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning import loggers as pl_loggers

EVAL_NAMES = [
    "spinbot_wiki",
    "spinbot_thesis",
    "spinbot_arxiv",
    "spinnerchief_arxiv_2w",
    "spinnerchief_arxiv_4w",
    "spinnerchief_thesis_2w",
    "spinnerchief_thesis_4w",
    "spinnerchief_wiki_2w",
    "spinnerchief_wiki_4w"
]

EVAL_PATHS = [
    "automated_evaluation_up/spinbot/corpus/paragraphs/wikipedia_paragraph",
    "automated_evaluation_up/spinbot/corpus/paragraphs/thesis_paragraph",
    "automated_evaluation_up/spinbot/corpus/paragraphs/arxiv_paragraph",
    "automated_evaluation_up/spinnerchief/corpus/paragraphs/arxiv_paragraphs_2w",
    "automated_evaluation_up/spinnerchief/corpus/paragraphs/arxiv_paragraphs_4w",
    "automated_evaluation_up/spinnerchief/corpus/paragraphs/thesis_paragraphs_2w",
    "automated_evaluation_up/spinnerchief/corpus/paragraphs/thesis_paragraphs_4w",
    "automated_evaluation_up/spinnerchief/corpus/paragraphs/wikipedia_paragraphs_2w",
    "automated_evaluation_up/spinnerchief/corpus/paragraphs/wikipedia_paragraphs_4w",
]

TRAIN_PATH = "automated_evaluation_up/spinbot/corpus/paragraphs/wikipedia_paragraphs_train"

def main():
    parser = HfArgumentParser((ModelArguments, ParaphraseDataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=2,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=data_args.model_max_length
    )
    
    language_model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if data_args.neptune_logging:
        neptune_logger = NeptuneLogger(
            project_name=os.environ['NEPTUNE_PROJECT'],
            experiment_name=model_args.config_name if model_args.config_name else model_args.model_name_or_path
        )

    train_dataset = ParaphraseDetectionDataset(data_dir=os.path.join(data_args.data_dir, TRAIN_PATH), tokenizer=tokenizer, task_name="paraphrase_detection")
    val_datasets= [              
        ParaphraseDetectionDataset(data_dir=os.path.join(data_args.data_dir, EVAL_PATH), tokenizer=tokenizer, name=EVAL_NAME) for (EVAL_PATH, EVAL_NAME) in zip(EVAL_PATHS, EVAL_NAMES)
    ]
    
    model = LMFinetuner(language_model, tokenizer, training_args.learning_rate, model_args.batch_size, train_dataset, val_datasets, data_args, freeze_backend=False)
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(training_args.output_dir, model_args.model_name_or_path))

    trainer = pl.Trainer(
        # auto_lr_find=True,
        # auto_scale_batch_size=True,
        max_epochs=int(training_args.num_train_epochs),
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        weights_save_path=training_args.output_dir,
        gpus=torch.cuda.device_count(),
        precision=16 if training_args.fp16 and torch.cuda.is_available() else 32,
        distributed_backend='ddp' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None,
        progress_bar_refresh_rate=training_args.logging_steps,
        logger=[neptune_logger, tb_logger] if data_args.neptune_logging else tb_logger,
    )    
    trainer.fit(model)
    model.lm.save_pretrained(os.path.join(training_args.output_dir, model_args.model_name_or_path))

if __name__ == '__main__':
    main()

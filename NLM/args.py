from dataclasses import dataclass, field
        
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: str = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: str = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: str = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    batch_size: int = field(
        default=8, metadata={"help": "Which batch size to use"}
    )

@dataclass
class ParaphraseDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={"help": "The name of the task to use: wiki, arxiv, thesis."})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    num_workers: int  = field(
        default=None, metadata={"help": "How many workers used for data loaders"}
    )
    shuffle: bool = field(
        default=True, metadata={"help": "Whether to shuffle the training set or not"}
    )
    neptune_logging: bool = field(
        default=False, metadata={"help": "Use neptune to log to API or not"}
    )
    def __post_init__(self):
        self.task_name = self.task_name.lower()

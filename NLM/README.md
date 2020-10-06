# Training, and Report

## Download and prepare dataset
For this step you need the packages wget, unzip and tar.
```sh
sh prepare_data.sh /path/to/store/dataset
```

## Installing the requirements

```sh
pip install -r requirements.txt
````

## Training/Validation

> The script will will use all $CUDA_VISIBLE_DEVICES by default.


python3 main.py [options] 

  Options:

        --model_name_or_path: 
          Name of the model from https://huggingface.co or path to a pretrained model with config.json and pytorch_model.bin.

        --output_dir: 
          Path to the output directory for logging and saving checkpoints.

        --data_dir: 
          Directory where the dataset was downloaded and extracted to.
          If you used prepare_data.sh it should be /path/to/store/dataset

        --num_train_epochs: 
          Number of training epochs.

        --logging_steps:
          Log every n steps
        
        --batch_size:
          Batch size to use for training and evaluation

        --learning_rate:
          Whether to evaluate or not.

        --fp16: [Optional]
          Whether to use floating point 16 bit training or not.
          Requires torch >= 1.6.0 or NVIDIA apex installed.

        --model_max_length [Optional]
          Maximum sequence length of the model. Default will take the default sequence length


        --neptune_logging: [Optional]
          Whether to log to neptune.ai or not. Needs environment variables $NEPTUNE_PROJECT and $NEPTUNE_API_TOKEN

        --num_workers: [Optional]
          How many CPUs to use for the dataloader. Default goes to the number of available host cpus.


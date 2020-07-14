import torch
from torchvision import datasets, transforms
from pathlib import Path

# training
dataset = datasets.MNIST
normal_classes = [1]
input_rescaling = lambda x: (x - .5) * 2.
input_rescaling_inv = lambda x: x * .5 + .5
input_limits = (input_rescaling(0.), input_rescaling(1.))
default_transform = transforms.Compose([transforms.ToTensor(), input_rescaling])

# evaluation
positive_is_anomaly = False

# Reproducability
seed = 313  # Random seed to use

# data
# data loader
train_batch_size = 256
validation_batch_size = 256
test_batch_size = 256
dataloader_num_workers = 4
dataloader_pin_memory = True
dataloader_shuffle = True
dataloader_drop_last = True

# data I/O
log_data_feed_loop_interval = 8

output_root = './output'
data_dir = '.' + '/data'  # Location for the dataset
Path(data_dir).mkdir(parents=True, exist_ok=True)

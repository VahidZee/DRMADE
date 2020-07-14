from torch.utils.data import Dataset, Subset, DataLoader
import src.config as config
import numpy as np



class DatasetSelection(Dataset):
    def __init__(self,
                 dataset,
                 root=config.data_dir,
                 train=True,
                 classes=None,
                 transform=config.default_transform,
                 target_transform=lambda x: x,
                 download=True,
                 return_indexes=False
                 ):
        self.transform = transform
        self.target_transform = target_transform
        self.return_indexes = return_indexes
        self.whole_data = dataset(root, train, transform=transform, target_transform=target_transform,
                                  download=download)
        self.data = self.whole_data
        self.labels = np.array(self.data.targets)
        if classes is not None:
            classes = np.array(classes)
            selection = np.isin(self.labels, classes)
            self.data = Subset(self.data, np.arange(len(self.labels))[selection])
            self.labels = self.labels[selection]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inputs, targets = self.data[index]
        if self.return_indexes:
            return inputs, (targets, index)
        return inputs, targets

    def input_shape(self):
        return self.data[0][0].shape

    def get_dataloader(self, batch_size,
                       shuffle=config.dataloader_shuffle,
                       num_workers=config.dataloader_num_workers,
                       pin_memory=config.dataloader_pin_memory,
                       drop_last=config.dataloader_drop_last,
                       sampler=None
                       ):
        return DataLoader(self, batch_size, shuffle, num_workers=num_workers, pin_memory=pin_memory,
                          drop_last=drop_last, sampler=sampler)

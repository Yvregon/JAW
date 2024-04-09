import torch

class CustomDatasetTransformer(torch.utils.data.Dataset):
    """Custom dataset declaration. Write here the transformations you want aply to your custom dataset. Inherit form :class: `torch.utils.data.Dataset`"""

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        input, target = self.base_dataset[index]
        return self.transform(input), target

    def __len__(self):
        return len(self.base_dataset)

from .data_utils import compute_mean_std, DatasetTransformer
import torch
import torchvision.transforms as transforms
import torchvision

# Example with valid_ratio = 0.2 : going to use 80%/20% split for train/valid
def load_dataset_FashionMNIST_with_standardization(dataset_path, valid_ratio=0.2, num_threads=4, batch_size=128):

    # Load the dataset for the training/validation sets
    train_valid_dataset = torchvision.datasets.FashionMNIST(
        root=dataset_path,
        train=True,
        transform=None,
        download=True,
    )

    # Load the test set
    test_dataset = torchvision.datasets.FashionMNIST(
        root=dataset_path,
        transform=None,
        train=False
    )

    # Split it into training and validation sets
    nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset))
    nb_valid = int(valid_ratio * len(train_valid_dataset))
    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(
        train_valid_dataset, [nb_train, nb_valid]
    )

    normalizing_dataset = train_dataset
    train_dataset = DatasetTransformer(train_dataset, transforms.ToTensor())
    valid_dataset = DatasetTransformer(valid_dataset, transforms.ToTensor())
    test_dataset = DatasetTransformer(test_dataset, transforms.ToTensor())

    ## NORMALISATION
    # Loading the dataset is using 4 CPU threads
    # Using minibatches of 128 samples, except for the last that can be smaller.

    normalizing_dataset = train_dataset
    normalizing_loader = torch.utils.data.DataLoader(
        dataset=normalizing_dataset, batch_size=batch_size, num_workers=num_threads
    )

    # Compute mean and variance from the training set
    mean_train_tensor, std_train_tensor = compute_mean_std(normalizing_loader)

    data_transforms = transforms.Compose(
        [transforms.Lambda(lambda x: (x - mean_train_tensor) / std_train_tensor)]
    )

    train_dataset = DatasetTransformer(train_dataset, data_transforms)
    valid_dataset = DatasetTransformer(valid_dataset, data_transforms)
    test_dataset = DatasetTransformer(test_dataset, data_transforms)

    ## DATALOADERS

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,  # <-- this reshuffles the data at every epoch
        num_workers=num_threads,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_threads,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_threads,
    )

    return train_loader, valid_loader, test_loader

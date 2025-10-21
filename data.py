import math
import torch
import torch.utils.data
from pathlib import Path
from torchvision import datasets, transforms
import multiprocessing

from .helpers import compute_mean_and_std, get_data_location
import matplotlib.pyplot as plt


def get_data_loaders(
    batch_size: int = 32, valid_size: float = 0.2, num_workers: int = 1, limit: int = -1
):
    """
    Create and returns the train, validation and test data loaders.

    :param batch_size: size of the mini-batches
    :param valid_size: fraction of the dataset to use for validation. For example 0.2
                       means that 20% of the dataset will be used for validation
    :param num_workers: number of workers to use in the data loaders. Use num_workers=1. 
    :param limit: maximum number of data points to consider
    :return a dictionary with 3 keys: 'train', 'valid' and 'test' containing respectively the
            train, validation and test data loaders
    """

    data_loaders = {"train": None, "valid": None, "test": None}

    base_path = Path(get_data_location())

    # Compute mean and std of the dataset
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")

    # Improved data augmentation for better generalization
    data_transforms = {   
        "train": transforms.Compose([      
            transforms.Resize(256),      
            transforms.RandomCrop(224),  # FIXED: Changed from RandomResizedCrop to RandomCrop for better consistency
            transforms.RandomHorizontalFlip(p=0.5),       
            transforms.RandomRotation(degrees=10),  # FIXED: Reduced rotation to prevent excessive distortion
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # FIXED: Reduced jitter
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
        
        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
        
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    }

    # Create train and validation datasets
    # Use separate folders for train and validation if they exist
    train_path = base_path / "train"
    test_path = base_path / "test"
    
    # Check if validation folder exists, otherwise split from train
    if (base_path / "val").exists() or (base_path / "valid").exists():
        # Use separate validation folder if it exists
        val_path = base_path / "val" if (base_path / "val").exists() else base_path / "valid"
        
        train_data = datasets.ImageFolder(
            train_path,
            transform=data_transforms["train"]
        )
        
        valid_data = datasets.ImageFolder(
            val_path,
            transform=data_transforms["valid"]
        )
        
        # No need to split indices since we have separate validation set
        train_idx = None
        valid_idx = None
        
    else:
        # Split from train dataset (original approach)
        train_data = datasets.ImageFolder(
            train_path,
            transform=data_transforms["train"]
        )
        
        # The validation dataset will be a split from the train dataset
        valid_data = datasets.ImageFolder(
            train_path,
            transform=data_transforms["valid"]
        )
        
        # Obtain training indices that will be used for validation
        n_tot = len(train_data)
        indices = torch.randperm(n_tot)

        # If requested, limit the number of data points to consider
        if limit > 0:
            indices = indices[:limit]
            n_tot = limit

        split = int(math.ceil(valid_size * n_tot))
        train_idx, valid_idx = indices[split:], indices[:split]
    
    test_data = datasets.ImageFolder(
        test_path,
        transform=data_transforms["test"]
    )
   
    # Prepare data loaders based on whether we have separate validation folder or not
    if train_idx is None:
        # Separate validation folder exists
        data_loaders["train"] = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,  # FIXED: Added shuffle for training
            num_workers=num_workers,
        )
        
        data_loaders["valid"] = torch.utils.data.DataLoader(
            valid_data,
            batch_size=batch_size,
            shuffle=False,  # No need to shuffle validation
            num_workers=num_workers,
        )
    else:
        # Split from train dataset - use samplers
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)
        
        data_loaders["train"] = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
        )
        
        data_loaders["valid"] = torch.utils.data.DataLoader(
            valid_data,
            batch_size=batch_size,
            sampler=valid_sampler,
            num_workers=num_workers,
        )

    # Create test data loader
    # Improved test data loader handling
    if limit > 0:
        # Use Subset instead of SubsetRandomSampler for test to maintain order
        test_indices = torch.arange(min(limit, len(test_data)))
        test_subset = torch.utils.data.Subset(test_data, test_indices)
        test_data_limited = test_subset
    else:
        test_data_limited = test_data

    data_loaders["test"] = torch.utils.data.DataLoader(
        test_data_limited,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # Important: don't shuffle test data
    )

    # Print dataset sizes for debugging
    print(f"Training set size: {len(data_loaders['train'].dataset)}")
    print(f"Validation set size: {len(data_loaders['valid'].dataset)}")
    print(f"Test set size: {len(data_loaders['test'].dataset)}")
    print(f"Number of classes: {len(train_data.classes)}")
    print(f"Class names: {train_data.classes}")

    return data_loaders


def visualize_one_batch(data_loaders, max_n: int = 5):
    """
    Visualize one batch of data.

    :param data_loaders: dictionary containing data loaders
    :param max_n: maximum number of images to show
    :return: None
    """

    # obtain one batch of training images
    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    # Undo the normalization (for visualization purposes)
    mean, std = compute_mean_and_std()
    invTrans = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )

    images = invTrans(images)

    # Get class names from the train data loader
    class_names = data_loaders["train"].dataset.classes
    
    # Convert from BGR to RGB and ensure proper dimensions
    #Improved image handling for visualization
    if images.dim() == 4 and images.shape[1] == 3:  # [batch, channels, height, width]
        images = images.permute(0, 2, 3, 1)  # Change to [batch, height, width, channels]
    
    images = images.clip(0, 1)

    # Plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in range(min(max_n, len(images))):  # FIXED: Added bounds check
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        ax.set_title(class_names[labels[idx].item()])


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    return get_data_loaders(batch_size=2, num_workers=0)


def test_data_loaders_keys(data_loaders):
    assert set(data_loaders.keys()) == {"train", "valid", "test"}, "The keys of the data_loaders dictionary should be train, valid and test"


def test_data_loaders_output_type(data_loaders):
    # Test the data loaders
    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    assert isinstance(images, torch.Tensor), "images should be a Tensor"
    assert isinstance(labels, torch.Tensor), "labels should be a Tensor"
    assert images[0].shape[-1] == 224, "The tensors returned by your dataloaders should be 224x224. Did you forget to resize and/or crop?"


def test_data_loaders_output_shape(data_loaders):
    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    assert len(images) == 2, f"Expected a batch of size 2, got size {len(images)}"
    assert len(labels) == 2, f"Expected a labels tensor of size 2, got size {len(labels)}"


def test_visualize_one_batch(data_loaders):
    visualize_one_batch(data_loaders, max_n=2)
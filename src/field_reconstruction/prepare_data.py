import torch
import torch.optim as optim
import torch.nn as nn


def create_dataloaders(ds, batch_size=32):
    """
    Create DataLoaders for training and validation datasets.
    
    Args:
        ds: Dataset to split.
        batch_size: Batch size for DataLoader.
    
    Returns:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
    """
    
    # Split dataset into train and validation sets
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(ds, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
    
def prepare_data(dataset, config):
    """
    Prepare the dataset for training and testing.
    
    Args:
        dataset: Dataset to prepare.
    
    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    
    print("🔍 Loading dataset...")
    if dataset == "weatherbench2_5vars_flat":
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/weatherbench2_5vars_flat.nc"))
        ds = xr.open_dataset(data_path)
    
    if dataset == "weatherbench2_5vars_3d":
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/weatherbench2_5vars_3d.nc"))
        ds = xr.open_dataset(data_path)
    
    print("Loaded dataset:", data_path)
    
    #Perform train, val split and put in dataloaders
    train_loader, val_loader = create_dataloaders(ds, config.batch_size)
    print("🔍 Dataset prepared.")
    return train_loader, val_loader
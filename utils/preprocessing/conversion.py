import os
import torch

def cached_loader2tensor(data, cache_path=None):
    """
    Converts a dataset to a tensor by loading it in batches and caches the result.
    
    If a cached tensor file exists at 'cache_path', the function loads the tensor from the file,
    skipping the conversion. Otherwise, it performs the conversion, saves the tensor, and returns it.
    
    Args:
        data (Dataset): The dataset to convert.
        flatten (bool): If True, the resulting tensor is flattened.
        cache_path (str, optional): Path to the cache file. If not provided, a default filename is used.
        
    Returns:
        torch.Tensor: The resulting tensor on the appropriate device.
    """
    # Define a default cache file name if none is provided.
    if cache_path is None:
        cache_path = "cached_tensor.pt"
    
    # If the cached file exists, load and return the tensor.
    if os.path.exists(cache_path):
        print(f"Loading cached tensor from {cache_path}")
        return torch.load(cache_path,weights_only=True)
    
    # Otherwise, convert the dataset to a tensor.
    loader = torch.utils.data.DataLoader(data, batch_size=1024, shuffle=False, pin_memory=True, num_workers=4)
    images = []
    
    for batch, _ in loader:
        images.append(batch)
    
    images = torch.cat(images, dim=0)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = images.to(device, non_blocking=True)
    
    # Save the converted tensor to the cache file.
    torch.save(images, cache_path)
    print(f"Saved converted tensor to {cache_path}")
    
    return images

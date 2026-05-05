import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = [f for f in os.listdir(root) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),   
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def create_dataset(monet_path, photo_path, batch_size):
    
    """
    This function takes the path to the monet and photo images and returns a dataloader for each.
    Args:
        monet_path (str): Path to the monet images.
        photo_path (str): Path to the photo images.
        batch_size (int): Batch size for the dataloader.
    Returns:
        tuple: A tuple containing the monet dataloader and the photo dataloader.
    """
    
    monet_dataset = ImageDataset(root=monet_path, transform=data_transform)
    photo_dataset = ImageDataset(root=photo_path, transform=data_transform)

    monet_loader = DataLoader(monet_dataset, batch_size=batch_size, shuffle=True)
    photo_loader = DataLoader(photo_dataset, batch_size=batch_size, shuffle=True)

    return monet_loader, photo_loader
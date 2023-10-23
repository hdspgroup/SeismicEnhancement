import torchvision.transforms as TF
import torchvision.datasets as datasets


from torch.utils.data import Dataset, DataLoader
from .utils import DeviceDataLoader
from .config import TrainingConfig
import numpy as np
import os


def read_numpy_file(filename):
    img = np.load(filename, allow_pickle=True)
    img = np.expand_dims(img, axis=-1)
    return img


class BuildDataset(Dataset):

    def __init__(
        self,
        data_path,
        transform=None,
        pre_load=False
    ):

        self.data_path = data_path
        self.imgs_names = os.listdir(data_path)
        self.imgs_names = [
            x for x in self.imgs_names if x.endswith(('.npy'))]

        self.transform = transform

        self.imgs = []
        self.pre_load = pre_load

        if pre_load:
            self.imgs = self.pre_loader()

    def pre_loader(self):

        imgs = []

        for img_name in self.imgs_names:
            img_path = os.path.join(self.data_path, img_name)
            imgs.append(read_numpy_file(img_path))

        return imgs

    def __len__(self):
        return len(self.imgs_names)

    def __getitem__(self, idx):

        if self.pre_load:
            sample = self.imgs[idx]
        else:
            img_path = os.path.join(self.data_path, self.imgs_names[idx])
            sample = read_numpy_file(img_path)

        if self.transform:
            sample = self.transform(sample)

        return sample


class CondBuildDataset(Dataset):

    def __init__(
        self,
        data_path,
        transform=None,
        pre_load=True
    ):

        self.data_path = data_path
        self.imgs_names = os.listdir(data_path)
        self.imgs_names = [
            x for x in self.imgs_names if x.endswith(('.npy'))]

        self.transform = transform

        self.imgs = []
        self.pre_load = pre_load

        self.data_paths = {
            'patch':  "./data/patches_f3_v3",
            'offset': "./data/patches_f3_v4",
        }

        if pre_load:
            self.imgs = self.pre_loader()

        

    def pre_loader(self):

        imgs = []

        subfolders = os.listdir(self.data_path)
        subfolders = [x for x in subfolders if x.isdigit()]

        for subfolder in subfolders:

            subfolder_path = os.path.join(self.data_path, subfolder)
            imgs_list = os.listdir(subfolder_path)
            imgs_list = [
                x for x in imgs_list if x.endswith(('.png'))]

            class_id = int(subfolder)

            for img_name in imgs_list:

                type_image = img_name.split("_")[0]
                image_id = img_name.split("_")[-1].split(".")[0]
                img_path = os.path.join(self.data_paths[type_image], "patch_"+image_id+".npy")
                imgs.append((read_numpy_file(img_path), class_id))


        return imgs

    def __len__(self):
        if self.pre_load:
            return len(self.imgs)
        return len(self.imgs_names)

    def __getitem__(self, idx):

        if self.pre_load:
            sample = self.imgs[idx]
        else:
            img_path = os.path.join(self.data_path, self.imgs_names[idx])
            sample = read_numpy_file(img_path)

        if self.transform:
            img, class_id = sample
            img = self.transform(img)
            sample = (img, class_id)

        return sample


def get_dataset(dataset_name='MNIST', training_config=TrainingConfig):

    size = training_config.IMG_SHAPE[-1]
    conditional = training_config.CONDITIONAL

    transforms = TF.Compose(
        [
            TF.ToTensor(),
            TF.RandomCrop((size, size)),
            # TF.Resize((size, size)),
            # TF.Resize((64, 64),
            #           interpolation=TF.InterpolationMode.BICUBIC,
            #           antialias=True),
            #             TF.RandomHorizontalFlip(),

            TF.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
        ]
    )

    if dataset_name.upper() == "MNIST":
        dataset = datasets.MNIST(
            root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Cifar-10":
        dataset = datasets.CIFAR10(
            root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Cifar-100":
        dataset = datasets.CIFAR10(
            root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Flowers":
        dataset = datasets.ImageFolder(
            root="/kaggle/input/flowers-recognition/flowers", transform=transforms)
    elif dataset_name == "Seismic":
        
        if conditional:
            dataset = CondBuildDataset("./data/f3_patches100f3v2", transform=transforms)
        else:
            dataset = BuildDataset("/data/patches_f3_v4", transform=transforms)

    return dataset


def get_dataloader(dataset_name='MNIST',
                   batch_size=32,
                   pin_memory=False,
                   shuffle=True,
                   num_workers=0,
                   device="cpu"
                   ):
    dataset = get_dataset(dataset_name=dataset_name)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            pin_memory=pin_memory,
                            num_workers=num_workers,
                            shuffle=shuffle
                            )
    device_dataloader = DeviceDataLoader(dataloader, device)
    return device_dataloader


def inverse_transform(tensors):
    """Convert tensors from [-1., 1.] to [0., 255.]"""
    return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * 255.0

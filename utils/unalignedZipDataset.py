from torch.utils.data import Dataset
from monai.transforms import Compose
import random

class UnalignedZipDataset(Dataset):
    def __init__(self, A_paths, B_paths, transform: Compose, serial_batches = False) -> None:
        super().__init__()
        self.A_paths = A_paths
        self.B_paths = B_paths
        self.transform = transform
        self.A_size = 0 if A_paths is None else len(A_paths)
        self.B_size = len(B_paths)

    def __len__(self) -> int:
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return min(self.A_size, self.B_size)

    def __getitem__(self, index) -> dict:
        B_path = self.B_paths[index % self.B_size]
        if self.A_paths is not None:
            A_path = self.A_paths[random.randint(0, self.A_size - 1)]
            data = {"real_A": A_path, "path_A": A_path, "real_B": B_path, "path_B": B_path}
        else:
            data = {"real_B": B_path, "path_B": B_path}
        data_transformed = self.transform(data)
        return data_transformed


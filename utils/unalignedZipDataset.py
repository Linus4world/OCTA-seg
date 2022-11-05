from torch.utils.data import Dataset
from monai.transforms import Compose
import random

class UnalignedZipDataset(Dataset):
    def __init__(self, A_paths: list[str], B_paths: list[str], A_seg_paths: list[str], transform: Compose, phase = "train") -> None:
        super().__init__()
        self.A_paths = A_paths
        self.B_paths = B_paths
        self.A_seg_paths = A_seg_paths
        self.transform = transform
        self.A_size = 0 if A_paths is None else len(A_paths)
        self.B_size = 0 if B_paths is None else len(B_paths)
        self.A_seg_size = 0 if A_seg_paths is None else len(A_seg_paths)
        self.phase = phase

    def __len__(self) -> int:
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def __getitem__(self, index) -> dict:
        data = dict()
        if self.phase=="test":
            if self.A_paths is not None:
                A_path = self.A_paths[index % self.A_size]
                return self.transform({"image": A_path, "path": A_path})
            else:
                B_path = self.B_paths[index % self.B_size]
                return self.transform({"image": B_path, "path": B_path})

        if self.A_paths is not None:
            A_path = self.A_paths[index % self.A_size]
            data["path_A"] = A_path
            data["real_A"] = A_path
        if self.B_paths is not None:
            B_path = self.B_paths[random.randint(0, self.B_size - 1)]
            data["path_B"] = B_path
            data["real_B"] = B_path
        if self.A_seg_paths is not None:
            A_seg_path = self.A_seg_paths[index % self.A_size]
            data["path_A_seg"] = A_seg_path
            data["real_A_seg"] = A_seg_path
        data_transformed = self.transform(data)
        return data_transformed


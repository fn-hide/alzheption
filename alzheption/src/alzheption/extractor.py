import os
import torch
import pickle
import numpy as np
import pandas as pd
import torchvision as tv
from PIL import Image
from tqdm import tqdm


class AlzheptionExtractor():
    def __init__(
            self,
            path_dataset: str, 
            test_size: float, 
            train_transform: tv.transforms.Compose, 
            test_transform: tv.transforms.Compose,
            batch_size=256,
        ):
        self.path_dataset = path_dataset
        self.test_size = test_size
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.batch_size = batch_size

        self._dataset: tv.datasets.ImageFolder | None = None
        self._train_indices: torch.utils.data.Subset | None = None
        self._test_indices: torch.utils.data.Subset | None = None
        self._dataset_train: torch.utils.data.Subset | None = None
        self._dataset_test: torch.utils.data.Subset | None = None
        self._train_loader: torch.utils.data.DataLoader | None = None
        self._test_loader: torch.utils.data.DataLoader | None = None
        self._device: torch.device | None = None
        self._model: tv.models.Inception3 | None = None
        self._train_features: np.ndarray | None = None
        self._test_features: np.ndarray | None = None
        self._train_labels: np.ndarray | None = None
        self._test_labels: np.ndarray | None = None

    def __call__(self) -> tuple[np.ndarray, np.ndarray]:
        return self.train_features, self.test_features

    def save_extractor(self, dir_path=".", suffix: str | None = None) -> None:
        self._model.to(torch.device("cpu"))

        if os.path.exists(dir_path) is False:
            print(f"Destination path doesn't exists. Creating new dirs: {dir_path}")
            os.makedirs(dir_path)

        with open(f"{dir_path}/AlzheptionExtractor{suffix}.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_extractor(cls, filepath="./AlzheptionExtractor.pkl", path_dataset_new: str | None = None) -> "AlzheptionExtractor":
        with open(filepath, "rb") as f:
            obj: AlzheptionExtractor = pickle.load(f)
        
        if not os.path.exists(obj.path_dataset):
            obj._dataset = None
        
        if isinstance(path_dataset_new, str):
            obj.path_dataset = path_dataset_new
        
        return obj

    def show_dataset_count(self) -> None:
        """Show dataset count"""
        print(f"Train dataset size : {len(self.dataset_train)}")
        print(f"Test dataset size  : {len(self.dataset_test)}")

    def get_sample_of_original_dataset(self, index_of_indices=0, part="test") -> Image:
        """Get sample of original dataset"""
        if part not in ["train", "test"]:
            raise ValueError(f"Part must be 'train' or 'test', not {part}!")

        subset = eval(f"self.{part}_indices.indices")
        index = subset[index_of_indices]
        print(f"Get sample of original dataset ({part}) with index: {index}")
        print()

        x, y = self.dataset[index]
        return Image.fromarray(np.array(x))

    def get_sample_of_preprocessed_dataset(self, index=0, part="test") -> Image:
        """Get sample of preprocessed dataset"""
        if part not in ["train", "test"]:
            raise ValueError(f"Part must be 'train' or 'test', not {part}!")

        print(f"Get sample of preprocessed dataset ({part}) with index: {index}")
        print()

        x, y = eval(f"self.dataset_{part}[{index}]")
        return Image.fromarray((x.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    @property
    def dataset(self) -> tv.datasets.ImageFolder:
        if self._dataset is None:
            self._dataset = self._load_dataset()
        return self._dataset

    def _load_dataset(self, transform: tv.transforms.Compose | None = None) -> tv.datasets.ImageFolder:
        """Load dataset without transformations"""
        return tv.datasets.ImageFolder(self.path_dataset, transform=transform)

    @property
    def train_indices(self) -> torch.utils.data.Subset:
        """Get train indices"""
        if self._train_indices is None:
            self._train_indices, self._test_indices = self._get_indices()
        return self._train_indices

    @property
    def test_indices(self) -> torch.utils.data.Subset:
        """Get test indices"""
        if self._test_indices is None:
            self._train_indices, self._test_indices = self._get_indices()
        return self._test_indices

    def _get_indices(self) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
        """Get train and test indices"""
        train_size = int((1 - self.test_size) * len(self.dataset))
        test_size = len(self.dataset) - train_size
        return torch.utils.data.random_split(range(len(self.dataset)), [train_size, test_size])

    @property
    def dataset_train(self) -> torch.utils.data.Subset:
        """Get train dataset"""
        if self._dataset_train is None:
            self._dataset_train = self._get_dataset(self.train_transform, self.train_indices)
        return self._dataset_train

    @property
    def dataset_test(self) -> torch.utils.data.Subset:
        """Get test dataset"""
        if self._dataset_test is None:
            self._dataset_test = self._get_dataset(self.test_transform, self.test_indices)
        return self._dataset_test

    def _get_dataset(self, transform: tv.transforms.Compose, indices: torch.utils.data.Subset) -> torch.utils.data.Subset:
        """Get dataset with given indices and transform"""
        return torch.utils.data.Subset(self._load_dataset(transform), indices.indices)

    @property
    def train_loader(self) -> torch.utils.data.DataLoader:
        """Get train loader"""
        if self._train_loader is None:
            self._train_loader = self._get_loader(self.dataset_train)
        return self._train_loader

    @property
    def test_loader(self) -> torch.utils.data.DataLoader:
        """Get test loader"""
        if self._test_loader is None:
            self._test_loader = self._get_loader(self.dataset_test, shuffle=False)
        return self._test_loader

    def _get_loader(self, dataset: torch.utils.data.Subset, shuffle=True) -> torch.utils.data.DataLoader:
        """Get loader for given dataset"""
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    @property
    def device(self) -> torch.device:
        """Get device"""
        if self._device is None:
            self._device = self._get_device()
        return self._device

    def _get_device(self) -> torch.device:
        """Get device"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def model(self) -> tv.models.Inception3:
        """Load pre-trained InceptionV3 model and set as feature extractor"""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> tv.models.Inception3:
        """Load pre-trained InceptionV3 model and set as feature extractor"""
        model = tv.models.inception_v3(weights=True)
        model.aux_logits = False
        
        model.fc = torch.nn.Identity()
        
        # # Ambil fitur sebelum FC (dari GAP layer)
        # model.fc = torch.nn.Sequential(
        #     torch.nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
        #     torch.nn.Flatten()  # Flatten agar bisa digunakan oleh KELM
        # )
        
        # model = torch.nn.Sequential(
        #     model.Conv2d_1a_3x3,
        #     model.Conv2d_2a_3x3,
        #     model.Conv2d_2b_3x3,
        #     model.maxpool1,
        #     model.Conv2d_3b_1x1,
        #     model.Conv2d_4a_3x3,
        #     model.maxpool2,
        #     model.Mixed_5b,
        #     model.Mixed_5c,
        #     model.Mixed_5d,
        #     model.Mixed_6a,
        #     model.Mixed_6b,
        #     model.Mixed_6c,
        #     model.Mixed_6d,
        #     model.Mixed_6e,
        #     model.Mixed_7a,
        #     model.Mixed_7b,
        #     model.Mixed_7c  # Ambil fitur dari sini
        # )

        # Move to GPU if available and use DataParallel
        if torch.cuda.device_count() > 1:
            print(f'Using {torch.cuda.device_count()} GPUs!')
            model = torch.nn.DataParallel(model)
        model = model.to(self.device)
        return model

    @property
    def train_features(self) -> np.ndarray:
        """Get train features"""
        if self._train_features is None:
            self._train_features, self._train_labels = self._extract_features("train")
        return self._train_features

    @property
    def train_labels(self) -> np.ndarray:
        """Get train labels"""
        if self._train_labels is None:
            self._train_features, self._train_labels = self._extract_features("train")
        return self._train_labels

    @property
    def test_features(self) -> np.ndarray:
        """Get test features"""
        if self._test_features is None:
            self._test_features, self._test_labels = self._extract_features("test")
        return self._test_features

    @property
    def test_labels(self) -> np.ndarray:
        """Get test labels"""
        if self._test_labels is None:
            self._test_features, self._test_labels = self._extract_features("test")
        return self._test_labels

    def _extract_features(self, part="test") -> tuple[np.ndarray, list[int]]:
        """Function to extract features from InceptionV3"""
        if part not in ["train", "test"]:
            raise ValueError(f"Part must be 'train' or 'test', not {part}!")

        data_loader = eval(f"self.{part}_loader")

        self.model.eval()
        features, labels = [], []
        with torch.no_grad():
            for images, label in tqdm(data_loader, desc=f"Extracting '{part}'"):
                images = images.to(self.device)
                output = self.model(images)
                features.append(output.cpu().numpy())
                labels.extend(label.tolist())
        features = np.vstack(features)
        return features, labels

    def save_features(self, dst_path=".") -> None:
        if os.path.exists(dst_path) is False:
            print(f"Destination path doesn't exists. Creating new dirs: {dst_path}")
            os.makedirs(dst_path)

        """Save features"""
        preprocess_train_name = "_".join([str(name).split('(')[0] for name in self.train_transform.transforms])
        preprocess_test_name = "_".join([str(name).split('(')[0] for name in self.test_transform.transforms])

        np.savez(f"{dst_path}/TrainFeatures_{preprocess_train_name}.npz", TrainFeatures=self.train_features)
        np.savez(f"{dst_path}/TrainLabels_{preprocess_train_name}.npz", TrainLabels=self.train_labels)
        np.savez(f"{dst_path}/TestFeatures_{preprocess_test_name}.npz", TestFeatures=self.test_features)
        np.savez(f"{dst_path}/TestLabels_{preprocess_test_name}.npz", TestLabels=self.test_labels)
    
    def view_train_label_distribution(self):
        return self._view_label_distribution()
    
    def view_test_label_distribution(self):
        return self._view_label_distribution(is_train=False)
    
    def _view_label_distribution(self, is_train=True) -> pd.DataFrame:
        data = self.train_labels if is_train else self.test_labels
        df = pd.DataFrame(np.unique(data, return_counts=True)).transpose()
        df.columns = ["label", "count"]
        
        return df
    
    def balance_feature(self) -> None:
        self._train_features, self._train_labels = self._balance_feature()
        self._test_features, self._test_labels = self._balance_feature(subset="test")
    
    def _balance_feature(self, subset="train") -> tuple[np.ndarray, list[int]]:
        if subset not in ("train", "test"):
            raise ValueError("Subset must be in ('train', 'test')")
        
        df = eval(f"self.view_{subset}_label_distribution()")
        feature = eval(f"self.{subset}_features")
        label = eval(f"self.{subset}_labels")
        
        minval = df.sort_values(by=["count"]).head(1)["count"].item()
        df = pd.DataFrame({"label": label}).groupby("label").head(minval)
        index = df.index.tolist()

        return feature[index], np.array(label)[index]

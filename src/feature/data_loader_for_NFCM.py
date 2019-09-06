import torchvision
import torch
from torch.utils.data import Dataset
import numpy as np
import random


# ラベル	クラス
# 0	T-シャツ/トップ (T-shirt/top)
# 1	ズボン (Trouser)
# 2	プルオーバー (Pullover)
# 3	ドレス (Dress)
# 4	コート (Coat)
# 5	サンダル (Sandal)
# 6	シャツ (Shirt)
# 7	スニーカー (Sneaker)
# 8	バッグ (Bag)
# 9	アンクルブーツ (Ankle boot)

near_cat_dict = {0:[0,1,5,6,7,9], 1:[0,1,2,5,6,7,9], 2:[1,2,5,7], 3:[3,4,8],
                 4:[1,4,6,8,9], 5:[0,1,5,6], 6:[1,5,6,7,9], 7:[0,1,6,7],
                 8:[3,4,8,9], 9:[1,4,6,8,9]}
# far_cat = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}


class FMNISTDataset(Dataset):
    def __init__(self, n_class=10, train=True):
        super().__init__()
        self.n_class = n_class
        self.train = train
        self.n_relation = self.n_class**2
        self.fashion_mnist_data = torchvision.datasets.FashionMNIST(
            './fashion-mnist',
            transform=torchvision.transforms.ToTensor(),
            train=train,
            download=True)
        self.labels = [fnmn[1] for fnmn in torchvision.datasets.FashionMNIST('./fashion-mnist')]
        self.labels = np.array(self.labels, dtype=np.int32)

    def __len__(self):
        return len(self.fashion_mnist_data)

    def __getitem__(self, idx):
        image, cat = self.fashion_mnist_data[idx]
        if not self.train:
            return image, cat

        near_cat = self._get_near_category(cat)
        far_cat = self._get_far_category()
        near_image = self._get_near_image(near_cat)
        far_image = self._get_far_image(far_cat)

        l_for_tag = list(range(self.n_class))
        # l_for_tag.pop(cat)
        near_relational_tag = cat*(self.n_class-1) + l_for_tag.index(near_cat)
        far_relational_tag = cat*(self.n_class-1) + l_for_tag.index(far_cat)
        return (image,
                cat,
                near_image,
                near_cat,
                far_image,
                far_cat,
                near_relational_tag,
                far_relational_tag
            )

    def _get_near_category(self, category):
        self.l_near_cat = near_cat_dict[category]
        idx = random.randint(0, len(self.l_near_cat)-1)
        return self.l_near_cat[idx]

    def _get_far_category(self):
        l_far_car = list(set(range(self.n_class)) - set(self.l_near_cat))
        idx = random.randint(0, len(l_far_car)-1)
        return l_far_car[idx]

    def _get_near_image(self, near_category):
        idxs = np.where(self.labels==near_category)[0]
        random.shuffle(idxs)
        idx = idxs[0]
        return self.fashion_mnist_data[idx][0]

    def _get_far_image(self, far_category):
        idxs = np.where(self.labels==far_category)[0]
        random.shuffle(idxs)
        idx = idxs[0]
        return self.fashion_mnist_data[idx][0]

    def _for_onehot(self, idx, vec_len):
        onehot = np.zeros(vec_len, dtype=np.int32)
        onehot[idx] = 1
        return onehot


def loader(dataset, batch_size, shuffle=True):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0)
    return loader

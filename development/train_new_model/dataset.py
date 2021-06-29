import cv2, os
import numpy as np
import torch
from torch.utils.data import Dataset
from instafilter.utils import features_from_image


class ColorizedDataset(Dataset):
    def __init__(self, f_source, f_target, device):

        if os.path.isfile(f_source) and os.path.isfile(f_target):
            # load single images
            img0 = cv2.imread(str(f_source))
            f0 = features_from_image(img0)

            img1 = cv2.imread(str(f_target))
            f1 = features_from_image(img1)
        else:
            l_im_fps = [f for f in os.listdir(f_source) if f.endswith(('.jpg','.JPG'))]
            assert len(l_im_fps)>0, f"not jpg nor JPG files found in {f_source}"

            f0_all = [features_from_image(
                        cv2.imread(os.path.join(f_source, f))
                        )
                for f in l_im_fps]
            f1_all = [features_from_image(
                        cv2.imread(os.path.join(f_target, f))
                        )
                for f in l_im_fps]
            f0 = np.concatenate(f0_all, axis = 0)
            f1 = np.concatenate(f1_all, axis = 0)

        assert f0.shape == f1.shape
        print('--- ColorizedDataset:')
        print(f"\tinput dataset shape: {f0.shape}")
        print(f"\ttarget dataset shape: {f1.shape}")

        self.x = torch.tensor(f0).to(device)
        self.y = torch.tensor(f1).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

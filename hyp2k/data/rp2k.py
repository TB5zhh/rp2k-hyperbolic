import os
from logging import log, INFO
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json

def loadimg(path: str, permute=bool, aug=None):
    trans = transforms.Compose(aug)
    img = Image.open(path).convert('RGB')
    img = trans(img)
    img = img.permute(0, 2, 1)
    return img


class RP2kDataset(Dataset):
    def __init__(
            self,
            path,
            mode: str,
            args,
            num=-1,
            aug=[transforms.RandomResizedCrop(224),
                 transforms.ToTensor()]):
        '''
        mode: Dataset type, 'train' / 'eval'
        num: Number of samples for each class, -1 means all samples
        '''
        self.len = 0
        self.data = []
        self.config = args
        self.aug = aug
        if mode != 'train' and mode != 'val':
            raise RuntimeError("Please specify train/val set! (train/val)")

        l = os.listdir(os.path.join(path, mode))
        for idx, dir in enumerate(l):
            if self.config.load_all and (idx + 1) % 100 == 0:
                log(INFO, f"---> Loading #{idx + 1} of {len(l)} categories")
            if dir == '1331':
                print("Skip 'others' category.")
                continue
            subdir = os.path.join(path, mode, dir)
            if os.path.isdir(subdir):
                if num == -1:
                    for img_name in os.listdir(subdir):
                        if self.config.load_all:
                            img0 = loadimg(os.path.join(subdir, img_name),
                                           aug=self.aug)
                            img1 = loadimg(os.path.join(subdir, img_name),
                                           aug=self.aug)
                            self.data.append(((img0, img1), dir))
                        else:
                            self.data.append((os.path.join(subdir,
                                                           img_name), dir))
                        self.len += 1
                else:
                    for img_name in os.listdir(subdir)[:num]:
                        if self.config.load_all:
                            img0 = loadimg(os.path.join(subdir, img_name),
                                           aug=self.aug)
                            img1 = loadimg(os.path.join(subdir, img_name),
                                           aug=self.aug)
                            self.data.append(((img0, img1), dir))
                        else:
                            self.data.append((os.path.join(subdir,
                                                           img_name), dir))
                        self.len += 1

    def __getitem__(self, index):
        if self.config.load_all:
            imgs = self.data[index][0]
            cate = self.data[index][1]
            return (imgs, int(cate))
        else:
            img0 = loadimg(self.data[index][0], aug=self.aug)
            img1 = loadimg(self.data[index][0], aug=self.aug)
            cate = self.data[index][1]
            return ((img0, img1), int(cate))

    def __len__(self):
        return self.len


def test():
    pass


if __name__ == '__main__':
    test()

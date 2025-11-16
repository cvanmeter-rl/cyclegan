import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import sys
import numpy as np
import torch

class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        if opt.mask_dir_A is not None:
            self.dir_A_mask = getattr(opt,'mask_dir_A',None)

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)
    def mask_path_from_A(self, A_path):
        """
        Build the mask path from the A image filename.
        If mask_dir_A is provided, we keep the filename (stem + suffix).
        """
        if self.dir_A_mask is None:
            return None
        fname = os.path.basename(A_path)
        return os.path.join(self.dir_A_mask, fname)

    def load_mask_long(self, mask_path):
        """
        Load per-pixel class indices as torch.long [H, W].
        DO NOT normalize; do NOT use bilinear.
        """
        if mask_path is None or (not os.path.exists(mask_path)):
            return None
        # load as 8-bit grayscale, values like {0..C-1, 255}
        m = np.array(Image.open(mask_path), dtype=np.uint8)  # [H,W]
        # convert to torch.long
        return torch.from_numpy(m.astype(np.int64))          # [H,W], long

    def __getitem__(self, index):
        k = 10 #retry k times to get a valid image if invalid is found

        for _ in range(k):
            A_path = self.A_paths[index % self.A_size]
            try:
                A_img = Image.open(A_path).convert('RGB')
                break
            except Exception as e:
                print(f"[WARN] Skipping unreadable A: {A_path} ({e})", file=sys.stderr)
                index = (index + 1) % self.A_size
            else:
                raise RuntimeError("Too many unreadable A images in a row.")

        for _ in range(k):
            if self.opt.serial_batches:
                index_B = index % self.B_size
            else:
                index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
            try: 
                B_img = Image.open(B_path).convert('RGB')
                break
            except Exception as e:
                print(f"[WARN] Skipping unreadable B: {B_path} ({e})", file=sys.stderr)
            else:
                raise RuntimeError("Too many unreadable B images in a row.")
        
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        sample = {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

        mask_path = self.mask_path_from_A(A_path)
        A_mask = self.load_mask_long(mask_path)
        A_mask = self.transform(A_mask)
        if A_mask is not None:
            sample['A_mask'] = A_mask
        return sample

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'

"""Pooled paired dataset over all six blur domains for all-in-one training.

Reads a meta-info file, one pair per line, rooted at a single `dataroot`:

    variant_1/train/blur/0001.png variant_1/train/sharp/0001.png 0
    variant_2/train/blur/0001.png variant_2/train/sharp/0001.png 1
    ...

Each item returns lq, gt, domain_id (int, for per-domain eval and embedding
tagging). With `two_view: true` it also returns lq2 -- a second independent
crop+aug of the same blurred image -- forming AirNet's contrastive positive
pair (gt aligns with the first crop only).
"""
import os.path as osp
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import paired_random_crop, random_augmentation
from basicsr.utils import FileClient, img2tensor, imfrombytes, padding


class Dataset_PairedAiO(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean')
        self.std = opt.get('std')
        self.root = opt['dataroot']
        self.two_view = opt.get('two_view', False)
        self.gt_size = opt.get('gt_size')

        with open(opt['meta_info_file']) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        self.pairs = []
        for ln in lines:
            lq_rel, gt_rel, dom = ln.split()
            self.pairs.append((osp.join(self.root, lq_rel),
                               osp.join(self.root, gt_rel), int(dom)))

        if opt['phase'] == 'train':
            self.geometric_augs = opt.get('geometric_augs', True)

    def _read(self, path, key):
        img_bytes = self.file_client.get(path, key)
        return imfrombytes(img_bytes, float32=True)   # HWC, BGR, [0,1]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        lq_path, gt_path, domain = self.pairs[index % len(self.pairs)]
        img_lq = self._read(lq_path, 'lq')
        img_gt = self._read(gt_path, 'gt')

        if self.opt['phase'] == 'train':
            gt_size = self.gt_size
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            lq2 = None
            if self.two_view:
                # second view: independent crop+aug of the same blurred image
                img_lq2, _ = paired_random_crop(img_lq, img_gt, gt_size, scale, gt_path)
                img_lq2 = random_augmentation(img_lq2)[0]
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
            if self.two_view:
                lq2 = img2tensor(img_lq2, bgr2rgb=True, float32=True)

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        out = {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path,
               'domain_id': torch.tensor(domain, dtype=torch.long)}
        if self.opt['phase'] == 'train' and self.two_view:
            out['lq2'] = lq2
        return out

    def __len__(self):
        return len(self.pairs)
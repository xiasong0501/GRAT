import os
import math
import torch
import random
import numpy as np
import pandas as pd
from torch import nn
from PIL import Image
import scipy.stats as st
from torch.utils import data
from torch.nn import functional as F

class NIPS_GAME(data.Dataset):
    def __init__(self, dir, csv_path, transforms=None):
        self.dir = dir   
        self.csv = pd.read_csv(csv_path, engine='python')
        self.transforms = transforms

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['ImageId'] + ".png"
        
        Truelabel = img_obj['TrueLabel'] - 1
        TargetClass = img_obj['TargetClass'] - 1
        img_path = os.path.join(self.dir, ImageID)
        pil_img = Image.open(img_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            data = pil_img
        return data, ImageID, Truelabel, TargetClass

    def __len__(self):
        return len(self.csv)

def seed_torch(seed):
    """Set a random seed to ensure that the results are reproducible"""  
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def save_img(save_path, img, split_channel=False):
    img_ = np.array(img * 255).astype('uint8')
    if split_channel:
        for i in range(img_.shape[2]):
            ch_path = save_path + "@channel{}.jpg".format(i)
            ch = Image.fromarray(img_[:, :, i])
            ch.save(ch_path)
    else:
        Image.fromarray(img_).save(save_path)

class ANDA:
    def __init__(self, device, data_shape=(1, 3, 299, 299)):
        self.data_shape = data_shape
        self.device = device

        self.n_models = 0
        self.noise_mean = torch.zeros(data_shape, dtype=torch.float).to(device)
        self.noise_cov_mat_sqrt = torch.empty((0, np.prod(data_shape)), dtype=torch.float).to(device)

    def sample(self, n_sample=1, scale=0.0, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        mean = self.noise_mean
        cov_mat_sqrt = self.noise_cov_mat_sqrt

        if scale == 0.0:
            assert n_sample == 1
            return mean.unsqueeze(0)

        assert scale == 1.0
        k = cov_mat_sqrt.shape[0]
        cov_sample = cov_mat_sqrt.new_empty((n_sample, k), requires_grad=False).normal_().matmul(cov_mat_sqrt)
        cov_sample /= (k - 1)**0.5

        rand_sample = cov_sample.reshape(n_sample, *self.data_shape)
        sample = mean.unsqueeze(0) + scale * rand_sample
        sample = sample.reshape(n_sample, *self.data_shape)
        return sample

    def collect_model(self, noise):
        mean = self.noise_mean
        cov_mat_sqrt = self.noise_cov_mat_sqrt
        assert noise.device == cov_mat_sqrt.device
        bs = noise.shape[0]
        # first moment
        mean = mean * self.n_models / (self.n_models + bs) + noise.data.sum(dim=0, keepdim=True) / (self.n_models + bs)

        # square root of covariance matrix
        dev = (noise.data - mean).view(bs, -1)
        cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev), dim=0)

        self.noise_mean = mean
        self.noise_cov_mat_sqrt = cov_mat_sqrt
        self.n_models += bs

    def clear(self):
        self.n_models = 0
        self.noise_mean = torch.zeros(self.data_shape, dtype=torch.float).to(self.device)
        self.noise_cov_mat_sqrt = torch.empty((0, np.prod(self.data_shape)), dtype=torch.float).to(self.device)
        
 
def is_sqr(n):
    a = int(math.sqrt(n))
    return a * a == n

def get_theta(i, j):
    theta = torch.tensor([[[1, 0, i], [0, 1, j]]], dtype=torch.float)
    return theta

def get_thetas(n, min_r=-0.5, max_r=0.5):
    range_r = torch.linspace(min_r, max_r, n)
    thetas = []
    for i in range_r:
        for j in range_r:
            thetas.append(get_theta(i, j))
    thetas = torch.cat(thetas, dim=0)
    return thetas

def translation(thetas, imgs):
    grids = F.affine_grid(thetas, imgs.size(), align_corners=False).to(imgs.device)
    output = F.grid_sample(imgs, grids, align_corners=False)
    return output

def scale_transform(input_tensor, m=5):
    outs = [(input_tensor) / (2**i) for i in range(m)]
    x_batch = torch.cat(outs, dim=0)
    return x_batch

# def scale_transform(input_tensor, m=5):
#     shape = input_tensor.shape
#     outs = [(input_tensor) / (2**i) for i in range(m)]
#     x_batch = torch.cat(outs, dim=0)
#     new_shape = x_batch.shape
#     x_batch = x_batch.reshape(m, *shape).transpose(1, 0).reshape(*new_shape)
#     return x_batch

class Translation_Kernel:
    def __init__(self, len_kernel=15, nsig=3, kernel_name='gaussian'):
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel
        
def input_diversity(input_tensor, resize=512, diversity_prob=0.5):
    if torch.rand(1) >= diversity_prob:
        return input_tensor
    image_width = input_tensor.shape[-1]
    assert image_width == 512, "only support ImageNet"
    resize=550
    rnd = torch.randint(image_width, resize, ())
    # rnd=resize
    rescaled = F.interpolate(input_tensor, size=[rnd, rnd], mode='bilinear', align_corners=True)
    h_rem = resize - rnd
    w_rem = resize - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, ())
    pad_right = w_rem - pad_left
    padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.)(rescaled)
    return padded


def get_minibatch(x: torch.Tensor, y: torch.Tensor, minibatch: int):
    nsize = x.shape[0]
    start = 0
    while start < nsize:
        end = min(nsize, start + minibatch)
        yield x[start:end], y[start:end]
        start += minibatch
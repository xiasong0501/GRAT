import torch
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.image as mpimg
import argparse
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
# from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import test_single_volume
from importlib import import_module
from segment_anything import sam_model_registry
import matplotlib.pyplot as plt
from datasets.dataset_synapse import Synapse_dataset
from einops import repeat
from torch.autograd import Variable
from PIL import Image
import torch.nn.functional as F
import torch.fft
import random
import time
import cv2
from one_sample_train import one_sample_train
import torch.distributions as dist
from torchvision.utils import save_image
from concurrent.futures import ThreadPoolExecutor
import torchvision.transforms.functional as TF
parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str, default='testset/test_vol_h5/')
parser.add_argument('--num_classes', type=int, default=8)
parser.add_argument('--output_dir', type=str, default='/output')
parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
parser.add_argument('--dataset', type=str, default='Synapse', help='Experiment name')
parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse/', help='list_dir')
parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
parser.add_argument('--lora_ckpt', type=str, default='checkpoints/epoch_159.pth', help='The checkpoint from LoRA')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
parser.add_argument('--consider_lora', action='store_true', help='Whether to consider lora when attack')
parser.add_argument('--test_batch', type=str, default=None)
parser.add_argument('--out_path', type=str, default='testset/advimages_ours_smoothedmulti_attack/')
parser.add_argument('--noise_sd', type=float, default=0.15)
parser.add_argument('--smoothed_momentum', type=float, default=0.25)
parser.add_argument('--smoothed_freq', type=int, default=3)
parser.add_argument('--sftm_hpyer', type=int, default=4)
parser.add_argument('--loss_thres', type=float, default=0.25)
parser.add_argument('--batch_iter_time', type=int, default=4)
parser.add_argument('--attack_iter', type=int, default=5)
parser.add_argument('--mul_modedl', type=int, default=0)
parser.add_argument('--upd_size', type=float, default=0.2)
parser.add_argument('--step_size', type=float, default=0.2)
parser.add_argument('--noise_path', type=str, default='common_perb/common_perturbation_meta_0.05sp_bound8_it6_0.2upsize_ls0.25.pth')

args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def return_mean_value(input_tensor):
    input_tensor=torch.abs(input_tensor)
    list_mean=[]
    for i in range(len(input_tensor)):
        list_mean.append(input_tensor[i].mean()) 
    input_mean=torch.stack(list_mean)
    # print(input_mean.shape)
    input_mean=input_mean.view(-1, 1,1,1,1)
    return input_mean


def common_direct_loss(selected_features_layer,selected_targets_layer,input_mean):
    # input_mean=input_mean*torch.normal(1,0.25,size=(len(input_mean),1)).view(-1, 1,1,1,1).cuda()
    common_direction= torch.mean((selected_features_layer-selected_targets_layer)/input_mean,dim=0)
    l1_losses = torch.abs(common_direction).mean()
    return l1_losses

def common_direct_loss_weighted(selected_features_layer,selected_targets_layer,input_mean,gradient2,gradient1):
    # input_mean=input_mean*torch.normal(1,0.25,size=(len(input_mean),1)).view(-1, 1,1,1,1).cuda()
    gradient2=gradient2.permute(0,3,1,2)
    gradient1=gradient1.permute(0,3,1,2)
    common_direction= selected_features_layer-selected_targets_layer
    # assert 1==0, print(common_direction[0].shape,gradient2.shape)
    common_direction_extra = (common_direction[0]*gradient2 + common_direction[0]*gradient1)/(input_mean*2) 
    common_direction= torch.mean((common_direction)/input_mean,dim=0)
    l1_losses = torch.abs(common_direction).mean()
    # l1_losses= (torch.abs(torch.mean(common_direction*(selected_features_layer-selected_targets_layer),dim=0))).mean()
    return l1_losses

def calculate_loss(features,targets,features_layer,targets_layer):

    loss_org=criterion(features*0.9999, targets)#/torch.abs(targets).mean()
    # loss_cosine=-F.cosine_similarity(features.view(1, -1), targets.view(1, -1))
    loss_mse=criterion_mse(features*0.9999, targets)
    num_layers=6
    indices = random.sample(range(num_layers), 4)
    selected_features_layer = torch.stack([features_layer[i+10-num_layers] for i in indices])
    selected_targets_layer = torch.stack([targets_layer[i+10-num_layers] for i in indices])
    input_mean=return_mean_value(selected_targets_layer)
    feature_difference= (selected_features_layer*0.999-selected_targets_layer)/input_mean
    Gaussian_est =  torch.randn_like(feature_difference) * 0 + 1
    feature_difference=torch.mul(feature_difference,Gaussian_est)
    l1_losses= (torch.abs(torch.mean(feature_difference,dim=0))).mean()
    loss =0*l1_losses+0*loss_org+1*loss_mse
    targets=targets.view(*targets.shape[:-2], -1)
    features = features.view(*features.shape[:-2], -1)
    features=F.softmax(features,dim=2)
    targets=F.softmax(targets,dim=2)
    info_loss =1*F.kl_div(features.log(), targets, reduction='batchmean')
    return 0.01*info_loss 

def calculate_loss_smoothed(features_noised,features_layer_noised,targets_noised,targets_layer_noised):
    num_layers=6
    indices = random.sample(range(num_layers), 4)
    features_noised=torch.mean(features_noised,dim=0).unsqueeze(0)
    loss_org=criterion(features_noised*0.9999, targets_noised)
    # indices_s = random.sample(range(num_layers+2), 6)
    selected_smoothed_layer = torch.stack([features_layer_noised[i+10-num_layers] for i in indices])
    targets_layer_noised = torch.stack([targets_layer_noised[i+10-num_layers] for i in indices])
    selected_smoothed_layer=torch.mean(selected_smoothed_layer,dim=1)
    targets_layer_noised=torch.mean(targets_layer_noised,dim=1)
    input_mean=return_mean_value(targets_layer_noised)
    feature_difference= (selected_smoothed_layer*0.999-targets_layer_noised)/input_mean
    Gaussian_est =  torch.randn_like(feature_difference) * 1 + 1
    feature_difference=torch.mul(feature_difference,Gaussian_est)
    l1_losses= (torch.abs(torch.mean(feature_difference,dim=0))).mean()
    smoothed_loss =1*l1_losses+1*loss_org

    # selected_smoothed_layer = torch.mean(selected_smoothed_layer,dim=1).unsqueeze(0)
    # selected_smoothed_layer=selected_smoothed_layer.permute(1,0,2,3,4)


    # features_noised = features_noised.view(*features_noised.shape[:-2], -1)
    # selected_smoothed_layer=selected_smoothed_layer.view(*selected_smoothed_layer.shape[:-2], -1)
    # targets_noised=targets_noised.view(*targets_noised.shape[:-2], -1)
    # targets_layer_noised=targets_layer_noised.view(*targets_layer_noised.shape[:-2], -1)

    # # assert 1==0, print((targets_layer_noised.shape,features_noised.shape))
    # if (targets_layer_noised.shape[1])>1:
    #     targets_layer_noised = torch.mean(targets_layer_noised,dim=1).unsqueeze(0)
    #     targets_layer_noised=targets_layer_noised.permute(1,0,2,3)
    # assert (targets_layer_noised.shape==selected_smoothed_layer.shape), print('shape not same')
    # #assert 1==0, print(targets_layer_noised.shape)
    # input_mean_noised=return_mean_value(targets_layer_noised)
    # if len(targets_noised.shape)==3:
    #     targets_noised=torch.mean(targets_noised,dim=0)
    # features_noised=torch.mean(features_noised,dim=0)  
    # lambd= args.sftm_hpyer
    # targets_noised=F.softmax(targets_noised*lambd,dim=1) 
    # targets_layer_noised=F.softmax(targets_layer_noised*lambd,dim=3)  
    # #####kl div
    # features_noised=F.softmax(features_noised*lambd,dim=1)
    # selected_smoothed_layer=F.softmax(selected_smoothed_layer*lambd,dim=3) 
    # selected_smoothed_layer=selected_smoothed_layer[:,0,:,:].permute(1,0,2)
    # targets_layer_noised=targets_layer_noised[:,0,:,:].permute(1,0,2)
    # # # assert 1==0,print(features_noised.shape,selected_smoothed_layer.shape)
    # mean_prob=0.5*(features_noised+targets_noised)
    # smoothed_loss =1*F.kl_div(features_noised.log(), targets_noised, reduction='batchmean')+0*F.kl_div(selected_smoothed_layer.log(), targets_layer_noised, reduction='batchmean')
    # # smoothed_loss = criterion(features_noised*0.9999, targets_noised)
    # ##### crossentropy 
    # # features_noised=features_noised*lambd*2
    # # selected_smoothed_layer=selected_smoothed_layer*lambd*0.5
    # # target_layer_indices = torch.max(targets_layer_noised, dim=3)[1][:,0,:]
    # # target_indices = torch.max(targets_noised, dim=1)[1]
    # # criterion = torch.nn.CrossEntropyLoss()
    # # selected_smoothed_layer = selected_smoothed_layer.squeeze(1).permute(0,2,1)
    # # # features_noised=features_noised.unsqueeze(0)
    # # # target_indices=target_indices.unsqueeze(0)
    # # # assert 1==0, print(target_indices.shape,selected_smoothed_layer.shape)
    # # smoothed_loss=1*criterion(features_noised,target_indices)+1*criterion(selected_smoothed_layer,target_layer_indices)
    # #
    # # smoothed_loss=-1*(features_noised * torch.log(targets_noised + 1e-9)).mean()-5*(selected_smoothed_layer * torch.log(targets_layer_noised + 1e-9)).mean()
    # # smoothed_loss=torch.abs((selected_smoothed_layer*0.9999-targets_layer_noised)/input_mean_noised).mean()
    return smoothed_loss

def Minimize_domain_mean(features,mean,std_dev):
    target= torch.normal(mean, std_dev)
    l2=criterion_mse(features*0.9999, target)#/torch.abs(targets).mean()
    l1=criterion(features*0.9999, target)
    loss =-1*l1-10*l2
    return loss 

def cal_smoothed_loss(features_noised,targets_noised,transform_matrix,mean_std):
    features_tensor=features_noised.transpose(0,1)
    features_data = features_tensor.view(features_tensor.shape[0],features_tensor.shape[1],-1)   # channel*batch*dim 

    targets_tensor=targets_noised.transpose(0,1)
    targets_data = targets_tensor.view(targets_tensor.shape[0],targets_tensor.shape[1],-1)
    transform_matrix_inverse=transform_matrix[1]
    transform_matrix=transform_matrix[0]
    # mean=targets_data.mean(dim=1,keepdim=True)
    # std=targets_data.std(dim=1,keepdim=True)+0.0001
    mean=mean_std[0]
    std=mean_std[1]
    features_data=(features_data-mean)/std
    targets_data=(targets_data-mean)/std
    # assert 1==0,print(features_data.shape,targets_data.shape,transform_matrix.shape)
    feature_reduced=torch.matmul(features_data,transform_matrix)
    feature_recovered=torch.matmul(feature_reduced,transform_matrix_inverse)
    targets_reduced=torch.matmul(targets_data,transform_matrix)
    targets_recovered=torch.matmul(targets_reduced,transform_matrix_inverse)
    # loss=torch.abs(((targets_recovered*std)+mean)-((targets_data*std)+mean)).mean()
    targets_reduced=targets_reduced.mean(dim=1,keepdim=True)
    feature_reduced=feature_reduced.mean(dim=1,keepdim=True)
    targets_recovered=targets_recovered.mean(dim=1,keepdim=True)
    feature_recovered=feature_recovered.mean(dim=1,keepdim=True)
    smoothed_loss=(torch.abs(feature_recovered*0.9999-targets_recovered)).mean()
   
    # print(loss,torch.abs(((targets_data*std)+mean).mean(dim=1,keepdim=True)).mean())
    return smoothed_loss


def process_image(image_path):
    with Image.open(image_path) as img:
        img_resized = img.resize((512, 512))  # 调整图片大小
        img_array = np.asarray(img_resized) / 255.0  # 将图片转换为numpy数组并归一化
    return img_array

# def load_and_transform_images(folder_path):
#     image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.png', '.jpg', '.jpeg'))]
#     transformed_images = []

#     # 使用多线程加载和处理图片
#     with ThreadPoolExecutor() as executor:
#         for img_array in executor.map(process_image, image_paths):
#             transformed_images.append(img_array)

#     transformed_images = np.array(transformed_images)
#     transformed_images = torch.tensor(transformed_images).permute(0, 3, 1, 2).float()  # 调整维度以符合PyTorch的要求
#     return transformed_images


def load_and_transform_images(folder_path, batch_size=4000):
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.png', '.jpg', '.jpeg'))]
    transformed_images = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        with ThreadPoolExecutor() as executor:
            # 使用map来并行处理当前批次中的所有图片
            results = list(executor.map(process_image, batch_paths))
            # 移除因错误而未成功处理的图片
            results = [img for img in results if img is not None]
            batch_images.extend(results)

        # 在每个批次处理完毕后处理和转换图片数据
        if batch_images:
            batch_images = np.array(batch_images)
            batch_images = torch.tensor(batch_images).permute(0, 3, 1, 2).float()
            transformed_images.append(batch_images)

    # 将所有批次合并为一个数据集
    if transformed_images:
        transformed_images = torch.cat(transformed_images, dim=0)
    return transformed_images

# def load_and_transform_images(folder_path):
#     transformed_images = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(('.png', '.jpg', '.jpeg')):
#             image_path = os.path.join(folder_path, filename)
#             with Image.open(image_path) as img:
#                 # 调整图片大小
#                 img_resized = img.resize((512, 512))
#                 # 将图片转换为numpy数组并归一化
#                 img_array = np.asarray(img_resized) / 255.0
#                 transformed_images.append(img_array)
#     transformed_images=np.array(transformed_images)
#     transformed_images=torch.tensor(transformed_images).permute(0,3,1,2).float()
#     return transformed_images

def return_org_mean(model,data):
    feature_list=[]
    for i in range(len(data)):
        input=data[i]
        input=input.unsqueeze(0)
        with torch.no_grad():
            features,features_layer=model.image_encoder(input)
        feature_list.append(features)
        # if i ==1:
            # break
    
    # assert 1==0,print(features.shape)
    feature=torch.stack(feature_list)
    mean = feature.mean(dim=0)
    variance = feature.var(dim=0)
    std_dev = variance.sqrt()
    # mean=sum(feature_list)/len(feature_list)
    
    # assert 1==0,print(mean.shape)
    return mean,std_dev

def return_eta(momentum,step_size):
    upd_momentum=momentum*(step_size/torch.mean(abs(momentum)))
    momentum_binary = torch.where(momentum > 0, torch.tensor(1.0), torch.tensor(0.0))
    eta = torch.clamp(upd_momentum, -1*step_size, -0.25*step_size)*(1-momentum_binary) + momentum_binary*torch.clamp(upd_momentum, 0.25*step_size, 1*step_size)
    return eta

def eta_search(input_tensor,eta,step,loss,model,targets,targets_layer,epsilon):
    var_steps = [-0.5,-0.25,-0.125,0,0.1]
    # var_steps= [-0.5]
    # input_tensor=input_tensor.detach()
    # eta=eta.detach()
    with torch.no_grad():
        step_weight=step/(abs(eta).max())
        adv_input=input_tensor+eta
        upd_input_list=[]
        for var_step in var_steps:
            upd_eta = eta+var_step*step_weight*eta
            upd_eta = torch.clamp(upd_eta, -epsilon, epsilon)
            upd_input=input_tensor+upd_eta
            upd_input_list.append(upd_input)
        upd_input_tensors=torch.stack(upd_input_list,dim=0).squeeze(1)
        upd_input_list.append(adv_input)
        # print(upd_input_tensors.shape)
        loss_list=[]
        features,features_layer=model.image_encoder(upd_input_tensors)
        for i in range(len(features)):
        #    print(features_layer[i].shape)
            new_features_layer = []
            for tensor in features_layer:
                new_tensor = tensor[i, :].unsqueeze(0)  
                new_features_layer.append(new_tensor)
            scale=torch.mean(torch.abs(upd_input_list[i]-input_tensor))/torch.mean(torch.abs(upd_input_list[4]-input_tensor))
            loss_upd=calculate_loss(features[i].unsqueeze(0),targets,new_features_layer,targets_layer)#/(scale**0.5)
            loss_list.append(loss_upd)
        # loss_list.append(loss)
        index= loss_list.index(max(loss_list))
        upd_eta=upd_input_list[index]-input_tensor
        print(index,loss_list,step_weight)
        del upd_input_list,loss_list
    return upd_eta


def get_length(length):
    num_block = 2
    length = int(length)
    rand = torch.rand(num_block)
    rand_norm = torch.round(rand * length / rand.sum()).int()
    max_idx = torch.argmax(rand_norm)
    rand_norm[max_idx] += length - rand_norm.sum()
    return tuple(rand_norm.tolist())

def shuffle_rotate(x,x_c):
    num_block = 2
    n, c, w, h = x.shape
    width_length, height_length = get_length(w), get_length(h)
    while min(width_length)<1 or min(height_length) <1:
         width_length, height_length = get_length(w), get_length(h)

    width_perm = torch.randperm(num_block)
    height_perm = torch.randperm(num_block)

    # Angle calculations simplified for demonstration
    angles = torch.randn(x.size(0)) * 2  # smaller angle for visibility
    angles=angles.int()
    # Split and permute width
    x_split_w = x.split(width_length, dim=2)
    x_w_perm = torch.cat([x_split_w[i] for i in width_perm], dim=2)
    # Split and permute height
    x_split_h_l = [x_split_w[i].split(height_length, dim=3) for i in width_perm]
    x_h_perm_list = []

    x_h_perm = torch.concat([torch.concat([TF.rotate(strip[i], int(angles), interpolation=TF.InterpolationMode.BILINEAR) for i in height_perm],axis=3 ) for strip in x_split_h_l],axis=2)

    # x_c=x
    x_c_split_w = x_c.split(width_length, dim=2)
    x_c_w_perm = torch.cat([x_c_split_w[i] for i in width_perm], dim=2)
    # Split and permute height
    x_c_split_h_l = [x_c_split_w[i].split(height_length, dim=3) for i in width_perm]
    x_c_h_perm_list = []

    x_c_h_perm = torch.concat([torch.concat([TF.rotate(strip[i], int(angles), interpolation=TF.InterpolationMode.BILINEAR) for i in height_perm],axis=3 ) for strip in x_c_split_h_l],axis=2)
    return [x_h_perm,x_c_h_perm]
def BSR(x,x_c,num_copies):
    x_t=[]
    x_c_t=[]
   
    for i in range(num_copies):
       trans_x= shuffle_rotate(x,x_c)
       x_t.append(trans_x[0])
       x_c_t.append(trans_x[1])
    #    print(trans_x[0]==trans_x[1])
    return torch.concat(x_t,axis=0),torch.concat(x_c_t,axis=0)

sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])
sam=sam.to(device)
sam_l, img_embedding_size_l = sam_model_registry['vit_l'](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint='checkpoints/sam_vit_l_0b3195.pth', pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])
if args.mul_modedl==1:
    sam_l=sam_l.cuda()
else:
    sam_l=sam.cuda()
if args.consider_lora:
    pkg = import_module(args.module)
    # net = pkg.LoRA_Sam(sam, args.rank).cuda()
    net = pkg.LoRA_Sam_adv(sam, args.rank).cuda()
    assert args.lora_ckpt is not None
    net.reset_parameters()
    output_path='testset/advimages-test_batch1_loar/'
else:
    output_path='testset/advimages-test_batch1_lowpass/'
output_path=args.out_path
scaler = GradScaler()
dataset_name = args.dataset
dataset_config = {
    'Synapse': {
        'Dataset': Synapse_dataset,
        'volume_path': args.volume_path,
        'list_dir': args.list_dir,
        'num_classes': args.num_classes,
        'z_spacing': 1
    }
}
db_config=dataset_config[dataset_name]
db_test = db_config['Dataset'](base_dir=args.volume_path, list_dir=args.list_dir, split='test_vol')
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4)

# mean=torch.load('mean.pth')
# std_var=torch.load('std_var.pth')
data_org = load_and_transform_images('images')
# print(data_org.shape)
data_org=data_org[:,0,:,:]
print(data_org.shape)
# for batch_data in image_generator("images"):
#     data_org=batch_data[:,0,:,:]
#     print(data_org.shape)

for i_batch, sampled_batch in tqdm(enumerate(testloader)): #every-time one image from testloader
    h, w = sampled_batch['image'].shape[2:]
    # assert 1==0, print(h,w)
    # the shape of the image is 1*512*512
    image, label, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'][0]
    
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy() #transform into batch*h*w
    # assert 1==0,print(image.shape,data_org.shape)
    image=np.concatenate((image[0:0],np.array(data_org)[0:]),axis=0)
    # np.random.shuffle(image)
    # assert 1==0, print(label.shape,np.unique(label[2]))
    if case_name !=args.test_batch and args.test_batch is not None:
            continue
    batch_iter_times=args.batch_iter_time
    loss_thres=args.loss_thres
    attack_iters=args.attack_iter
    common_perturbation = torch.load('common_perb/common_perturbation_meta_0.025sp_bound2_it5batchit20.05upsize_ls0.005mulmd0.pth')
    model_select='b'
    for iters in range(batch_iter_times):
        attack_iters=args.attack_iter
        step_size=args.step_size/255
            # step_size=step_size/((iters+1))
        epsilon=10/255

        if iters%2==1:
            model_select='l'
            # attack_iters=3
            step_size=epsilon/attack_iters
            # sam_l=sam_l.cuda()
            # sam=sam.cpu()
        else:
            model_select='b'
            # sam_l=sam_l.cpu()
            # sam=sam.cuda()s'
        count=0
        np.random.shuffle(image)
        for ind in range(image.shape[0]):  ### iter batch 里面的每张图片
            input_image = image[ind, :, :]
            mask = label[0, :, :]<1
            mask_torch=torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().cuda()
            mask_torch = repeat(mask_torch, 'b c h w -> b (repeat c) h w', repeat=3)
            # print(mask,mask_torch)
            input_image_torch = torch.from_numpy(input_image).unsqueeze(0).unsqueeze(0).float().cuda()
            input_image_torch = repeat(input_image_torch, 'b c h w -> b (repeat c) h w', repeat=3)
            if iters ==0 and ind ==0:
                # common_perturbation = torch.zeros_like(input_image_torch.detach())
                common_perturbation_upd=torch.zeros_like(input_image_torch.detach())
                common_perturbation_l = torch.zeros_like(input_image_torch.detach())
                common_perturbation_b = torch.zeros_like(input_image_torch.detach())
            criterion_mse=torch.nn.MSELoss()
            criterion=torch.nn.L1Loss()
            criterion_cosine = torch.nn.CosineEmbeddingLoss()
            input_image=input_image_torch
            repeat_times=3
            noise_sd=args.noise_sd
            input_image_torch_noise=input_image_torch.repeat(repeat_times*3,1,1,1)
            input_image_torch_noise = input_image_torch_noise+ torch.randn_like(torch.tensor(input_image_torch_noise), device='cuda')*noise_sd

            with torch.no_grad():
                if model_select=='l':
                    targets,targets_layer=sam_l.image_encoder(input_image)
                    noised_targets,noised_targets_layer=sam_l.image_encoder(input_image_torch_noise)
                else:
                    targets,targets_layer=sam.image_encoder(input_image)
                    noised_targets,noised_targets_layer=sam.image_encoder(input_image_torch_noise)      

            max_val,min_val=1,0

            if ind==0:
                momentum = torch.zeros_like(input_image_torch.detach())

            targets_noised,targets_layer_noised=targets,targets_layer
            loss,smoothed_loss=0,0
            smoothed_direction=torch.tensor(0).cuda()
            eta,smoothed_eta=torch.tensor(0).cuda(),torch.tensor(0).cuda()
            bias=0
            eta_old=0
            # input_image_torch = input_image_torch+ torch.randn_like(torch.tensor(input_image_torch), device='cuda')*0.05
            perturb_img = Variable(input_image_torch, requires_grad=True)
            loss_tensor_dict = {}
            # if iters%2==1:
            perturb_img = Variable(input_image_torch.data + common_perturbation, requires_grad=True)
            num_copies=1
            tansform_p=random.uniform(0,1)
            for _ in range(attack_iters):
                start_time = time.time()
                grad = None
                perturb_img_input=perturb_img[0,0,:,:]          
                perturb_img_input = perturb_img_input.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
                avg_grad = torch.zeros_like(perturb_img).detach().cuda()
                # print(perturb_img_input.shape,num_copies)
                if tansform_p>0.7:
                    perturb_img_input_batch,input_image_torch_batch = BSR(perturb_img_input,input_image_torch,num_copies)
                else:
                    perturb_img_input_batch,input_image_torch_batch =perturb_img_input,input_image_torch 
                # perturb_img = Variable(torch.clamp(perturb_img, min_val, max_val), requires_grad=True)                

                if model_select=='l':
                    sam_l.image_encoder.zero_grad()
                else:
                    sam.image_encoder.zero_grad()  
                
                # with torch.no_grad():
                    # noised_targets,noised_targets_layer=sam.image_encoder(input_image_noised)
                
                # perturb_img_input_noised=1*perturb_img_input+(0.1*(4-_))**3*noise
                # for index_bat in range(num_copies):
                if model_select=='l':
                    features,features_layer = sam_l.image_encoder(perturb_img_input_batch[0].unsqueeze(0))
                    with torch.no_grad():
                        targets_c,targets_layer_c=sam_l.image_encoder(input_image_torch_batch[0].unsqueeze(0))
                else:
                    features,features_layer = sam.image_encoder(perturb_img_input_batch[0].unsqueeze(0))
                    with torch.no_grad():
                        targets_c,targets_layer_c=sam.image_encoder(input_image_torch_batch[0].unsqueeze(0))
                
                if model_select=='l':
                    # loss=calculate_loss(features,targets,features_layer,targets_layer)
                    loss=criterion_mse(features*0.9999, targets_c)
                else:
                    loss=criterion_mse(features*0.9999, targets_c)
                if loss > loss_thres:
                    eta = perturb_img.data - input_image_torch.data
                    print('this image fullfill requirement')
                    # if _ ==0:
                    count=count+1
                    break 
                loss_tensor_dict[loss.detach().cpu()] = perturb_img_input.detach().cpu()
                del features,features_layer
                grad=torch.autograd.grad(loss, perturb_img,
                                        retain_graph=True, create_graph=False)[0]
                loss = loss.detach()
                grad=grad / torch.norm(grad, p=1)
                momentum = 0.5 * momentum.detach() +grad# / torch.norm(grad, p=1)
                momentum=momentum.detach()
                eta = return_eta(momentum,step_size)
                eta = 1*step_size * momentum.sign()
                del grad
            
                eta=eta
                perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
                eta = torch.clamp(perturb_img.data - input_image_torch.data, -epsilon, epsilon)
                end_time = time.time()
                backward_time = end_time - start_time
                backward_time = "{:.3f}".format(backward_time)
                print('the loss is:', loss,'the smoothed loss is:', smoothed_loss,'the b_time is:', backward_time, count,'/',len(image),loss_thres,iters)#,'the avg energy is:',avg_energy)
                perturb_img = Variable(input_image_torch.data + eta, requires_grad=True)
                # perturb_img = Variable(torch.clamp(perturb_img, min_val, max_val), requires_grad=True)
            # max_loss = max(loss_tensor_dict.keys())
            # perturb_img = loss_tensor_dict[max_loss]
            # print('the max loss is',max_loss)
            upd_size=(epsilon)/(step_size*attack_iters*image.shape[0])
            common_perturbation = torch.clamp((common_perturbation+upd_size*(eta-common_perturbation)),-epsilon, epsilon)
            print('the max and min value is',common_perturbation.max(),common_perturbation.min())
            adv_noise= common_perturbation.clone().detach().cpu()*255
            save_image(adv_noise, 'universal_perb.png')
            adv_noise = adv_noise.numpy()
            # adv_noise = (adv_noise * 255).astype('uint8')
            del eta,smoothed_eta
            # cv2.imwrite('adv_noise.jpg', adv_noise)

            print('the feature is generated')
            image_data = perturb_img[0,0].detach().cpu().numpy()
            del perturb_img,perturb_img_input
            folder_path=output_path+case_name
            image_path=folder_path+'/'+str(ind)+'.png'

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Folder '{folder_path}' was created.")
            # plt.imsave(image_path, image_data, cmap='gray', vmin=0, vmax=1)
            # print('the image is saved:',image_path)
            torch.cuda.empty_cache()
        if model_select=='l':
            common_perturbation_l=common_perturbation.detach().clone()
        else:
            common_perturbation_b=common_perturbation.detach().clone()
        if iters%2==1:
            common_perturbation_upd=(0.3*common_perturbation_l+0.7*common_perturbation_b)
        if args.mul_modedl==1:
            common_perturbation=common_perturbation_upd
        print(count,'/',len(image))
        if count/len(image)>0.8:
            loss_thres=loss_thres*1.5
        # if count/len(image)<0.3:
        #     loss_thres=loss_thres*0.5
torch.save(common_perturbation.clone().detach(),args.noise_path)
torch.save(momentum.clone().detach(),'momentum_common.pth')
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)



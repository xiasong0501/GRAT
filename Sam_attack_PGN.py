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
import torchvision.transforms as transforms
import time
import cv2
from one_sample_train import one_sample_train
import torch.distributions as dist
from torchvision.utils import save_image
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
parser.add_argument('--UMI_GR', type=int, default=0)
parser.add_argument('--sftm_hpyer', type=int, default=4)
parser.add_argument('--noise_path', type=str, default='common_perb/common_perturbation_meta_0.2sp_bound4_it5batchit20.05upsize_ls0.01mulmd1.pth')
parser.add_argument('--att_step_size', type=float, default=2.0)
parser.add_argument('--att_bound', type=float, default=10.0)
parser.add_argument('--att_numstep', type=int, default=10)
parser.add_argument('--dir_search', type=int, default=0)
parser.add_argument('--random_seed', type=int, default=42)


args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def input_diversity(x):
    resize_rate=0.9
    diversity_prob=0.1
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)

    if resize_rate < 1:
        img_size = img_resize
        img_resize = x.shape[-1]

    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(
        x, size=[rnd, rnd], mode="bilinear", align_corners=False
    )
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left

    padded = F.pad(
        rescaled,
        [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()],
        value=0,
    )

    return padded if torch.rand(1) < diversity_prob else x

def normalize_tensor(input_tensor):
    len_tensor=len(input_tensor)
    min_vals = input_tensor.view(len_tensor, -1).min(dim=1, keepdim=True)[0].view(len_tensor, 1, 1, 1, 1)
    max_vals = input_tensor.view(len_tensor, -1).max(dim=1, keepdim=True)[0].view(len_tensor, 1, 1, 1, 1)

    # Normalizing
    normalized_tensor = (input_tensor - min_vals) / (max_vals - min_vals)
    return normalized_tensor

def return_mean_value(input_tensor):
    input_tensor=torch.abs(input_tensor)
    list_mean=[]
    for i in range(len(input_tensor)):
        list_mean.append(input_tensor[i].mean()) 
    input_mean=torch.stack(list_mean)
    # print(input_mean.shape)
    input_mean=input_mean.view(-1, 1,1,1,1)
    return input_mean


def calculate_loss(features,targets,features_layer,targets_layer,GE):

    loss_org=criterion(features*0.9999, targets)#/torch.abs(targets).mean()
    loss_cosine=-F.cosine_similarity(features.view(1, -1), targets.view(1, -1))
    # loss_mse=criterion_mse(features*0.9999, targets)
    pre_layer= int(len(features_layer)*0.9)
    num_layers=6
    indices = random.sample(range(num_layers), num_layers-2)
    selected_features_layer = torch.stack([features_layer[i+pre_layer-num_layers] for i in indices])
    selected_targets_layer = torch.stack([targets_layer[i+pre_layer-num_layers] for i in indices])
    input_mean=return_mean_value(selected_targets_layer)
    feature_difference= (selected_features_layer*0.999-selected_targets_layer)/input_mean
    Gaussian_est =  torch.randn_like(feature_difference) * 1 + 1
    feature_difference=torch.mul(feature_difference,Gaussian_est)
    l1_losses= (torch.abs(torch.mean(feature_difference,dim=0))).mean()
    if GE == True:
        loss =1*l1_losses+1*loss_org
    else:
        loss =1*loss_org
    targets=targets.view(*targets.shape[:-2], -1)
    features = features.view(*features.shape[:-2], -1)
    features=F.softmax(features,dim=2)
    targets=F.softmax(targets,dim=2)
    info_loss =1*F.kl_div(features.log(), targets, reduction='batchmean')

    return loss+0.00*info_loss


def Minimize_domain_mean(features,mean,std_dev):
    target= torch.normal(mean, std_dev)
    l2=criterion_mse(features*0.9999, target)#/torch.abs(targets).mean()
    l1=criterion(features*0.9999, target)
    loss =-1*l1#-10*l2
    return loss 



def load_and_transform_images(folder_path):
    transformed_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                # 调整图片大小
                img_resized = img.resize((512, 512))
                # 将图片转换为numpy数组并归一化
                img_array = np.asarray(img_resized) / 255.0
                transformed_images.append(img_array)
    transformed_images=np.array(transformed_images)
    transformed_images=torch.tensor(transformed_images).permute(0,3,1,2).float()
    return transformed_images

def return_org_mean(model,data):
    feature_list=[]
    for i in range(len(data)):
        input=data[i]
        input=input.unsqueeze(0)
        input=input[0,0,:,:]
        input=input.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
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

def return_random_img(image):
    n = image.shape[0]
    random_int = random.randrange(0, n)
    img = image[random_int]
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().cuda()
    img = repeat(img, 'b c h w -> b (repeat c) h w', repeat=3)
    return img
# data_org = load_and_transform_images('images-100')
# data_org=data_org.cuda()

def high_pass_enhance(eta):
    laplacian_kernel = torch.tensor([[-0.125, -0.125, -0.125],
                                 [-0.125, 1, -0.125],
                                 [-0.125, -0.125, -0.125]], dtype=torch.float32).cuda()
    laplacian_kernel = laplacian_kernel.expand(3, 1, 3, 3)
    eta_high = F.conv2d(eta, laplacian_kernel, padding=1, groups=3)
    return 1*eta+1*eta_high



sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])



sam=sam.to(device)
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




for i_batch, sampled_batch in tqdm(enumerate(testloader)): #every-time one image from testloader
    h, w = sampled_batch['image'].shape[2:]
    # assert 1==0, print(h,w)
    # the shape of the image is 1*512*512
    image, label, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'][0]
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy() #transform into batch*h*w
    # assert 1==0, print(label.shape,np.unique(label[2]))
    
    if case_name !=args.test_batch and args.test_batch is not None:
            continue
    trans_analyze_path = 'trans_analyze.txt'
    with open(trans_analyze_path, 'a') as file:
        file.write('\nthe result for PGN log\n')
        file.write(str(args.UMI_GR))
        file.write('\n')
    for ind in range(image.shape[0]):
        input_image = image[ind, :, :]
        mask = label[ind, :, :]<1
        mask_torch=torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().cuda()
        mask_torch = repeat(mask_torch, 'b c h w -> b (repeat c) h w', repeat=3)
        # print(mask,mask_torch)
        input_image_torch = torch.from_numpy(input_image).unsqueeze(0).unsqueeze(0).float().cuda()
        input_image_torch = repeat(input_image_torch, 'b c h w -> b (repeat c) h w', repeat=3)
        criterion_mse=torch.nn.MSELoss()
        criterion=torch.nn.L1Loss()
        criterion_cosine = torch.nn.CosineEmbeddingLoss()

        input_image=input_image_torch


        with torch.no_grad():
            infer_time= time.time()
            if args.consider_lora:
                targets,targets_layer=net.sam.image_encoder(input_image)

            else:
                targets,targets_layer=sam.image_encoder(input_image)
    
            end_infer_time= time.time()
            print('infer_time_is:',(end_infer_time-infer_time))

        
        common_perturbation = torch.load(args.noise_path)
        max_val,min_val=(input_image_torch.max(),input_image_torch.min())
        step_size=(input_image_torch.max()-input_image_torch.min())*(args.att_step_size)/255
        epsilon=(input_image_torch.max()-input_image_torch.min())*(args.att_bound)/255
        # assert 1==0, print(max_val,min_val) 
        if ind==0:
            momentum = torch.zeros_like(input_image_torch.detach())
        # momentum =momentum
        momentum_old = momentum
        if args.consider_lora:
            net.reset_parameters()
        # targets=torch.randn_like(targets, device='cuda') * 0.15
        targets_noised,targets_layer_noised=targets,targets_layer
        loss,smoothed_loss=0,0
        smoothed_direction=torch.tensor(0).cuda()
        eta,smoothed_eta=torch.tensor(0).cuda(),torch.tensor(0).cuda()
        eta_old=0
        perturb_img = Variable(input_image_torch, requires_grad=True)
        loss_tensor_dict = {}
        if args.UMI_GR==1:
            meta_ini = True
            GE = True
        else:
            meta_ini = False
            GE = False   
        num_rep= 8
        zeta=0.5
        delta=0.5

        for _ in range(args.att_numstep):

            avg_grad = torch.zeros_like(perturb_img).detach().cuda()
            start_time = time.time()
            for rep_num in range(num_rep):
            
                grad = None
                perturb_img_input=perturb_img[0,0,:,:] + torch.rand_like(perturb_img[0,0,:,:]).uniform_(-epsilon*zeta, epsilon*zeta)
               

                if args.consider_lora:
                    net.sam.image_encoder.zero_grad()
                else:
                    sam.image_encoder.zero_grad()  
                
                perturb_img_input = perturb_img_input.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
                if _ ==0 and args.UMI_GR==0:
                    perturb_img_input=perturb_img_input+torch.randn_like(perturb_img_input)*0.0001
                if args.consider_lora:
                    features,features_layer = net.sam.image_encoder(perturb_img_input)
                else:
                    features,features_layer = sam.image_encoder(perturb_img_input)
                    
                loss=calculate_loss(features,targets,features_layer,targets_layer,GE)
                loss_l2=criterion_mse(features, targets).detach().cpu()
                # loss_r=calculate_loss(features_r,targets_rand,features_layer_r,targets_layer_rand)
                loss = loss#+0.3*loss_r
                loss_tensor_dict[loss.detach().cpu()] = perturb_img_input.detach().cpu()
                del features,features_layer
               

                g1= torch.autograd.grad(loss, perturb_img,
                                        retain_graph=False, create_graph=False)[0]
                perturb_img_input_star = perturb_img_input.detach() + step_size * (-g1)/torch.abs(g1).mean([1, 2, 3], keepdim=True)
                loss = loss.detach()

                nes_perturb_img_input = perturb_img_input_star.detach()
                nes_perturb_img_input = Variable(nes_perturb_img_input, requires_grad = True)
                features,features_layer = sam.image_encoder(nes_perturb_img_input)
                loss=calculate_loss(features,targets,features_layer,targets_layer,GE)
                g2 = torch.autograd.grad(loss, nes_perturb_img_input,
                                            retain_graph=False, create_graph=False)[0]

                avg_grad += (1-delta)*g1 + delta*g2
                del features,features_layer
            
            eta= (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True).detach()
            eta = 0.5 * momentum + eta
            momentum = eta
                # smoothed_eta=smoothed_eta*0.7
            del g1,g2,avg_grad
            eta=step_size * eta.sign()
            # eta=eta
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - input_image_torch.data, -epsilon, epsilon)
            # print(eta.max())
            if _==0 and meta_ini:
                eta = 2.5*eta+1*common_perturbation
                # eta=eta_l
            if (_ ==12 and meta_ini):
                perturb_img = Variable(input_image_torch.data + 0.25*eta, requires_grad=True)
                eta = torch.clamp(perturb_img.data - input_image_torch.data, -epsilon, epsilon)

            # eta = return_proj(eta.detach(),(common_perturbation/ torch.norm(common_perturbation, p=1)))
            end_time = time.time()
            backward_time = end_time - start_time
            backward_time = "{:.3f}".format(backward_time)
            start_time = time.time()
            # eta = eta_search(input_image_torch,eta,step_size,loss,sam,targets,targets_layer,epsilon)
            end_time = time.time()
            search_time = end_time - start_time
            search_time = "{:.3f}".format(search_time)
            print('the loss is:', loss.item(),'the b_time is:', backward_time)#,'the avg energy is:',avg_energy)
            eta = torch.clamp(eta, -epsilon, epsilon)
            perturb_img = Variable(input_image_torch.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, min_val, max_val), requires_grad=True)

        trans_analyze_path = 'trans_analyze.txt'
        with open(trans_analyze_path, 'a') as file:
            file.write(str(loss_l2.item()))
            file.write('\n')
        adv_noise= eta.clone().detach().cpu()*255
        save_image(adv_noise, 'output_image.png')
        adv_noise = adv_noise.numpy()
        adv_noise = (adv_noise * 255).astype('uint8')
        del eta,smoothed_eta
        cv2.imwrite('adv_noise.jpg', adv_noise)

        print('the feature is generated')
        image_data = perturb_img[0,0].detach().cpu().numpy()
        del perturb_img,perturb_img_input
        # image_data =image_data.astype(np.uint8)
        # image = Image.fromarray(image_data.transpose(1, 2, 0)) 
        folder_path=output_path+case_name
        image_path=folder_path+'/'+str(ind)+'.png'

        if not os.path.exists(folder_path):
            # 如果文件夹不存在，创建它
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' was created.")
        # image.save(image_path)
        plt.imsave(image_path, image_data, cmap='gray', vmin=0, vmax=1)
        print('the image is saved:',image_path)
        torch.cuda.empty_cache()
    # break

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



import os
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
import imageio
from einops import repeat
from icecream import ic
from torch.autograd import Variable

def return_adv_img(input_image,input_image_clean,net):
        net.eval()
        # input_image = image[ind, :, :]
        input_image_torch_clean = torch.from_numpy(input_image_clean).unsqueeze(0).unsqueeze(0).float().cuda()
        input_image_torch_clean = repeat(input_image_torch_clean, 'b c h w -> b (repeat c) h w', repeat=3)
        input_image_torch = torch.from_numpy(input_image).unsqueeze(0).unsqueeze(0).float().cuda()
        input_image_torch = repeat(input_image_torch, 'b c h w -> b (repeat c) h w', repeat=3)
        criterion=torch.nn.MSELoss()
        perturb_img = Variable(input_image_torch, requires_grad=True)#modify the shape of image, x=(x-mean)/std
        input_image=input_image_torch
        with torch.no_grad():
            targets,targets_layer=net.sam.image_encoder(input_image_torch_clean)
        max_val,min_val=(input_image_torch.max(),input_image_torch.min())
        step_size=(input_image_torch.max()-input_image_torch.min())*1/255
        epsilon=(input_image_torch.max()-input_image_torch.min())*8/255
        
        for _ in range(1):
            perturb_img_input=perturb_img
            opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            net.sam.image_encoder.zero_grad()
            if isinstance(criterion, torch.nn.MSELoss):
                perturb_img_input=perturb_img_input[:,0,:,:]
                perturb_img_input = perturb_img_input.unsqueeze(1).repeat(1, 3, 1, 1)
                features,features_layer = net.sam.image_encoder(perturb_img_input)
                loss = criterion(features, targets*0.9999)
            print('the loss is:', loss)
            # losses.append(loss.detach())
            loss.backward(retain_graph=True)
            eta = step_size * perturb_img.grad.data.sign()
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - input_image_torch.data, -epsilon, epsilon)
            perturb_img = Variable(input_image_torch.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, min_val, max_val), requires_grad=True)
        # print('aver')
        image_data = perturb_img_input[0,0].detach().cpu().numpy()
        del perturb_img,perturb_img_input
        return image_data,loss.detach()

class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calc focal loss
        :param preds: size: [B, N, C] or [B, C], corresponds to detection and classification tasks  [B, C, H, W]: segmentation
        :param labels: size: [B, N] or [B]  [B, H, W]: segmentation
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.low(1 - preds_softmax) == (1 - pt) ** r

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes




def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, image_clean, label, net, classes, multimask_output, patch_size=[256, 256], input_size=[224, 224],
                       test_save_path=None, case=None, z_spacing=1):
    image, image_clean, label = image.squeeze(0).cpu().detach().numpy(),image_clean.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    trans_analyze_path = 'trans_analyze.txt'
    with open(trans_analyze_path, 'a') as file:
        file.write('\nthe test result\n')
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        losses=[]
        for ind in range(image_clean.shape[0]):
            slice_clean=image_clean[ind, :, :]
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            # if x != input_size[0] or y != input_size[1]:
            #     slice = zoom(slice, (input_size[0] / x, input_size[1] / y), order=3)  # previous using 0

            # new_x, new_y = slice.shape[0], slice.shape[1]  # [input_size[0], input_size[1]]
            # if new_x != patch_size[0] or new_y != patch_size[1]:
            #     slice = zoom(slice, (patch_size[0] / new_x, patch_size[1] / new_y), order=3)  # previous using 0, patch_size[0], patch_size[1]
           
            slice,loss=return_adv_img(slice,slice_clean,net)
            trans_analyze_path = 'trans_analyze.txt'
            with open(trans_analyze_path, 'a') as file:
                file.write(str(loss.item()))
                file.write('\n')
            inputs = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
            # inputs=inputs+torch.randn_like(inputs, device='cuda') * 0.25
            net.eval()
            with torch.no_grad():
                outputs = net(inputs, multimask_output, patch_size[0])
                output_masks = outputs['masks']
                # assert 1==0, print(outputs['low_res_logits'].shape)
                out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                out_h, out_w = out.shape
                if x != out_h or y != out_w:
                    pred = zoom(out, (x / out_h, y / out_w), order=0)
                else:
                    pred = out
                prediction[ind] = pred
            losses.append(loss)
        # only for debug
        # if not os.path.exists('/output/images/pred'):
        #     os.makedirs('/output/images/pred')
        # if not os.path.exists('/output/images/label'):
        #     os.makedirs('/output/images/label')
        # assert prediction.shape[0] == label.shape[0]
        # for i in range(label.shape[0]):
        #     imageio.imwrite(f'/output/images/pred/pred_{i}.png', prediction[i])
        #     imageio.imwrite(f'/output/images/label/label_{i}.png', label[i])
        # temp = input('kkpsa')
        print('the average loss is',(sum(losses)/len(losses)))
    else:
        x, y = image.shape[-2:]
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
        inputs = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)
    metric_list = []
    for i in range(1, classes + 1):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
    return metric_list,(sum(losses)/len(losses))

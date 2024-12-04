import nibabel as nib
import matplotlib.pyplot as plt
import os

nii_gt = nib.load('outputs/predictions/case0008_gt.nii.gz')
data_array_gt = nii_gt.get_fdata()

nii_img_pred = nib.load('outputs/predictions/case0008_pred.nii.gz')
data_array_pred=nii_img_pred.get_fdata()
nii_img = nib.load('outputs/predictions/case0008_img.nii.gz')
data_array_img = nii_img.get_fdata()


nii_samadv_img_pred = nib.load('output/adv_sam_lora/predictions/case0008_pred.nii.gz')
data_samarray_adv_pred=nii_samadv_img_pred.get_fdata()
nii_samadv_img = nib.load('output/adv_sam_lora/predictions/case0008_img.nii.gz')
data_samarray_adv_img = nii_samadv_img.get_fdata()


nii_loraadv_img_pred = nib.load('output/adv_sam_lora/predictions/case0008_pred.nii.gz')
data_loraarray_adv_pred=nii_loraadv_img_pred.get_fdata()
nii_loraadv_img = nib.load('output/adv_sam_lora/predictions/case0008_img.nii.gz')
data_loraarray_adv_img = nii_loraadv_img.get_fdata()

for i in range(data_array_img.shape[2]):
    # if len(np.unique(data_array_gt[:, :, i]))<=1:
        # continue

    fig, axes = plt.subplots(2, 2, figsize=(5, 5))

    axes[0,0].imshow(data_array_gt[:, :, i], cmap='gray')
    axes[0,0].axis('off')
    axes[0,0].set_title('ground_truth')


    axes[1,0].imshow(data_array_pred[:, :, i], cmap='gray')
    axes[1,0].axis('off')
    axes[1,0].set_title('clean_pred')


    axes[0,1].imshow(data_samarray_adv_img[:, :, i], cmap='gray')
    axes[0,1].axis('off')
    axes[0,1].set_title('att_sam_img')

    axes[1,1].imshow(data_samarray_adv_pred[:, :, i], cmap='gray')
    axes[1,1].axis('off')
    axes[1,1].set_title('att_sam_pred')

    # axes[0,2].imshow(data_loraarray_adv_img[:, :, i], cmap='gray')
    # axes[0,2].axis('off')
    # axes[0,2].set_title('att_lora_img')
    # # fig.delaxes(axes[2,1])


    # axes[1,2].imshow(data_loraarray_adv_pred[:, :, i], cmap='gray')
    # axes[1,2].axis('off')
    # axes[1,2].set_title('att_lora_pred')
    # # fig.delaxes(axes[2,1])
    
    plt.tight_layout()
    out_dir='visualized_attack/attack_nolora/' 
    if not os.path.exists(out_dir):
        # 如果文件夹不存在，创建它
        os.makedirs(out_dir)
        print(f"Folder '{out_dir}' was created.")
    img_path=out_dir+'gt'+str(i)+'.png'
    plt.savefig(img_path,dpi=200)
    print(img_path)
    #plt.show()
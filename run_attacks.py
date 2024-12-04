import subprocess

noise_sd =  0
smoothed_momentum = 0
outpath= 'testset/advimages_baseline/'

#hyperparameters for UMI
loss_threses = [0.0025]
batch_iter_times = [4]
attack_iters=[3]
attack_iter=3
step_sizes = [0.025]
mul_modedls = [0]
upd_sizes = [0]

#selected the the basic attack strategy
methods = ['MI-FGSM']

#modify this to decide whether to use our proposed method 
UMI_GRs=[1]

sftm_hpyer=26 #

# the hyperparameter for adversarial noise; step_size,bound and num steps
att_step_size= 2 
att_bound= 10
att_numstep = 10

device = 0
 

lora_ckpts=['checkpoints/epoch_159.pth']# those are checkpoints by adversarial training: ['checkpoints/epoch_5_ld0.1_it1_bd10_out0.pth','checkpoints/epoch_5_ld0.1_it1_bd10_out1.pth','checkpoints/epoch_0_lb0.5_bd10_it1_out1.pth','checkpoints/epoch_2_ld0.5_bd10_iter1_out0.pth','checkpoints/epoch_5_ld0.1_bd10_it5_out0.pth','checkpoints/epoch_5_ld0.5_it5_bd10_out0.pth']
random_seeds=[100]

for mul_modedl in mul_modedls:
    for upd_size in upd_sizes:
        for loss_thres in loss_threses:
            for step_size in step_sizes:
                for batch_iter_time in batch_iter_times:

                        # this part calculate the univerisal meta initialization
                        #########################################################
                        noise_path = 'common_perb/common_perturbation_meta_0.025sp_bound2_it7batchit41upsize_ls0.0025mulmd0_10000.pth'
                        command = f"CUDA_VISIBLE_DEVICES=1 python Sam_attack_universal.py --test_batch {'case0008'}  --out_path {outpath} --noise_sd {noise_sd} --smoothed_momentum {smoothed_momentum}  --sftm_hpyer {sftm_hpyer} --loss_thres {loss_thres} --batch_iter_time {batch_iter_time} --attack_iter {attack_iter} --mul_modedl {mul_modedl} --upd_size {upd_size} --step_size {step_size} --noise_path {noise_path}"
                        
                        # comments this line if you use the Universal noise proposed by us
                        # subprocess.run(command, shell=True)
                        
                        # this part generate the adversarial examples
                        #########################################################                       
                        for method in methods:
                            for UMI_GR in UMI_GRs:
                                for i in range(len(random_seeds)):
                                    random_seed=random_seeds[i]
                                    if method == 'Anda':
                                        outpath= 'testset/advimages_baselines/Anda'+str(UMI_GR)+'/'
                                        command_gen = f"CUDA_VISIBLE_DEVICES={device} python Sam_attack_Anda.py --out_path {outpath}  --noise_sd {noise_sd} --smoothed_momentum {smoothed_momentum}\
                                        --UMI_GR {UMI_GR} --sftm_hpyer {sftm_hpyer} --noise_path {noise_path} --att_step_size {att_step_size} --att_bound {att_bound} --dir_search {0} --att_numstep {att_numstep} --random_seed {random_seed}"
                                    if method == 'L2T':
                                        outpath= 'testset/advimages_baselines/L2T'+str(UMI_GR)+'/'
                                        command_gen = f"CUDA_VISIBLE_DEVICES={device} python Sam_attack_L2T.py --out_path {outpath}  --noise_sd {noise_sd} --smoothed_momentum {smoothed_momentum}\
                                        --UMI_GR {UMI_GR} --sftm_hpyer {sftm_hpyer} --noise_path {noise_path} --att_step_size {att_step_size} --att_bound {att_bound} --dir_search {0} --att_numstep {att_numstep} --random_seed {random_seed}"
                                    if method == 'PGN':
                                        outpath= 'testset/advimages_baselines/PGN'+str(UMI_GR)+'/'
                                        command_gen = f"CUDA_VISIBLE_DEVICES={device} python Sam_attack_PGN.py --out_path {outpath}  --noise_sd {noise_sd} --smoothed_momentum {smoothed_momentum} \
                                        --UMI_GR {UMI_GR} --sftm_hpyer {sftm_hpyer} --noise_path {noise_path} --att_step_size {att_step_size} --att_bound {att_bound} --dir_search {0} --att_numstep {att_numstep} --random_seed {random_seed}"
                                    if method == 'BSR':
                                        outpath= 'testset/advimages_baselines/BSR'+str(UMI_GR)+'/'
                                        command_gen = f"CUDA_VISIBLE_DEVICES={device} python Sam_attack_BSR.py --out_path {outpath}  --noise_sd {noise_sd} --smoothed_momentum {smoothed_momentum} \
                                        --UMI_GR {UMI_GR} --sftm_hpyer {sftm_hpyer} --noise_path {noise_path} --att_step_size {att_step_size} --att_bound {att_bound} --dir_search {0} --att_numstep {att_numstep} --random_seed {random_seed}"
                                    if method == 'MI-FGSM':
                                        outpath= 'testset/advimages_baselines/MI-FGSM'+str(UMI_GR)+'/'
                                        command_gen = f"CUDA_VISIBLE_DEVICES={device} python Sam_attack.py --test_batch {'case0008'} --out_path {outpath}  --noise_sd {noise_sd} --smoothed_momentum {smoothed_momentum} \
                                        --UMI_GR {UMI_GR} --sftm_hpyer {sftm_hpyer} --noise_path {noise_path} --att_step_size {att_step_size} --att_bound {att_bound} --dir_search {0} --att_numstep {att_numstep} --random_seed {random_seed}"
                                        # subprocess.run(command_gen, shell=True)
                                    if method == 'DMI-FGSM':
                                        outpath= 'testset/advimages_baselines/DMI-FGSM'+str(UMI_GR)+'/'
                                        command_gen = f"CUDA_VISIBLE_DEVICES={device} python Sam_attack_DMI.py --out_path {outpath}  --noise_sd {noise_sd} --smoothed_momentum {smoothed_momentum}\
                                        --UMI_GR {UMI_GR} --sftm_hpyer {sftm_hpyer} --noise_path {noise_path} --att_step_size {att_step_size} --att_bound {att_bound} --dir_search {0} --att_numstep {att_numstep} --random_seed {random_seed}"

                                    
                                    subprocess.run(command_gen, shell=True)

                        # this part evaluate the fine-tuned model under adversarial attacks
                        #########################################################   
                        for lora_ckpt in lora_ckpts:
                            for method in methods:
                                for UMI_GR in UMI_GRs:
                                    for i in range(len(random_seeds)):
                                        random_seed=random_seeds[i]
                                        if method == 'Anda':
                                            outpath= 'testset/advimages_baselines/Anda'+str(UMI_GR)+'/'
                                        if method == 'L2T':
                                            outpath= 'testset/advimages_baselines/L2T'+str(UMI_GR)+'/'
                                    
                                        if method == 'PGN':
                                            outpath= 'testset/advimages_baselines/PGN'+str(UMI_GR)+'/'
                                            
                                        if method == 'BSR':
                                            outpath= 'testset/advimages_baselines/BSR'+str(UMI_GR)+'/'
                                           
                                        if method == 'MI-FGSM':
                                            outpath= 'testset/advimages_baselines/MI-FGSM'+str(UMI_GR)+'/'
                                                                                    
                                        if method == 'ILPD':
                                            outpath= 'testset/advimages_baselines/ILPD'+str(UMI_GR)+'/'
                                           
                                        if method == 'DMI-FGSM':
                                            outpath= 'testset/advimages_baselines/DMI-FGSM'+str(UMI_GR)+'/'

                                        
                                        exp_name= noise_path+'_'+method+'_umi_gr'+str(UMI_GR)
                                        test_command = f"CUDA_VISIBLE_DEVICES={device} python test_adv_all.py --is_savenii --test_batch {'case0008'} --out_path {outpath} --noise_sd {noise_sd} --smoothed_momentum {smoothed_momentum}\
                                        --smoothed_freq {UMI_GR} --sftm_hpyer {sftm_hpyer} --noise_path {exp_name} --att_step_size {att_step_size} --att_bound {att_bound} --att_numstep {att_numstep} --lora_ckpt {lora_ckpt}"
                                        subprocess.run(test_command, shell=True)
                                        



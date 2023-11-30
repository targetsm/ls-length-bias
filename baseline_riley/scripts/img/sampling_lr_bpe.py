import sys
import math
import numpy as np
import matplotlib.pyplot as plt


data_ls = []
data_nols = []
data_ls_paper = []
data_nols_paper = []
labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']
error_ls = []
error_nols = []
for s in labels:
    for var in ['nols', 'ls']:
        f_nobpe = open(f'/cluster/scratch/ggabriel/baseline_riley/s_{s}/evaluation/sampling/{var}/generate_test_processed_nobpe.txt').readlines()
        f_bpe = open(f'/cluster/scratch/ggabriel/baseline_riley/s_{s}/evaluation/sampling/{var}/generate_test_processed_bpe.txt').readlines()
        total_hyp_sum = 0
        total_ref_sum = 0
        total_ratio_sum = 0
        total_num = 0
        for i in range(len(f_nobpe)):
            l1 = eval(f_nobpe[i])
            l2 = eval(f_bpe[i])
            l = [x-y+1 for (x,y) in zip(l2, l1)]
            src = l[0]
            ref = l[1]
            b = ref // 100
            hyp_sum = sum(l[2:])
            hyp_avg = hyp_sum/1000
            ratio = hyp_avg/ref
            ref_sum = 1000*ref
            total_hyp_sum += hyp_sum
            total_ref_sum += ref_sum
            total_ratio_sum += ratio
            total_num +=1
        print('lr_paper:', total_hyp_sum/total_ref_sum)
        print('lr_avg:', total_ratio_sum/total_num)
        if var == 'nols':
            data_nols.append(total_ratio_sum/total_num)
            data_nols_paper.append(total_hyp_sum/total_ref_sum)
        else:
            data_ls.append(total_ratio_sum/total_num)
            data_ls_paper.append(total_hyp_sum/total_ref_sum)
            

        for i in range(len(f_nobpe)):
            l1 = eval(f_nobpe[i])
            l2 = eval(f_bpe[i])
            l = [x-y+1 for (x,y) in zip(l2, l1)]
            src = l[0]
            ref = l[1]
            b = ref // 100
            hyp_sum = sum(l[2:])
            hyp_avg = hyp_sum/1000
            ratio = hyp_avg/ref
            ref_sum = 1000*ref
            std_dev = (ratio - total_ratio_sum/total_num)**2
        print('lr_std:', math.sqrt(std_dev/total_num)*(2.58))
        print('lr_se:', math.sqrt(std_dev/total_num)/math.sqrt(total_num)*(2.58))
        if var == 'nols':
            error_nols.append(math.sqrt(std_dev/total_num)/math.sqrt(total_num)*(2.58))
        else:
            error_ls.append(math.sqrt(std_dev/total_num)/math.sqrt(total_num)*(2.58))

plt.plot(labels, data_nols, '-o', color="darkgreen", label="lr_avg without LS")
plt.plot(labels, data_ls, '-o', color="darkred", label="lr_avg with LS")
plt.plot(labels, data_nols_paper, '--o', color="darkgreen", label='lr_paper without LS')
plt.plot(labels, data_ls_paper, '--o', color="darkred", label='lr_paper with LS')


plt.ylim(0.5, 1.2)
plt.yticks(np.arange(0.5, 1.2, 0.1))
plt.grid()
plt.title('Average sampling length ratio with bpe')
plt.ylabel('Lenght ratio')
plt.xlabel('Source sentence %')
plt.legend(loc="lower right")
plt.errorbar(labels, data_nols, error_nols, capsize=4, color='black', linestyle='none')
plt.errorbar(labels, data_ls, error_ls, capsize=4, color='black', linestyle='none')
plt.savefig('sampling_lr_bpe.png', dpi=400)

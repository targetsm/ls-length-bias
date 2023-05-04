import sys
import math
import numpy as np

data_ls = []
data_nols = []
labels = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']
error_ls = []
error_nols = []
for s in labels:
    for var in ['nols', 'ls']:
        f_nobpe = f'/cluster/scratch/ggabriel/baseline_riley/s_{s}/evaluation/sampling/{var}/generate_test_processed_nobpe.txt'
        f_bpe = f'/cluster/scratch/ggabriel/baseline_riley/s_{s}/evaluation/sampling/{var}/generate_test_processed_bpe.txt'
        for f in [f_bpe]:#, f_nobpe]:
            total_hyp_sum = 0
            total_ref_sum = 0
            total_ratio_sum = 0
            total_num = 0
            buckets = []
            with open(f) as f_o:
                for line in f_o:
                    l = eval(line)
                    src = l[0]
                    ref = l[1]
                    b = ref // 100
                    hyp_sum = sum(l[2:])
                    hyp_avg = hyp_sum/1000
                    ratio = hyp_avg/ref
                    ref_sum = 1000*ref
                    while len(buckets) <= b:
                        buckets.append([0,0,0,0])
                    buckets[b][0] += hyp_sum
                    buckets[b][1] += ref_sum
                    buckets[b][2] += ratio
                    buckets[b][3] += 1
                    total_hyp_sum += hyp_sum
                    total_ref_sum += ref_sum
                    total_ratio_sum += ratio
                    total_num +=1
                f_o.close()
            print(f)
            print('lr_paper:', total_hyp_sum/total_ref_sum)
            print('lr_avg:', total_ratio_sum/total_num)
            if var == 'nols':
                data_nols.append(total_ratio_sum/total_num)
            else:
                data_ls.append(total_ratio_sum/total_num)
            

            #data = []
            #for i in range(len(buckets)):
                #print(f'b {i*100} - {(i+1)*100}:')
                #print('lr_paper:', buckets[i][0]/buckets[i][1])
                #print('lr_avg:', buckets[i][2]/buckets[i][3])
                #data.append(buckets[i][2]/buckets[i][3])
            buckets_std = []

            for line in open(f):
                l = eval(line)
                src = l[0]
                ref = l[1]
                b = ref // 100
                hyp_sum = sum(l[2:])
                hyp_avg = hyp_sum/1000
                ratio = hyp_avg/ref
                ref_sum = 1000*ref
                while len(buckets_std) <= b:
                    buckets_std.append([0,0,0,0])
                buckets_std[b][0] += hyp_sum
                buckets_std[b][1] += ref_sum
                buckets_std[b][2] += (ratio - buckets[b][2]/buckets[b][3])**2
                buckets_std[b][3] += 1
                std_dev = (ratio - total_ratio_sum/total_num)**2
            print('lr_std:', math.sqrt(std_dev/total_num)*(1e+3))
            print('lr_se:', math.sqrt(std_dev/total_num)/math.sqrt(total_num)*(1e+3))
            #labels = []
            error = []
            for i in range(len(buckets_std)):
                #print(f'b {i*100} - {(i+1)*100}:')
                #labels.append(f'b {i*100} - {(i+1)*100}:')
                #print('lr_std:', math.sqrt(buckets_std[i][2]/buckets_std[i][3]))
                #print('lr_se:', math.sqrt(buckets_std[i][2]/buckets_std[i][3])/math.sqrt(buckets_std[i][3])) 
                error.append(math.sqrt(buckets_std[i][2]/buckets_std[i][3])/math.sqrt(buckets_std[i][3]))
            if var == 'nols':
                error_nols.append(math.sqrt(std_dev/total_num)/math.sqrt(total_num)*(1e+3))
            else:
                error_ls.append(math.sqrt(std_dev/total_num)/math.sqrt(total_num)*(1e+3))
import matplotlib.pyplot as plt
plt.plot(labels, data_nols, '--bo', color="darkgreen", label="lr_avg without LS")
plt.plot(labels, data_ls, '--bo', color="darkred", label="lr_avg with LS")
plt.ylim(0.5, 1.2)
plt.yticks(np.arange(0.5, 1.2, 0.1))
plt.grid()
plt.title('Scaled standard error for average sampling length ratio with bpe')
plt.ylabel('Lenght ratio')
plt.xlabel('Source sentence %')
plt.legend(loc="lower right")
plt.errorbar(labels, data_nols, error_nols, marker='o', capsize=4, color='green', markersize=4, linewidth=1, linestyle='-')
plt.errorbar(labels, data_ls, error_ls, marker='o', capsize=4, color='red', markersize=4, linewidth=1, linestyle='-')
plt.savefig('figure2_bpe.png', dpi=400)
        
'''import matplotlib.pyplot as plt
            x=labels
            y=data
            yerr=error
            width = 10
            height = 8
            plt.figure(figsize=(width, height))
            plt.bar(x, y)#, color=['lightgray','plum','plum','plum','lightblue','lightblue','lightblue','palegreen','palegreen','palegreen', 'khaki', 'khaki', 'khaki'])
            #plt.title('PNT1A Cell viability: Chlorpromazine, Filipin, Wortmannin and SB421543')
            #plt.xlabel('Inhibitor concentration (uM)')
            #plt.ylabel('Viability %')
            plt.errorbar(x, y, yerr, fmt='.', color='Black')#, elinewidth=2,capthick=10,errorevery=1, alpha=0.5, ms=4, capsize = 2)
            plt.savefig('figure.png', dpi=400)
            #exit()'''

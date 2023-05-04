import sys
import math
import numpy as np

data_ls = []
data_nols = []
labels = ['100']
error_ls = []
error_nols = []
for s in labels:
    for var in ['nols', 'ls']:
        f_nobpe = f'/cluster/home/ggabriel/ls-length-bias/baseline_riley/s_{s}/evaluation/sampling/{var}/generate_test_processed_nobpe.txt'
        f_bpe = f'/cluster/home/ggabriel/ls-length-bias/baseline_riley/s_{s}/evaluation/sampling/{var}/generate_test_processed_bpe.txt'
        for f in [f_nobpe]:#, f_nobpe]:
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
                #if var == 'nols':
                #    data_nols.append(total_ratio_sum/total_num)
                #else:
                #    data_ls.append(total_ratio_sum/total_num)
            

            for i in range(len(buckets)):
                print(f'b {i*100} - {(i+1)*100}:')
                print('lr_paper:', buckets[i][0]/buckets[i][1])
                print('lr_avg:', buckets[i][2]/buckets[i][3])
                if var == 'nols':
                    data_nols.append(buckets[i][2]/buckets[i][3])
                else:
                    data_ls.append(buckets[i][2]/buckets[i][3])

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
            #print('lr_std:', math.sqrt(std_dev/total_num)*(1e+3))
            #print('lr_se:', math.sqrt(std_dev/total_num)/math.sqrt(total_num)*(1e+3))
            labels = []
            #error = []
            for i in range(len(buckets_std)):
                print(f'b {i*100} - {(i+1)*100}:')
                labels.append(f'b {i*100} - {(i+1)*100}:')
                print('lr_std:', math.sqrt(buckets_std[i][2]/buckets_std[i][3]))
                print('lr_se:', math.sqrt(buckets_std[i][2]/buckets_std[i][3])/math.sqrt(buckets_std[i][3])) 
                #error.append(math.sqrt(buckets_std[i][2]/buckets_std[i][3])/math.sqrt(buckets_std[i][3])*(1e+1))
                if var == 'nols':
                    error_nols.append(math.sqrt(buckets_std[i][2]/buckets_std[i][3])/math.sqrt(buckets_std[i][3])*(1e+1))
                else:
                    error_ls.append(math.sqrt(buckets_std[i][2]/buckets_std[i][3])/math.sqrt(buckets_std[i][3])*(1e+1))

import matplotlib.pyplot as plt
x=labels
width = 10
height = 8
plt.figure(figsize=(width, height))
#plt.bar(x, data_nols, label='nols')
plt.bar(x, data_ls, label='ls')
plt.grid(axis='y')
plt.ylabel('Length ratio')
plt.xlabel('Buckets')
plt.title('Scaled standard error over length buckets (s_100)')
plt.errorbar(x, data_nols, error_nols, marker='o', capsize=4, color='black', ls='none')
plt.errorbar(x, data_ls, error_ls, marker='o', capsize=4, color='black', ls='none')
plt.savefig('buckets.png', dpi=400)

import sys

for s in ['0', '10', '20', '30', '40', '50', '60', '70' ,'80', '90', '100']:
    for var in ['nols', 'ls']:
        f_name = f'/cluster/scratch/ggabriel/baseline_riley/s_{s}/evaluation/sampling/{var}/generate-test.txt'
        f = open(f_name)
        i = 0
        total_sum_sys_bpe = 0
        total_sum_ref_bpe = 0
        total_sum_sys_nobpe = 0
        total_sum_ref_nobpe = 0
        ratio_sum_bpe = 0
        ratio_sum_nobpe = 0
        total_sum = 0
        for line in f:
            if i == 0:
                src = line.split()[1:]
            elif i == 1:
                ref = line.split()[1:]
            elif (i%3)==0:
                total_sum_sys_bpe += len(' '.join(line.split()[2:]))
                total_sum_ref_bpe += len(' '.join(ref))
                total_sum_sys_nobpe += len(''.join(line.split()[2:]))
                total_sum_ref_nobpe += len(''.join(ref))
                ratio_sum_bpe += len(' '.join(line.split()[2:]))/len(' '.join(ref))
                ratio_sum_nobpe += len(''.join(line.split()[2:]))/len(''.join(ref))
                total_sum += 1
            i += 1
            if i >= 3002:
                i = 0
        print(f_name)
        print('avg_sum_divide_bpe:', total_sum_sys_bpe, total_sum_ref_bpe, total_sum_sys_bpe/total_sum_ref_bpe)
        print('avg_sum_divide_nobpe:', total_sum_sys_nobpe, total_sum_ref_nobpe, total_sum_sys_nobpe/total_sum_ref_nobpe)
        print('avg_divide_avg_bpe:', ratio_sum_bpe, total_sum, ratio_sum_bpe/total_sum)
        print('avg_divide_avg_nobpe:', ratio_sum_nobpe, total_sum, ratio_sum_nobpe/total_sum)
        f.close()

import sys

buckets = []
for i in range(10):
    buckets.append([0,0,0,0,0,0,0])
for s in ['100']:
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
                bucket = len(''.join(ref))//100
            elif (i%3)==0:
                buckets[bucket][0] += len(' '.join(line.split()[2:]))
                buckets[bucket][1] += len(' '.join(ref))
                buckets[bucket][2] += len(''.join(line.split()[2:]))
                buckets[bucket][3] += len(''.join(ref))
                buckets[bucket][4] += len(' '.join(line.split()[2:]))/len(' '.join(ref))
                buckets[bucket][5] += len(''.join(line.split()[2:]))/len(''.join(ref))
                buckets[bucket][6] += 1
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
                #print(buckets)
        print(f_name)
        print('avg_sum_divide_bpe:', total_sum_sys_bpe, total_sum_ref_bpe, total_sum_sys_bpe/total_sum_ref_bpe)
        print('avg_sum_divide_nobpe:', total_sum_sys_nobpe, total_sum_ref_nobpe, total_sum_sys_nobpe/total_sum_ref_nobpe)
        print('avg_divide_avg_bpe:', ratio_sum_bpe, total_sum, ratio_sum_bpe/total_sum)
        print('avg_divide_avg_nobpe:', ratio_sum_nobpe, total_sum, ratio_sum_nobpe/total_sum)
        
        for i in range(100):
            print(f'BUCKET {i*100} - {(i+1)*100}:')
            print('avg_sum_divide_bpe:', buckets[i][0], buckets[i][1], buckets[i][0]/buckets[i][1])
            print('avg_sum_divide_nobpe:', buckets[i][2], buckets[i][3], buckets[i][2]/buckets[i][3])
            print('avg_divide_avg_bpe:', buckets[i][4], buckets[i][6], buckets[i][4]/buckets[i][6])
            print('avg_divide_avg_nobpe:', buckets[i][5], buckets[i][6], buckets[i][5]/buckets[i][6])
        f.close()

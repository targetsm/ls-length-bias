import sys

for s in ['0']:#, '10', '20', '30', '40', '50', '60', '70' ,'80', '90', '100']:
    for var in ['ls']:
        f_name = f'/cluster/scratch/ggabriel/baseline_riley/s_{s}/evaluation/sampling/{var}/generate-test.txt'
        f_out_nobpe = open(f'/cluster/scratch/ggabriel/baseline_riley/s_{s}/evaluation/sampling/{var}/generate_test_processed_nobpe.txt', 'w')
        f_out_bpe = open(f'/cluster/scratch/ggabriel/baseline_riley/s_{s}/evaluation/sampling/{var}/generate_test_processed_bpe.txt', 'w')
        f = open(f_name)
        i = 0
        total_sum = 0
        this_line_nobpe = []
        this_line_bpe = []
        for line in f:
            if i == 0 or i == 1:
                this_line_nobpe.append(len(''.join(line.split()[1:])))
                this_line_bpe.append(len(' '.join(line.split()[1:])))
            elif (i%3)==0:
                this_line_nobpe.append(len(''.join(line.split()[2:])))
                this_line_bpe.append(len(' '.join(line.split()[2:])))
            i += 1
            if i >= 3002:
                f_out_nobpe.write(str(this_line_nobpe) + '\n')
                f_out_bpe.write(str(this_line_bpe) + '\n')
                #print(this_line_nobpe, this_line_bpe)
                this_line_nobpe = []
                this_line_bpe = []
                i = 0
        f.close()

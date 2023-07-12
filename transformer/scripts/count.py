#ls_list = ['0', '0.005', '0.01', '0.05', '0.1', '0.5']
ls_list = ['0.5']
#hyp_dict = {'base':[], '4k':[], '64k':[]}
hyp_dict = {'no_pos':[]}

for d in hyp_dict.keys():
    for l in ls_list:
        print(d, l)
        f = open(f'/cluster/scratch/ggabriel/transformer/{d}/ls_{l}/evaluation/sampling/generate-test.txt','r')
        f_out = open(f'/cluster/home/ggabriel/ls-length-bias/transformer/{d}/evaluation/ls_{l}/generate-test-sampling.txt', 'w')
        line_out = []
        i = 0
        for line in f:
            if line[0] == 'S':
                line_out.append(len(line.split()[1:]))
            elif line[0] == 'T':
                line_out.append(len(line.split()[1:]))
            elif line[0] == 'H':
                line_out.append(len(line.split()[2:]))
            if i == 3001:
                f_out.write(str(line_out) + '\n')
                line_out = []
                i = 0
                continue
            i += 1 
            

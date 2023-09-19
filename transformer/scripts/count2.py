#ls_list = ['0', '0.005', '0.01', '0.5']
ls_list = ['0.1', '0.05']
#hyp_dict = {'base':[], '4k':[], '64k':[]}
hyp_dict = {'rel_pos_4k':[]}

for d in hyp_dict.keys():
    for l in ls_list:
        print(d, l)
        f1 = open(f'/cluster/scratch/ggabriel/transformer/{d}/ls_{l}/evaluation/sampling/generate-test.txt','r')
        f2 = open(f'/cluster/scratch/ggabriel/transformer/{d}/ls_{l}/evaluation/sampling2/generate-test.txt','r')
        f_out = open(f'/cluster/home/ggabriel/ls-length-bias/transformer/{d}/evaluation/ls_{l}/generate-test-sampling.txt', 'w')
        line_out = []
        i = 0
        for line in f1:
            #if i==0: 
            #    print(i, line)
            #if i==1:
            #    print(i, line)
            if i == 901:
                for line2 in f2:
                    #if i==1501:
                    #    print(i, line2)
                    if i == 3002:
                        #print(line)
                        #print(line2)
                        #exit()
                        f_out.write(str(line_out) + '\n')
                        line_out = []
                        i = 0
                        break
                    if line2[0] == '2':
                        continue
                    if line2[0] == 'S':
                        source2 = line2
                    elif line2[0] == 'T':
                        target2 = line2
                    elif line2[0] == 'H':
                        line_out.append(len(line2.split()[2:]))
                    i += 1
                #f_out.write(str(line_out) + '\n')
                #line_out = []
                #i = 0
                continue
            if line[0] == '2':
                continue
            if line[0] == 'S':
                line_out.append(len(line.split()[1:]))
            elif line[0] == 'T':
                line_out.append(len(line.split()[1:]))
            elif line[0] == 'H':
                line_out.append(len(line.split()[2:]))
            i += 1 
            

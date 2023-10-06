import sys
import matplotlib.pyplot as plt
import math





f = open(sys.argv[1]).readlines()


total_len = 0
count = 0
for line in f:
    total_len += len(line.split())
    count += 1
mean = total_len/count

variance=0
for line in f:
    variance += (len(line.split())-mean)**2
print(sys.argv[1], total_len, count, total_len/count, math.sqrt(variance/count))


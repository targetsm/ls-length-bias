lines = open('../data/iwslt17.de-en.bpe16k/test.bpe.de-en.en').readlines()

lines.sort(key=lambda x: len(x.split()),reverse=False)
print(lines[:10])
with open('../data/iwslt17.de-en.bpe16k/test_split_short.en', 'w') as f:
    for line in lines[:(len(lines)//2)]:
        f.write(line)

with open('../data/iwslt17.de-en.bpe16k/test_split_long.en', 'w') as f:
    for line in lines[(len(lines)//2):]:
        f.write(line)

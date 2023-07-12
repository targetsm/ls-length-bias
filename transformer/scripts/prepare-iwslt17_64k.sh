#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e

source ../../venv/bin/activate

SRCS=(
    "de"
)
TGT=en

SCRIPTS=../../fairseq/scripts
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py

BPESIZE=65536
ORIG=../data/iwslt17_orig
DATA=../data/iwslt17.de-en.bpe64k
mkdir -p "$ORIG" "$DATA"

TRAIN_MINLEN=0  # remove sentences with <1 BPE token
TRAIN_MAXLEN=250  # remove sentences with >250 BPE tokens

URLS=(
    "https://wit3.fbk.eu/archive/2017-01-trnted/texts/de/en/de-en.tgz"
)
ARCHIVES=(
    "de-en.tgz"
)
VALID_SETS=(
    "IWSLT17.TED.dev2010.de-en"
)
TEST_SETS=(
    "IWSLT17.TED.tst2010.de-en IWSLT17.TED.tst2011.de-en IWSLT17.TED.tst2012.de-en IWSLT17.TED.tst2013.de-en IWSLT17.TED.tst2014.de-en IWSLT17.TED.tst2015.de-en"
)

echo "pre-processing train data..."
for SRC in "${SRCS[@]}"; do
    for LANG in "${SRC}" "${TGT}"; do
        cat "$ORIG/${SRC}-${TGT}/train.tags.${SRC}-${TGT}.${LANG}" \
            | grep -v '<url>' \
            | grep -v '<talkid>' \
            | grep -v '<keywords>' \
            | grep -v '<speaker>' \
            | grep -v '<reviewer' \
            | grep -v '<translator' \
            | grep -v '<doc' \
            | grep -v '</doc>' \
            | sed -e 's/<title>//g' \
            | sed -e 's/<\/title>//g' \
            | sed -e 's/<description>//g' \
            | sed -e 's/<\/description>//g' \
            | sed 's/^\s*//g' \
            | sed 's/\s*$//g' \
            > "$DATA/train.${SRC}-${TGT}.${LANG}"
    done
done

echo "removing partial copies from train data..."
python filter.py $DATA/train.de-en.de $DATA/train.de-en.en

echo "pre-processing valid data..."
for ((i=0;i<${#SRCS[@]};++i)); do
    SRC=${SRCS[i]}
    VALID_SET=(${VALID_SETS[i]})
    for ((j=0;j<${#VALID_SET[@]};++j)); do
        FILE=${VALID_SET[j]}
        for LANG in "$SRC" "$TGT"; do
            grep '<seg id' "$ORIG/${SRC}-${TGT}/${FILE}.${LANG}.xml" \
                | sed -e 's/<seg id="[0-9]*">\s*//g' \
                | sed -e 's/\s*<\/seg>\s*//g' \
                | sed -e "s/\’/\'/g" \
                > "$DATA/valid.${SRC}-${TGT}.${LANG}"
        done
    done
done

rm -f $DATA/test.*
echo "pre-processing test data..."
for ((i=0;i<${#SRCS[@]};++i)); do
    SRC=${SRCS[i]}
    VALID_SET=(${TEST_SETS[i]})
    for ((j=0;j<${#VALID_SET[@]};++j)); do
        FILE=${VALID_SET[j]}
        for LANG in "$SRC" "$TGT"; do
            grep '<seg id' "$ORIG/${SRC}-${TGT}/${FILE}.${LANG}.xml" \
                | sed -e 's/<seg id="[0-9]*">\s*//g' \
                | sed -e 's/\s*<\/seg>\s*//g' \
                | sed -e "s/\’/\'/g" \
                >> "$DATA/test.${SRC}-${TGT}.${LANG}"
        done
    done
done

# learn BPE with sentencepiece
TRAIN_FILES=$(for SRC in "${SRCS[@]}"; do echo $DATA/train.${SRC}-${TGT}.${SRC}; echo $DATA/train.${SRC}-${TGT}.${TGT}; done | tr "\n" ",")
echo "learning joint BPE over ${TRAIN_FILES}..."
python "$SPM_TRAIN" \
    --input=$TRAIN_FILES \
    --model_prefix=$DATA/sentencepiece.bpe \
    --vocab_size=$BPESIZE \
    --character_coverage=1.0 \
    --model_type=bpe

# encode train/valid
echo "encoding train with learned BPE..."
for SRC in "${SRCS[@]}"; do
    python "$SPM_ENCODE" \
        --model "$DATA/sentencepiece.bpe.model" \
        --output_format=piece \
        --inputs $DATA/train.${SRC}-${TGT}.${SRC} $DATA/train.${SRC}-${TGT}.${TGT} \
        --outputs $DATA/train.bpe.${SRC}-${TGT}.${SRC} $DATA/train.bpe.${SRC}-${TGT}.${TGT} \
        --min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN
done

echo "encoding valid with learned BPE..."
SRC=de
python "$SPM_ENCODE" \
    --model "$DATA/sentencepiece.bpe.model" \
    --output_format=piece \
    --inputs $DATA/valid.${SRC}-${TGT}.${SRC} $DATA/valid.${SRC}-${TGT}.${TGT} \
    --outputs $DATA/valid.bpe.${SRC}-${TGT}.${SRC} $DATA/valid.bpe.${SRC}-${TGT}.${TGT}

echo "encoding test with learned BPE..."
SRC=de
python "$SPM_ENCODE" \
    --model "$DATA/sentencepiece.bpe.model" \
    --output_format=piece \
    --inputs $DATA/test.${SRC}-${TGT}.${SRC} $DATA/test.${SRC}-${TGT}.${TGT} \
    --outputs $DATA/test.bpe.${SRC}-${TGT}.${SRC} $DATA/test.bpe.${SRC}-${TGT}.${TGT}


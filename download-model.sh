#!/bin/bash

# Downloads the weights of the pretrained language model. First parameter is the directory name.
# Usage:
# download-model.sh <output-directory>

MODEL_DIR=${1-model} # save to model if no param is set
echo "Downloading to $MODEL_DIR"
mkdir -p $MODEL_DIR
pushd $MODEL_DIR
wget https://raw.githubusercontent.com/openai/finetune-transformer-lm/master/model/encoder_bpe_40000.json
wget https://raw.githubusercontent.com/openai/finetune-transformer-lm/master/model/params_0.npy
wget https://raw.githubusercontent.com/openai/finetune-transformer-lm/master/model/params_1.npy
wget https://raw.githubusercontent.com/openai/finetune-transformer-lm/master/model/params_2.npy
wget https://raw.githubusercontent.com/openai/finetune-transformer-lm/master/model/params_3.npy
wget https://raw.githubusercontent.com/openai/finetune-transformer-lm/master/model/params_4.npy
wget https://raw.githubusercontent.com/openai/finetune-transformer-lm/master/model/params_5.npy
wget https://raw.githubusercontent.com/openai/finetune-transformer-lm/master/model/params_6.npy
wget https://raw.githubusercontent.com/openai/finetune-transformer-lm/master/model/params_7.npy
wget https://raw.githubusercontent.com/openai/finetune-transformer-lm/master/model/params_8.npy
wget https://raw.githubusercontent.com/openai/finetune-transformer-lm/master/model/params_9.npy
wget https://raw.githubusercontent.com/openai/finetune-transformer-lm/master/model/params_shapes.json
wget https://raw.githubusercontent.com/openai/finetune-transformer-lm/master/model/vocab_40000.bpe
popd

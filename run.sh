#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -C TitanX
#SBATCH --gres=gpu:1

module load cuda10.0/toolkit
module load cuDNN/cuda10.0

source /home/mms496/.bashrc

mkdir -p /local/mms496/stylegan

cd /local/mms496/stylegan
cp -R /home/mms496/StyleVAE_Experiments/code/StyleGAN.pytorch .


python -u StyleGAN.pytorch/train.py --start_depth 5 --config /StyleGAN.pytorch/configs/sample_ffhq_128.yaml

wait          # wait until programs are finished

cd /home/mms496/StyleVAE_Experiments/stylegan


echo $$
mkdir o`echo $$`
cd o`echo $$`

cp -R /local/mms496/StyleGAN.pytorch .
rm -rf /local/mms496
#!/bin/bash
#SBATCH --time=14:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -C TitanX
#SBATCH --gres=gpu:1

module load cuda10.0/toolkit
module load cuDNN/cuda10.0

source /home/mms496/.bashrc

cd /home/mms496/StyleVAE_Experiments/stylegan

if [ -d "/home/mms496/StyleVAE_Experiments/stylegan/output_siglaplace_80" ] 
then
    echo $$
	mkdir oo`echo $$`
	cd oo`echo $$` 
	cp -R /home/mms496/StyleVAE_Experiments/stylegan/output_siglaplace_80 .
	rm -rf /home/mms496/StyleVAE_Experiments/stylegan/output_siglaplace_80
	cd /home/mms496/StyleVAE_Experiments/stylegan

else 
	echo 'no problem'
fi




python -u /home/mms496/StyleVAE_Experiments/code/StyleGAN.pytorch/train.py --start_depth 5 --config /home/mms496/StyleVAE_Experiments/code/StyleGAN.pytorch/configs/ffhq_128_siglaplace_80.yaml

wait          # wait until programs are finished

echo $$
mkdir o`echo $$`
cd o`echo $$`

cp -R /home/mms496/StyleVAE_Experiments/stylegan/output_siglaplace_80 .
rm -rf /home/mms496/StyleVAE_Experiments/stylegan/output_siglaplace_80

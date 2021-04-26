#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -C TitanX
#SBATCH --gres=gpu:1

module load cuda10.0/toolkit
module load cuDNN/cuda10.0

source /home/mms496/.bashrc

# rm -rf /local/mms496
# mkdir -p /local/mms496/stylegan

# cd /local/mms496/stylegan
# pwd
# ls -a
# cp -R /home/mms496/StyleVAE_Experiments/code/StyleGAN.pytorch .
# echo 'copied'
# pwd
# ls -a

if [ -d "/home/mms496/StyleVAE_Experiments/stylegan/output" ] 
then
    echo $$
	mkdir oo`echo $$`
	cd oo`echo $$` 
	cp -R /home/mms496/StyleVAE_Experiments/stylegan/output .
	rm -rf /home/mms496/StyleVAE_Experiments/stylegan/output
	cd ..




python -u code/StyleGAN.pytorch/train.py --start_depth 5 --config code/StyleGAN.pytorch/configs/sample_ffhq_128.yaml

wait          # wait until programs are finished

echo $$
mkdir o`echo $$`
cd o`echo $$`

cp -R /home/mms496/StyleVAE_Experiments/stylegan/output .
rm -rf /home/mms496/StyleVAE_Experiments/stylegan/output


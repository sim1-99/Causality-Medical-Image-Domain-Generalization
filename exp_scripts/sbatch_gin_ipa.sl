#!/bin/bash
#SBATCH -J gin_ipa
#SBATCH -p medium
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -c 16
#SBATCH --chdir=/home/schiarella
#SBATCH -o /home/schiarella/Causality-Medical-Image-Domain-Generalization/outs/%x-%j.out 
#SBATCH -e /home/schiarella/Causality-Medical-Image-Domain-Generalization/outs/%x-%j.err  

source ~/anaconda3/bin/activate "";
conda activate ginipa3;

export CUDA_VISIBLE_DEVICES=0 

cd /home/schiarella/Causality-Medical-Image-Domain-Generalization #/exp_scripts
python ginipa.py #bash feta_gin_ipa.sh

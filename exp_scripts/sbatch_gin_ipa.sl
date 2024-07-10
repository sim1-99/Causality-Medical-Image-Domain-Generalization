#!/bin/bash
#SBATCH -J gin_ipa_prova
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -c 32
#SBATCH --chdir=/home/schiarella
#SBATCH -o /home/schiarella/Causality-Medical-Image-Domain-Generalization/outs/%x-%j.out 
#SBATCH -e /home/schiarella/Causality-Medical-Image-Domain-Generalization/outs/%x-%j.err  

source ~/anaconda3/bin/activate "";
conda activate ginipa3;

export CUDA_VISIBLE_DEVICES=0 

cd /home/schiarella/Causality-Medical-Image-Domain-Generalization/exp_scripts
bash feta_gin_ipa.sh

#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem 36GB
#SBATCH --cpus-per-task 8
#SBATCH -t 7-00:00:00
#SBATCH --job-name pdunet
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

echo "start"
echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
echo
nvidia-smi
. /geoinfo_vol1/puzhao/miniforge3/etc/profile.d/conda.sh

# module --ignore-cache load "intel"
# PROJECT_DIR=/cephyr/NOBACKUP/groups/snic2021-7-104/puzhao-snic-500G/smp-seg-pytorch

# # Choose different config files
# # DIRS=($(find /cephyr/NOBACKUP/groups/snic2021-7-104/puzhao-snic-500G/smp-seg-pytorch/config/s1s2_cfg/))
# DIRS=($(find $PROJECT_DIR/config/s1s2_unet/))
# DIRS=${DIRS[@]:1}
# CFG=${DIRS[$SLURM_ARRAY_TASK_ID]}
# echo "Running simulation $CFG"

# # Choose different sensors
# SAT=('S1' 'S2' 'ALOS')

SAT=('alos' 's1' 's2')
CFG=${SAT[$SLURM_ARRAY_TASK_ID]}
echo "Running simulation $CFG"
echo "python3 main_s1s2_unet.py s1s2_unet=$CFG model.batch_size=32"
echo "---------------------------------------------------------------------------------------------------------------"

# conda activate pytorch
# PYTHONUNBUFFERED=1; python3 /home/p/u/puzhao/smp-seg-pytorch/main_s1s2_siamunet.py model.ARCH=$CFG model.max_epoch=20

conda activate paddle2
PYTHONUNBUFFERED=1; \
python3 train.py \
        --config configs/unet/wildfire_unet_s1s2.yml \
        --do_eval \
        --use_vdl \
        --save_interval 500 \
        --save_dir output
        # --config=./configs/quick_start/bisenet_optic_disc_512x512_1k.yml\
# PYTHONUNBUFFERED=1; python3 /home/p/u/puzhao/smp-seg-pytorch/main_s1s2_siamunet.py
# PYTHONUNBUFFERED=1; python3 main_s1s2_unet.py s1s2_unet=$CFG model.batch_size=32 model.ARCH=Paddle_unet
# PYTHONUNBUFFERED=1; python3 main_s1s2_unet.py model.batch_size=32 model.ARCH=Paddle_unet data.satellites=['S1','S2']

# PYTHONUNBUFFERED=1; python3 main_s1s2_unet.py model.batch_size=8 model.ARCH=UNet

#rm -rf $SLURM_SUBMIT_DIR/*.log
# rm -rf $SLURM_SUBMIT_DIR/*.out

echo "finish"

## run
# sbatch --array=1-2 geo_run_s1s2_unet.sh

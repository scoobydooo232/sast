source miniconda3/etc/profile.d/conda.sh
conda activate rvt
# cd sast/scripts/genx/
cd sast/
DATA_DIR=/workspace/dataset CKPT_PATH=/workspace/sast_step.ckpt USE_TEST=0 GPU_ID=0 DEST_DIR=/workspace/final_dataset_1
NUM_PROCESSES=5 
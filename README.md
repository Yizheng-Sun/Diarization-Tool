srun --partition=gpuL --gpus=1 --ntasks=1 --time=1-0 --pty bash
bash setup.sh
bash scripts/run_titanet_tracklet_batch.sh
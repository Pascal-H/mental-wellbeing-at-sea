#!/bin/bash

# Read the list of directories from directories.txt
mapfile -t dir_list < directories.txt

NUM_GPUS=8
starting_gpu=0  # <-- Set your starting GPU index here

# Array to track PIDs for each GPU
declare -a gpu_pids
# Array to track which directory is being processed on each GPU (for logging)
declare -a gpu_dirs

# Function to wait for any GPU to become free, returns GPU ID
wait_for_free_gpu() {
    while true; do
        for ((i=0; i<$NUM_GPUS; i++)); do
           gpu=$((starting_gpu + i))
            pid=${gpu_pids[$gpu]}
            # If no process or process finished
            if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
                echo $gpu
                return
            fi
        done
        sleep 1
    done
}

for dir_in in "${dir_list[@]}"; do
    gpu_id=$(wait_for_free_gpu)

    # Optionally, set your cache and output paths here
    cache_path="/scratch/phecker/project/audiary/projects/2021-safetytech_accelerator-mwas/passive-voyage_data_recorder/data/cache_analysis"
    output_path="/scratch/phecker/project/audiary/projects/2021-safetytech_accelerator-mwas/passive-voyage_data_recorder/data/output"

    # Start the process on the selected GPU
    python3 predict_main.py \
        "$dir_in" \
        "$cache_path" \
        "$output_path" \
        --lst_filter_mics "M1,M2,M3,V4,V5,M6" \
        "no_denoising" \
        "auvad" \
        "497f50d6-1.1.0" \
        "false" \
        "$gpu_id" &
        # 497f50d6-1.1.0 <- mobilenet
        # false/true <-- Set flag_transcribe

    pid=$!
    gpu_pids[$gpu_id]=$pid
    gpu_dirs[$gpu_id]=$dir_in

    echo "Started processing $dir_in on GPU $gpu_id (PID $pid)"
done

# Wait for all background jobs to finish
wait

echo "Batch processing completed."
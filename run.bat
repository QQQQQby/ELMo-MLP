@echo off

set BATCH_SIZE=16
set LEARNING_RATE=1e-2
set LAMB=0.0001
set NUM_EPOCHS=20
set DATASET_PATH=./data

set OUTPUT_DIR=./results/%BATCH_SIZE%_%LEARNING_RATE%_%LAMB%

python run_classifier.py ^
    --batch_size=%BATCH_SIZE% ^
    --learning_rate=%LEARNING_RATE% ^
    --num_epochs=%NUM_EPOCHS% ^
    --lamb=%LAMB% ^
    --dataset_path=%DATASET_PATH% ^
    --output_path=%OUTPUT_DIR%

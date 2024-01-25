# Train BART baseline
CUDA_VISIBLE_DEVICES=1 nohup python train.py \
    --model_path gogamza/kobart-base-v1 \
    --run_name kobart_baseline \
    --batch_size 64 \
    --path_to_train_data ./data/train.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/kobart_baseline.log &   

# Train BART unsmile
CUDA_VISIBLE_DEVICES=1 nohup python train.py \
    --model_path gogamza/kobart-base-v1 \
    --run_name kobart_unsmile_unfiltered \
    --batch_size 64 \
    --path_to_train_data ./data/integrated/unsmile_integrated_unfiltered.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/kobart_unsmile_unfiltered.log &   

# Train BART apeach
CUDA_VISIBLE_DEVICES=2 nohup python train.py \
    --model_path gogamza/kobart-base-v1 \
    --run_name kobart_apeach_unfiltered \
    --batch_size 64 \
    --path_to_train_data ./data/integrated/apeach_integrated_unfiltered.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/kobart_apeach_unfiltered.log &   

# Train BART beep
CUDA_VISIBLE_DEVICES=2 nohup python train.py \
    --model_path gogamza/kobart-base-v1 \
    --run_name kobart_beep_unfiltered \
    --batch_size 64 \
    --path_to_train_data ./data/integrated/beep_integrated_unfiltered.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/kobart_beep_unfiltered.log &   

# Train BART kold
CUDA_VISIBLE_DEVICES=3 nohup python train.py \
    --model_path gogamza/kobart-base-v1 \
    --run_name kobart_kold_unfiltered \
    --batch_size 64 \
    --path_to_train_data ./data/integrated/kold_integrated_unfiltered.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/kobart_kold_unfiltered.log &   

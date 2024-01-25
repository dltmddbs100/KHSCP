# Train T5 baseline
CUDA_VISIBLE_DEVICES=1 nohup python train.py \
    --model_path google/mt5-base \
    --run_name mt5_baseline \
    --batch_size 64 --learning_rate 5e-4 --max_epochs 10 \
    --path_to_train_data ./data/train.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/mt5_baseline.log &   

# Train T5 unsmile
CUDA_VISIBLE_DEVICES=6 nohup python train.py \
    --model_path google/mt5-base \
    --run_name mt5_unsmile_unfiltered \
    --batch_size 64 --learning_rate 5e-4 --max_epochs 10 \
    --path_to_train_data ./data/integrated/unsmile_integrated_unfiltered.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/mt5_unsmile_unfiltered.log &   

# Train T5 apeach
CUDA_VISIBLE_DEVICES=7 nohup python train.py \
    --model_path google/mt5-base \
    --run_name mt5_apeach_unfiltered \
    --batch_size 64 --learning_rate 5e-4 --max_epochs 10 \
    --path_to_train_data ./data/integrated/apeach_integrated_unfiltered.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/mt5_apeach_unfiltered.log &   

# Train T5 beep
CUDA_VISIBLE_DEVICES=3 nohup python train.py \
    --model_path google/mt5-base \
    --run_name mt5_beep_unfiltered \
    --batch_size 64 --learning_rate 5e-4 --max_epochs 10 \
    --path_to_train_data ./data/integrated/beep_integrated_unfiltered.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/mt5_beep_unfiltered.log &   

# Train T5 kold
CUDA_VISIBLE_DEVICES=4 nohup python train.py \
    --model_path google/mt5-base \
    --run_name mt5_kold_unfiltered \
    --batch_size 64 --learning_rate 5e-4 --max_epochs 10 \
    --path_to_train_data ./data/integrated/kold_integrated_unfiltered.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/mt5_kold_unfiltered.log &   

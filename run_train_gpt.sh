# Train GPT2 baseline
CUDA_VISIBLE_DEVICES=1 nohup python train.py \
    --model_path skt/kogpt2-base-v2 \
    --run_name kogpt2_baseline \
    --batch_size 64 \
    --path_to_train_data ./data/train.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/kogpt2_baseline.log &   

# Train GPT2 unsmile
CUDA_VISIBLE_DEVICES=1 nohup python train.py \
    --model_path skt/kogpt2-base-v2 \
    --run_name kogpt2_unsmile_unfiltered \
    --batch_size 64 \
    --path_to_train_data ./data/integrated/unsmile_integrated_unfiltered.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/kogpt2_unsmile_unfiltered.log &   

# Train GPT2 apeach
CUDA_VISIBLE_DEVICES=2 nohup python train.py \
    --model_path skt/kogpt2-base-v2 \
    --run_name kogpt2_apeach_unfiltered \
    --batch_size 64 \
    --path_to_train_data ./data/integrated/apeach_integrated_unfiltered.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/kogpt2_apeach_unfiltered.log &   

# Train GPT2 beep
CUDA_VISIBLE_DEVICES=2 nohup python train.py \
    --model_path skt/kogpt2-base-v2 \
    --run_name kogpt2_beep_unfiltered \
    --batch_size 64 \
    --path_to_train_data ./data/integrated/beep_integrated_unfiltered.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/kogpt2_beep_unfiltered.log &   

# Train GPT2 kold
CUDA_VISIBLE_DEVICES=3 nohup python train.py \
    --model_path skt/kogpt2-base-v2 \
    --run_name kogpt2_kold_unfiltered \
    --batch_size 64 \
    --path_to_train_data ./data/integrated/kold_integrated_unfiltered.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/kogpt2_kold_unfiltered.log &   

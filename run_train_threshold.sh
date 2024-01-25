# Train T5 unsmile
CUDA_VISIBLE_DEVICES=4 nohup python train.py \
    --model_path google/mt5-base \
    --run_name mt5_unsmile_0_integrated \
    --batch_size 64 --learning_rate 5e-4 --max_epochs 10 \
    --path_to_train_data ./data/threshold_integrated/unsmile_0_integrated.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/threshold/mt5_unsmile_0_integrated.log &   

# Train T5 unsmile
CUDA_VISIBLE_DEVICES=6 nohup python train.py \
    --model_path google/mt5-base \
    --run_name mt5_unsmile_0.4_integrated \
    --batch_size 64 --learning_rate 5e-4 --max_epochs 10 \
    --path_to_train_data ./data/threshold_integrated/unsmile_0.4_integrated.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/threshold/mt5_unsmile_0.4_integrated.log &   

# Train T5 unsmile
CUDA_VISIBLE_DEVICES=7 nohup python train.py \
    --model_path google/mt5-base \
    --run_name mt5_unsmile_0.5_integrated \
    --batch_size 64 --learning_rate 5e-4 --max_epochs 10 \
    --path_to_train_data ./data/threshold_integrated/unsmile_0.5_integrated.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/threshold/mt5_unsmile_0.5_integrated.log &   

# Train T5 unsmile
CUDA_VISIBLE_DEVICES=5 nohup python train.py \
    --model_path google/mt5-base \
    --run_name mt5_unsmile_0.7_integrated \
    --batch_size 64 --learning_rate 5e-4 --max_epochs 10 \
    --path_to_train_data ./data/threshold_integrated/unsmile_0.7_integrated.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/threshold/mt5_unsmile_0.7_integrated.log &   

# Train T5 unsmile
CUDA_VISIBLE_DEVICES=4 nohup python train.py \
    --model_path google/mt5-base \
    --run_name mt5_unsmile_0.8_integrated \
    --batch_size 64 --learning_rate 5e-4 --max_epochs 10 \
    --path_to_train_data ./data/threshold_integrated/unsmile_0.8_integrated.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/threshold/mt5_unsmile_0.8_integrated.log &   

#####################################################################

# Train Kobart unsmile
CUDA_VISIBLE_DEVICES=4 nohup python train.py \
    --model_path gogamza/kobart-base-v1 \
    --run_name kobart_unsmile_0_integrated \
    --batch_size 64 \
    --path_to_train_data ./data/threshold_integrated/unsmile_0_integrated.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/threshold/kobart_unsmile_0_integrated.log &   

# Train Kobart unsmile
CUDA_VISIBLE_DEVICES=5 nohup python train.py \
    --model_path gogamza/kobart-base-v1 \
    --run_name kobart_unsmile_0.4_integrated \
    --batch_size 64 \
    --path_to_train_data ./data/threshold_integrated/unsmile_0.4_integrated.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/threshold/kobart_unsmile_0.4_integrated.log &   

# Train Kobart unsmile
CUDA_VISIBLE_DEVICES=6 nohup python train.py \
    --model_path gogamza/kobart-base-v1 \
    --run_name kobart_unsmile_0.5_integrated \
    --batch_size 64 \
    --path_to_train_data ./data/threshold_integrated/unsmile_0.5_integrated.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/threshold/kobart_unsmile_0.5_integrated.log &   

# Train Kobart unsmile
CUDA_VISIBLE_DEVICES=7 nohup python train.py \
    --model_path gogamza/kobart-base-v1 \
    --run_name kobart_unsmile_0.7_integrated \
    --batch_size 64 \
    --path_to_train_data ./data/threshold_integrated/unsmile_0.7_integrated.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/threshold/kobart_unsmile_0.7_integrated.log &   

# Train Kobart unsmile
CUDA_VISIBLE_DEVICES=5 nohup python train.py \
    --model_path gogamza/kobart-base-v1 \
    --run_name kobart_unsmile_0.8_integrated \
    --batch_size 64 \
    --path_to_train_data ./data/threshold_integrated/unsmile_0.8_integrated.csv \
    --path_to_valid_data ./data/valid.csv \
    --path_to_test_data ./data/test.csv > ./logs/threshold/kobart_unsmile_0.8_integrated.log &   
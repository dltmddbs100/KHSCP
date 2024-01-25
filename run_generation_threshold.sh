# mt5_unsmile_0
CUDA_VISIBLE_DEVICES=4 nohup python generate.py \
    --run_name mt5_unsmile_0_integrated \
    --model_path google/mt5-base \
    --test_model_path ./runs/mt5_unsmile_0_integrated/ > nohup.out &   

# mt5_unsmile_0.4
CUDA_VISIBLE_DEVICES=5 nohup python generate.py \
    --run_name mt5_unsmile_0.4_integrated \
    --model_path google/mt5-base \
    --test_model_path ./runs/mt5_unsmile_0.4_integrated/ > nohup.out &   

# mt5_unsmile_0.5
CUDA_VISIBLE_DEVICES=5 nohup python generate.py \
    --run_name mt5_unsmile_0.5_integrated \
    --model_path google/mt5-base \
    --test_model_path ./runs/mt5_unsmile_0.5_integrated/ > nohup.out &   

# mt5_unsmile_0.7
CUDA_VISIBLE_DEVICES=7 nohup python generate.py \
    --run_name mt5_unsmile_0.7_integrated \
    --model_path google/mt5-base \
    --test_model_path ./runs/mt5_unsmile_0.7_integrated/ > nohup.out &   

# mt5_unsmile_0.8
CUDA_VISIBLE_DEVICES=6 nohup python generate.py \
    --run_name mt5_unsmile_0.8_integrated \
    --model_path google/mt5-base \
    --test_model_path ./runs/mt5_unsmile_0.8_integrated/ > nohup.out &   

#####################################################################

# kobart_unsmile_0
CUDA_VISIBLE_DEVICES=4 nohup python generate.py \
    --run_name kobart_unsmile_0_integrated \
    --model_path gogamza/kobart-base-v1 \
    --test_model_path ./runs/kobart_unsmile_0_integrated/ > nohup.out &   

# kobart_unsmile_0.4
CUDA_VISIBLE_DEVICES=5 nohup python generate.py \
    --run_name kobart_unsmile_0.4_integrated \
    --model_path gogamza/kobart-base-v1 \
    --test_model_path ./runs/kobart_unsmile_0.4_integrated/ > nohup.out &   

# kobart_unsmile_0.5
CUDA_VISIBLE_DEVICES=6 nohup python generate.py \
    --run_name kobart_unsmile_0.5_integrated \
    --model_path gogamza/kobart-base-v1 \
    --test_model_path ./runs/kobart_unsmile_0.5_integrated/ > nohup.out &   

# kobart_unsmile_0.7
CUDA_VISIBLE_DEVICES=7 nohup python generate.py \
    --run_name kobart_unsmile_0.7_integrated \
    --model_path gogamza/kobart-base-v1 \
    --test_model_path ./runs/kobart_unsmile_0.7_integrated/ > nohup.out &   

# kobart_unsmile_0.8
CUDA_VISIBLE_DEVICES=7 nohup python generate.py \
    --run_name kobart_unsmile_0.8_integrated \
    --model_path gogamza/kobart-base-v1 \
    --test_model_path ./runs/kobart_unsmile_0.8_integrated/ > nohup.out &   
# mt5_baseline
CUDA_VISIBLE_DEVICES=1 nohup python generate.py \
    --run_name mt5_baseline \
    --model_path google/mt5-base \
    --test_model_path ./runs/mt5_baseline/ > nohup.out &  

# mt5_unsmile
CUDA_VISIBLE_DEVICES=5 nohup python generate.py \
    --run_name mt5_unsmile_unfiltered \
    --model_path google/mt5-base \
    --test_model_path ./runs/mt5_unsmile_unfiltered/ > nohup.out &   

# mt5_apeach
CUDA_VISIBLE_DEVICES=5 nohup python generate.py \
    --run_name mt5_apeach_unfiltered \
    --model_path google/mt5-base \
    --test_model_path ./runs/mt5_apeach_unfiltered/ > nohup.out &   

# mt5_beep
CUDA_VISIBLE_DEVICES=4 nohup python generate.py \
    --run_name mt5_beep_unfiltered \
    --model_path google/mt5-base \
    --test_model_path ./runs/mt5_beep_unfiltered/ > nohup.out &   

# mt5_kold
CUDA_VISIBLE_DEVICES=5 nohup python generate.py \
    --run_name mt5_kold_unfiltered \
    --model_path google/mt5-base \
    --test_model_path ./runs/mt5_kold_unfiltered/ > nohup.out &   

#######################################################################


# mt5_gpt3
CUDA_VISIBLE_DEVICES=5 nohup python generate.py \
    --run_name mt5_unsmile_prompt \
    --model_path google/mt5-base \
    --test_model_path ./runs/mt5_unsmile_prompt/ > nohup.out &   

# mt5_gpt3_only
CUDA_VISIBLE_DEVICES=6 nohup python generate.py \
    --run_name mt5_unsmile_prompt_only \
    --model_path google/mt5-base \
    --test_model_path ./runs/mt5_unsmile_prompt_only/ > nohup.out &   
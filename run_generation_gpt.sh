# kogpt2_baseline
CUDA_VISIBLE_DEVICES=1 nohup python generate.py \
    --run_name kogpt2_baseline \
    --model_path skt/kogpt2-base-v2 \
    --test_model_path ./runs/kogpt2_baseline/ > nohup.out &  

# kogpt2_unsmile
CUDA_VISIBLE_DEVICES=1 nohup python generate.py \
    --run_name kogpt2_unsmile_unfiltered \
    --model_path skt/kogpt2-base-v2 \
    --test_model_path ./runs/kogpt2_unsmile_unfiltered/ > nohup.out &   

# kogpt2_apeach
CUDA_VISIBLE_DEVICES=2 nohup python generate.py \
    --run_name kogpt2_apeach_unfiltered \
    --model_path skt/kogpt2-base-v2 \
    --test_model_path ./runs/kogpt2_apeach_unfiltered/ > nohup.out &   

# kogpt2_beep
CUDA_VISIBLE_DEVICES=3 nohup python generate.py \
    --run_name kogpt2_beep_unfiltered \
    --model_path skt/kogpt2-base-v2 \
    --test_model_path ./runs/kogpt2_beep_unfiltered/ > nohup.out &   

# kogpt2_kold
CUDA_VISIBLE_DEVICES=2 nohup python generate.py \
    --run_name kogpt2_kold_unfiltered \
    --model_path skt/kogpt2-base-v2 \
    --test_model_path ./runs/kogpt2_kold_unfiltered/ > nohup.out &   
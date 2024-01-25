# kobart_baseline
CUDA_VISIBLE_DEVICES=1 nohup python generate.py \
    --run_name kobart_baseline \
    --model_path gogamza/kobart-base-v1 \
    --test_model_path ./runs/kobart_baseline/ > nohup.out &  

# kobart_unsmile
CUDA_VISIBLE_DEVICES=1 nohup python generate.py \
    --run_name kobart_unsmile_unfiltered \
    --model_path gogamza/kobart-base-v1 \
    --test_model_path ./runs/kobart_unsmile_unfiltered/ > nohup.out &   

# kobart_apeach
CUDA_VISIBLE_DEVICES=2 nohup python generate.py \
    --run_name kobart_apeach_unfiltered \
    --model_path gogamza/kobart-base-v1 \
    --test_model_path ./runs/kobart_apeach_unfiltered/ > nohup.out &   

# kobart_beep
CUDA_VISIBLE_DEVICES=3 nohup python generate.py \
    --run_name kobart_beep_unfiltered \
    --model_path gogamza/kobart-base-v1 \
    --test_model_path ./runs/kobart_beep_unfiltered/ > nohup.out &   

# kobart_kold
CUDA_VISIBLE_DEVICES=4 nohup python generate.py \
    --run_name kobart_kold_unfiltered \
    --model_path gogamza/kobart-base-v1 \
    --test_model_path ./runs/kobart_kold_unfiltered/ > nohup.out &   
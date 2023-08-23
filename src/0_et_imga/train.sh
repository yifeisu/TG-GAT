ngpus=1
seed=0

flag="--root_dir ../../datasets

      --world_size ${ngpus}
      --seed ${seed}

      --feedback student

      --max_action_len 10
      --max_instr_len 100

      --lr 1e-5
      --iters 200000
      --log_every 1
      --batch_size 4
      --optim adamW

      --ml_weight 0.2      

      --feat_dropout 0.4
      --dropout 0.5
      
      --nss_w 0.1
      --nss_r 0

      --darknet_model_file ../../datasets/AVDN/pretrain_weights/yolo_v3.cfg
      --darknet_weight_file ../../datasets/AVDN/pretrain_weights/best.pt
      "

# train
CUDA_VISIBLE_DEVICES='2' python main.py --output_dir ./logs/ $flag

# eval
CUDA_VISIBLE_DEVICES='7' python main.py --output_dir ./logs/ $flag \
  --resume_file ./logs/ckpts/best_val_unseen \
  --inference True \
  --submit True

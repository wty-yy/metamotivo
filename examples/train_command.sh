# 避免python报错OSError: [Errno 24] Too many open files
ulimit -n 50000

# python fbcpr_train_humenv.py --compile --motions test_train_split/large1_small1_train_0_5track.1.txt \
python fbcpr_train_humenv.py --compile --motions test_train_split/large1_small1_train_0.1.txt \
  --motions_root /data/user/wutianyang/dataset/humenv_amass --prioritization --device cuda:1 \
  --use-wandb --tracking_eval-num-envs 10 --buffer_device cpu

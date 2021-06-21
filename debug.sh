# example training script, put argparse arguments here

checkpoint_dir=checkpoints/debug/

rm -r ${checkpoint_dir} # remove if you don't want to wipe the folder
python3 train.py \
--checkpoint_dir ${checkpoint_dir} \
--mode train \
--init gaussian_circles1.5,0.75 \
--use_net \
--batch_size 2 \

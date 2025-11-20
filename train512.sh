
# download the pretrained croco v2 checkpoint
#mkdir -p checkpoints/
#wget https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTLarge_BaseDecoder.pth -P checkpoints/
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 train.py \
    --train_dataset "1000 @ TartanAir(split='train', ROOT='/run/determined/workdir/data/feed_forward_event/Tartanair_tmp/indoor', aug_crop=16, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)" \
    --test_dataset "100 @ TartanAir(split='test', ROOT='/run/determined/workdir/data/feed_forward_event/Tartanair_tmp/indoor', resolution=(512,384), seed=777)" \
    --model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion "Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --pretrained "checkpoints/dust3r_fintune_512dpt_10epoch/checkpoint-last.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 1 --epochs 200 --batch_size 1 --accum_iter 16 \
    --save_freq 5 --keep_freq 10 --eval_freq 5 --disable_cudnn_benchmark \  
    --output_dir "checkpoints/dust3r_fintune_512dpt_1119"
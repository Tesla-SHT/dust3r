
# download the pretrained croco v2 checkpoint
#mkdir -p checkpoints/
#wget https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTLarge_BaseDecoder.pth -P checkpoints/
export OMP_NUM_THREADS=1
# the training of dust3r is done in 3 steps.
# for this example we'll do fewer epochs, for the actual hyperparameters we used in the paper, see the next section: "Our Hyperparameters"
# step 1 - train dust3r for 224 resolution
torchrun --nproc_per_node=4 train.py \
    --train_dataset "1000 @ Co3d(split='train', ROOT='/run/determined/workdir/data/feed_forward_event/dust3r/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter)" \
    --test_dataset "100 @ Co3d(split='test', ROOT='/run/determined/workdir/data/feed_forward_event/dust3r/co3d_subset_processed', resolution=224, seed=777)" \
    --model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion "Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --pretrained "checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 1 --epochs 200 --batch_size 8 --accum_iter 1 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 \
    --output_dir "checkpoints/dust3r_demo_224"	 
#Remember to change the --output-dir!!!!!!!!
# step 2 - train dust3r for 512 resolution
torchrun --nproc_per_node=4 train.py \
    --train_dataset "1000 @ Co3d(split='train', ROOT='/run/determined/workdir/data/feed_forward_event/dust3r/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)" \
    --test_dataset "100 @ Co3d(split='test', ROOT='/run/determined/workdir/data/feed_forward_event/dust3r/co3d_subset_processed', resolution=(512,384), seed=777)" \
    --model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion "Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --pretrained "checkpoints/dust3r_demo_224/checkpoint-best.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 1 --epochs 10 --batch_size 4 --accum_iter 4 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 \
    --output_dir "checkpoints/dust3r_demo_512"

# step 3 - train dust3r for 512 resolution with dpt
torchrun --nproc_per_node=4 train.py \
    --train_dataset "1000 @ Co3d(split='train', ROOT='/run/determined/workdir/data/feed_forward_event/dust3r/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)" \
    --test_dataset "100 @ Co3d(split='test', ROOT='/run/determined/workdir/data/feed_forward_event/dust3r/co3d_subset_processed', resolution=(512,384), seed=777)" \
    --model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion "Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
    --pretrained "checkpoints/dust3r_demo_512/checkpoint-best.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 1 --epochs 10 --batch_size 2 --accum_iter 8 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 --disable_cudnn_benchmark \
    --output_dir "checkpoints/dust3r_demo_512dpt"

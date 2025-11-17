import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

def get_image_pairs_sliding_window(folder_path, start_idx, end_idx, gap, step):
    """
    滑动窗口方式提取图片对
    :param folder_path: 图片文件夹路径
    :param start_idx: 起始索引
    :param end_idx: 结束索引
    :param gap: 两张图片之间的间隔
    :param step: 每次滑动的步长
    :return: 图片对列表 [(img1_path, img2_path, idx1, idx2), ...]
    """
    folder = Path(folder_path)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    all_images = sorted([
        str(f) for f in folder.iterdir() 
        if f.suffix.lower() in image_extensions
    ])
    
    if end_idx is None:
        end_idx = len(all_images) - 1
    
    image_pairs = []
    current_idx = start_idx
    
    while current_idx + gap <= end_idx and current_idx < len(all_images):
        idx1 = current_idx
        idx2 = current_idx + gap
        
        if idx2 < len(all_images):
            image_pairs.append((all_images[idx1], all_images[idx2], idx1, idx2))
            print(f"添加图片对: [{idx1}] 和 [{idx2}]")
        
        current_idx += step
    
    return image_pairs

def save_scene_screenshot(scene, scene_name, idx1, idx2, output_dir='output'):
    """
    保存场景的截图
    """
    os.makedirs(output_dir, exist_ok=True)
    scene_dir = os.path.join(output_dir, scene_name)
    os.makedirs(scene_dir, exist_ok=True)
    
    filename = f"{scene_name}_pair_{idx1:04d}_{idx2:04d}.png"
    filepath = os.path.join(scene_dir, filename)
    
    try:
        fig = plt.gcf()
        if fig is not None:
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"截图已保存到: {filepath}")
            plt.close(fig)
        else:
            print("无法获取当前图形，跳过截图")
    except Exception as e:
        print(f"保存截图时出错: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DUSt3R Sliding Window Batch Processing')
    parser.add_argument('--folder', type=str, required=True,
                        help='图片文件夹路径')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='起始索引')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='结束索引（None表示到最后）')
    parser.add_argument('--gap', type=int, default=5,
                        help='两张图片之间的间隔帧数')
    parser.add_argument('--step', type=int, default=5,
                        help='滑动窗口的步长')
    parser.add_argument('--scene_name', type=str, default='scene',
                        help='场景名称')
    parser.add_argument('--save_screenshot', action='store_true',
                        help='是否保存场景截图')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='输出文件夹路径')
    parser.add_argument('--model_path', type=str, 
                        default='checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth',
                        help='模型路径')
    
    args = parser.parse_args()
    
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    # 加载模型
    print(f"加载模型: {args.model_path}")
    model = AsymmetricCroCo3DStereo.from_pretrained(args.model_path).to(device)
    
    # 获取所有图片对
    image_pairs = get_image_pairs_sliding_window(
        args.folder, args.start_idx, args.end_idx, args.gap, args.step
    )
    
    print(f"\n共生成 {len(image_pairs)} 个图片对")
    
    # 处理每一对图片
    for pair_idx, (img1_path, img2_path, idx1, idx2) in enumerate(image_pairs):
        print(f"\n{'='*60}")
        print(f"处理第 {pair_idx+1}/{len(image_pairs)} 对: [{idx1}] - [{idx2}]")
        print(f"图片1: {img1_path}")
        print(f"图片2: {img2_path}")
        
        try:
            # 加载图片
            images = load_images([img1_path, img2_path], size=224)
            pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
            output = inference(pairs, model, device, batch_size=batch_size)

            # 全局对齐
            scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
            loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

            # 获取结果
            imgs = scene.imgs
            focals = scene.get_focals()
            poses = scene.get_im_poses()
            pts3d = scene.get_pts3d()
            confidence_masks = scene.get_masks()

            # 可视化（如果需要保存截图）
            if args.save_screenshot:
                scene.show()
                save_scene_screenshot(scene, args.scene_name, idx1, idx2, args.output_dir)
            
            print(f"✓ 处理完成")
            
        except Exception as e:
            print(f"✗ 处理失败: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("批量处理完成！")

'''
使用示例：
python usage_sliding_window.py \
    --folder /path/to/images \
    --start_idx 0 \
    --end_idx 100 \
    --gap 5 \
    --step 5 \
    --scene_name carwelding \
    --save_screenshot \
    --output_dir output
    
这将生成图片对：
[0,5], [5,10], [10,15], [15,20], ...
'''
'''
# 模式1: 两张图片
python usage_batch.py --mode two_images --image1 path/to/img1.png --image2 path/to/img2.png --save_screenshot --scene_name chateau

# 模式2: 文件夹批量
python usage_batch.py --mode folder --folder path/to/images --start_idx 0 --gap 5 --step 12 --save_screenshot --scene_name mydata
python usage_batch.py --mode folder --folder /run/determined/workdir/data/feed_forward_event/Tartanair_tmp/indoor/carwelding_easy_P001/1/images_rgb --start_idx 1 --end_idx 100 --gap 5 --step 12 --save_screenshot --scene_name carwelding_easy
'''
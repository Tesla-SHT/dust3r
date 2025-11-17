import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

def get_images_from_folder(folder_path, start_idx, end_idx, gap, step):
    """
    从文件夹中按照间隔提取图片
    :param folder_path: 图片文件夹路径
    :param start_idx: 起始索引
    :param end_idx: 结束索引
    :param gap: 图片之间的间隔
    :param step: 提取的图片数量
    :return: 图片路径列表
    """
    folder = Path(folder_path)
    # 获取所有图片文件（支持常见图片格式）
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    all_images = sorted([
        str(f) for f in folder.iterdir() 
        if f.suffix.lower() in image_extensions
    ])
    if end_idx is None:
        end_idx = len(all_images) - 1 + start_idx
    image_paths = []
    for i in range(step):
        idx = start_idx + i * gap
        if idx < len(all_images) and idx <= end_idx:
            image_paths.append(all_images[idx])
        else:
            print(f"Warning: 索引 {idx} 超出范围，跳过")
    
    return image_paths

def save_scene_screenshot(scene, scene_name, idx_info, output_dir='output'):
    """
    保存场景的截图
    :param scene: 场景对象
    :param scene_name: 场景名称
    :param idx_info: 索引信息字符串
    :param output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    #在output文件夹下创建场景子文件夹
    output_dir = os.path.join(output_dir, scene_name)
    os.makedirs(output_dir, exist_ok=True)
    # 生成文件名
    filename = f"{scene_name}_idx{idx_info}.png"
    filepath = os.path.join(output_dir, filename)
    
    try:
        # 尝试使用matplotlib保存当前figure
        fig = plt.gcf()
        if fig is not None:
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"截图已保存到: {filepath}")
        else:
            print("无法获取当前图形，跳过截图")
    except Exception as e:
        print(f"保存截图时出错: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DUSt3R Usage Script')
    parser.add_argument('--mode', type=str, choices=['two_images', 'folder'], default='two_images',
                        help='加载模式：two_images 或 folder')
    parser.add_argument('--image1', type=str, default='croco/assets/Chateau1.png',
                        help='第一张图片路径（mode=two_images时使用）')
    parser.add_argument('--image2', type=str, default='croco/assets/Chateau2.png',
                        help='第二张图片路径（mode=two_images时使用）')
    parser.add_argument('--folder', type=str, default=None,
                        help='图片文件夹路径（mode=folder时使用）')
    parser.add_argument('--start_idx', type=int, default=1,
                        help='起始索引（mode=folder时使用）')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='结束索引（mode=folder时使用）')
    parser.add_argument('--gap', type=int, default=5,
                        help='图片间隔（mode=folder时使用）')
    parser.add_argument('--step', type=int, default=10,
                        help='提取的图片数量（mode=folder时使用）')
    parser.add_argument('--scene_name', type=str, default='scene',
                        help='场景名称，用于保存截图')
    parser.add_argument('--save_screenshot', action='store_true',
                        help='是否保存场景截图')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='输出文件夹路径')
    
    args = parser.parse_args()
    
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    
    # 根据模式加载图片
    if args.mode == 'two_images':
        image_paths = [args.image1, args.image2]
        idx_info = "manual"
        print(f"加载两张图片: {image_paths}")
    else:  # folder mode
        if args.folder is None:
            raise ValueError("folder模式需要指定--folder参数")
        image_paths = get_images_from_folder(args.folder, args.start_idx, args.end_idx, args.gap, args.step)
        idx_info = f"{args.start_idx}_gap{args.gap}_step{args.step}"
        print(f"从文件夹加载 {len(image_paths)} 张图片: {image_paths}")
    
    if len(image_paths) < 2:
        raise ValueError(f"至少需要2张图片，当前只有 {len(image_paths)} 张")
    
    # load_images can take a list of images or a directory
    images = load_images(image_paths, size=224)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    # next we'll use the global_aligner to align the predictions
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    # visualize reconstruction
    #scene.show()
    
    # 保存截图
    if args.save_screenshot:
        save_scene_screenshot(scene, args.scene_name, idx_info, args.output_dir)
    
    plt.show()

'''
# 模式1: 两张图片
python usage_batch.py --mode two_images --image1 path/to/img1.png --image2 path/to/img2.png --save_screenshot --scene_name chateau

# 模式2: 文件夹批量
python usage_batch.py --mode folder --folder path/to/images --start_idx 0 --gap 5 --step 10 --save_screenshot --scene_name mydata
python usage_batch.py --mode folder --folder /run/determined/workdir/data/feed_forward_event/Tartanair_tmp/indoor/carwelding_easy_P001/1/images_rgb --start_idx 1 --end_idx 100 --gap 5 --step 10 --save_screenshot --scene_name carwelding_easy
'''
import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def create_motion_blur_image(image_paths, output_path, blend_mode='average', weights=None):
    """
    将多张图片叠加生成运动模糊效果
    :param image_paths: 图片路径列表
    :param output_path: 输出图片路径
    :param blend_mode: 混合模式 ('average', 'weighted')
    :param weights: 权重列表（用于weighted模式）
    """
    if len(image_paths) == 0:
        raise ValueError("图片列表为空")
    
    # 加载第一张图片获取尺寸
    first_img = Image.open(image_paths[0])
    width, height = first_img.size
    
    # 初始化累加数组
    blurred_img = np.zeros((height, width, 3), dtype=np.float32)
    
    # 设置权重
    if blend_mode == 'weighted' and weights is None:
        # 默认使用高斯权重，中间图片权重更大
        n = len(image_paths)
        sigma = n / 6.0
        center = (n - 1) / 2.0
        weights = [np.exp(-((i - center) ** 2) / (2 * sigma ** 2)) for i in range(n)]
        weights = np.array(weights)
        weights = weights / weights.sum()
    elif blend_mode == 'average':
        weights = np.ones(len(image_paths)) / len(image_paths)
    
    print(f"使用 {blend_mode} 模式混合 {len(image_paths)} 张图片")
    if weights is not None:
        print(f"权重: {weights}")
    
    # 混合图片
    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert('RGB')
        img = img.resize((width, height), Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        blurred_img += img_array * weights[i]
        print(f"处理 {i+1}/{len(image_paths)}: {Path(img_path).name}")
    
    # 转换回uint8
    blurred_img = np.clip(blurred_img, 0, 255).astype(np.uint8)
    
    # 保存结果
    result_img = Image.fromarray(blurred_img)
    result_img.save(output_path)
    print(f"\n运动模糊图片已保存到: {output_path}")
    
    return result_img

def get_images_from_folder(folder_path, start_idx, num_images, step=1):
    """
    从文件夹中提取图片
    :param folder_path: 文件夹路径
    :param start_idx: 起始索引
    :param num_images: 图片数量
    :param step: 步长
    """
    folder = Path(folder_path)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    all_images = sorted([
        str(f) for f in folder.iterdir() 
        if f.suffix.lower() in image_extensions
    ])
    
    image_paths = []
    for i in range(num_images):
        idx = start_idx + i * step
        if idx < len(all_images):
            image_paths.append(all_images[idx])
        else:
            print(f"Warning: 索引 {idx} 超出范围")
            break
    
    return image_paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='创建运动模糊图片')
    parser.add_argument('--folder', type=str, required=True,
                        help='图片文件夹路径')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='起始索引')
    parser.add_argument('--num_images', type=int, default=10,
                        help='要混合的图片数量')
    parser.add_argument('--step', type=int, default=1,
                        help='图片之间的步长')
    parser.add_argument('--blend_mode', type=str, choices=['average', 'weighted'], 
                        default='weighted',
                        help='混合模式：average(平均) 或 weighted(高斯加权)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径')
    parser.add_argument('--output_dir', type=str, default='output_blur',
                        help='输出文件夹')
    parser.add_argument('--scene_name', type=str, default='scene',
                        help='场景名称')
    parser.add_argument('--preview', action='store_true',
                        help='显示预览')
    
    args = parser.parse_args()
    
    # 获取图片列表
    image_paths = get_images_from_folder(args.folder, args.start_idx, args.num_images, args.step)
    
    if len(image_paths) < 2:
        raise ValueError(f"至少需要2张图片，当前只有 {len(image_paths)} 张")
    
    # 设置输出路径
    if args.output is None:
        os.makedirs(args.output_dir, exist_ok=True)
        output_filename = f"{args.scene_name}_blur_{args.start_idx}to{args.start_idx + (args.num_images-1)*args.step}_n{args.num_images}.png"
        output_path = os.path.join(args.output_dir, output_filename)
    else:
        output_path = args.output
    
    # 创建运动模糊图片
    result_img = create_motion_blur_image(image_paths, output_path, args.blend_mode)
    
    # 预览
    if args.preview:
        plt.figure(figsize=(10, 8))
        plt.imshow(result_img)
        plt.title(f'Motion Blur Image ({len(image_paths)} frames, {args.blend_mode} mode)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

'''
使用示例：

# 创建运动模糊图片（平均混合10张图片）
python create_motion_blur.py \
    --folder /path/to/images \
    --start_idx 0 \
    --num_images 10 \
    --step 1 \
    --blend_mode average \
    --scene_name carwelding \
    --preview

# 创建运动模糊图片（高斯加权混合20张图片，每2帧取1张）
python create_motion_blur.py \
    --folder /path/to/images \
    --start_idx 0 \
    --num_images 20 \
    --step 2 \
    --blend_mode weighted \
    --scene_name carwelding
'''
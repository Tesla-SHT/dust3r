import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy
import matplotlib
import numpy as np
matplotlib.use('Agg')
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
    手动绘制并保存场景的截图:
    - 不显示坐标轴刻度和坐标数值
    - 保留相机姿态(类似 scene.show() 的小坐标轴)
    - 相机视角位于两帧相机后方中点, 看向点云中心
    """
    os.makedirs(output_dir, exist_ok=True)
    scene_dir = os.path.join(output_dir, scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    filename = f"{scene_name}_pair_{idx1:04d}_{idx2:04d}.png"
    filepath = os.path.join(scene_dir, filename)

    try:
        pts3d_list = scene.get_pts3d()
        masks = scene.get_masks()
        im_poses = to_numpy(scene.get_im_poses())
        focals = scene.get_focals()   # 这里先保留, 方便以后根据焦距调节可视化

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        all_pts = []
        all_colors = []

        # 1. 点云收集
        for i, (pts, mask) in enumerate(zip(pts3d_list, masks)):
            pts_np = to_numpy(pts)
            mask_np = to_numpy(mask)

            valid_pts = pts_np[mask_np]
            if valid_pts.size == 0:
                continue
            all_pts.append(valid_pts)

            # 这里简化: 用固定颜色, 不从原图采样, 画面会更干净
            color = np.array([[0.3, 0.7, 1.0]])  # 淡蓝色
            colors = np.tile(color, (len(valid_pts), 1))
            all_colors.append(colors)

        if not all_pts:
            print("没有有效点云, 跳过保存:", filepath)
            return

        all_pts = np.vstack(all_pts)
        all_colors = np.vstack(all_colors)

        # 下采样防止过密
        max_points = 80000
        if len(all_pts) > max_points:
            idx = np.random.choice(len(all_pts), max_points, replace=False)
            all_pts = all_pts[idx]
            all_colors = all_colors[idx]

        ax.scatter(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2],
                   c=all_colors, s=1, alpha=0.8)

        # 2. 计算点云包围盒, 设置等比例范围
        bbox_min = np.min(all_pts, axis=0)
        bbox_max = np.max(all_pts, axis=0)
        center = (bbox_min + bbox_max) / 2.0
        span = float(np.max(bbox_max - bbox_min))
        if span == 0:
            span = 1.0

        ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
        ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
        ax.set_zlim(center[2] - span / 2, center[2] + span / 2)
        ax.set_box_aspect([1, 1, 1])

        # 3. 画相机姿态(尽量模仿 scene.show 的小坐标轴)
        #    这里假设 im_poses[i] = [R | t] 的 3x4 或 4x4 矩阵
        cam_radius = span * 0.05  # 相机坐标轴长度
        cam_centers = []
        for i, pose in enumerate(im_poses):
            pose = np.array(pose)
            if pose.shape == (3, 4):
                R = pose[:, :3]
                t = pose[:, 3]
            elif pose.shape == (4, 4):
                R = pose[:3, :3]
                t = pose[:3, 3]
            else:
                continue

            cam_centers.append(t)

            # 相机的小坐标轴
            x_axis = R[:, 0] * cam_radius
            y_axis = R[:, 1] * cam_radius
            z_axis = R[:, 2] * cam_radius

            # x: 红, y: 绿, z: 蓝
            ax.quiver(t[0], t[1], t[2],
                      x_axis[0], x_axis[1], x_axis[2],
                      color='r', linewidth=1.0)
            ax.quiver(t[0], t[1], t[2],
                      y_axis[0], y_axis[1], y_axis[2],
                      color='g', linewidth=1.0)
            ax.quiver(t[0], t[1], t[2],
                      z_axis[0], z_axis[1], z_axis[2],
                      color='b', linewidth=1.0)

        cam_centers = np.array(cam_centers) if cam_centers else None

        # 4. 设置“拍照机位”方向:
        #    取当前这对图片对应的两台相机(假设就是 im_poses[0], im_poses[1]),
        #    计算这两台相机中心的平均, 然后沿着它们“朝向反方向”再退远一点
        #    再根据这个方向通过 view_init 设定 elev/azim.
        #    注意: matplotlib 的 view_init 是全局视角, 不能像真正相机一样任意摆放,
        #    这里用一个近似: 通过方向向量计算 (elev, azim)。

        if cam_centers is not None and len(cam_centers) >= 2:
            c1, c2 = cam_centers[0], cam_centers[1]
            mid = (c1 + c2) / 2.0

            # 用两台相机平均“前向 z 轴”的反方向作为观看方向近似
            # R[:, 2] 是相机朝前的方向, 这里取反: 在两台相机后方
            def get_R(pose):
                pose = np.array(pose)
                if pose.shape == (3, 4):
                    return pose[:, :3]
                elif pose.shape == (4, 4):
                    return pose[:3, :3]
                else:
                    return None

            R1 = get_R(im_poses[0])
            R2 = get_R(im_poses[1])
            if R1 is not None and R2 is not None:
                forward1 = R1[:, 2]
                forward2 = R2[:, 2]
                view_dir = -(forward1 + forward2) / 2.0  # “后方”
            else:
                # 退化情况: 用相机中心指向点云中心的反方向
                view_dir = mid - center

            if np.linalg.norm(view_dir) < 1e-6:
                view_dir = np.array([0, 0, 1.0])

            view_dir = view_dir / np.linalg.norm(view_dir)

            # 把 view_dir 转成 (elev, azim)
            # matplotlib: z 轴向上, x 轴右, y 轴朝屏幕里(右手系)
            # azim: 绕 z 轴的角度, elev: 仰角
            vx, vy, vz = view_dir
            azim = np.degrees(np.arctan2(vy, vx))   # [-180, 180]
            elev = np.degrees(np.arcsin(vz))        # [-90, 90]

            ax.view_init(elev=elev, azim=azim)
        else:
            # 如果没有相机信息, 用一个默认视角
            ax.view_init(elev=20, azim=45)

        # 5. 去掉坐标轴刻度 & 坐标数字, 只保留干净画面 + 相机姿态
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        # 可以保留标题, 也可以干脆去掉
        ax.set_title(f'{scene_name} [{idx1:04d}] - [{idx2:04d}]', fontsize=12)

        # 去掉外边框的网格线, 画面更干净
        ax.grid(False)

        plt.tight_layout()
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"截图已保存到: {filepath}")

    except Exception as e:
        print(f"保存截图时出错: {e}")
        traceback.print_exc()

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
    parser.add_argument('--output_dir', type=str, default='output/finetune',
                        help='输出文件夹路径')
    parser.add_argument('--model_path', type=str, 
                        default='checkpoints/dust3r_fintune_512dpt/checkpoint-best.pth',
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
                scene.show(show_cams=True)
                #save_scene_screenshot(scene, args.scene_name, idx1, idx2, args.output_dir)
            
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
python usage_batch.py --image1 path/to/img1.png --image2 path/to/img2.png --save_screenshot --scene_name chateau

# 模式2: 文件夹批量
python usage_batch.py --folder path/to/images --start_idx 0 --gap 5 --step 12 --save_screenshot --scene_name mydata
python usage_batch.py --folder /run/determined/workdir/data/feed_forward_event/Tartanair_tmp/indoor/carwelding_easy_P001/1/images_rgb --start_idx 1 --end_idx 100 --gap 5 --step 12 --save_screenshot --scene_name carwelding_easy


python usage_batch.py --folder /run/user/1001/gvfs/sftp:host=login.cvgl.lab,port=22332/datasets/feed_forward_event/Tartanair_tmp/indoor/hospital_easy_P028/1/images_rgb --start_idx 150 --end_idx 250 --gap 6 --step 10 --save_screenshot --scene_name hospital_easy_P028

'''
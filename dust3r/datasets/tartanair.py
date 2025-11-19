
from collections import deque
import os
import os.path as osp
import itertools
import json

import cv2
import numpy as np

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class TartanAir(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg
        self.dataset_label = 'TartanAir'

        # -------- 1. 读取 selected_seqs_xxx.json --------
        # 新格式：{"hospital_easy_P019": {"1": [1, 2, 3, 4, 5]}, "hospital_easy_P028": {"1": [...]}, ...}
        with open(osp.join(self.ROOT, f'selected_seqs_{self.split}.json'), 'r') as f:
            data = json.load(f)
        
        # 展开成 (scene_name, seq_id, frame_list)
        self.scenes = {}  # key: (scene_name, seq_id), value: list of frame indices
        for scene_name, seq_dict in data.items():
            for seq_id, frame_list in seq_dict.items():
                self.scenes[(scene_name, seq_id)] = frame_list
        
        # self.scene_list: 所有 (scene_name, seq_id) 元组
        self.scene_list = list(self.scenes.keys())

        # -------- 2. 构造 pairs / combinations（逻辑模仿 Co3d，但长度用真实长度） --------
        # Co3d: for each scene: 100 frames, pairs with gap in [5,10,...,30]
        # 这里：对每个 scene，找到它实际的 index 列表 / 长度，然后按同样 gap 逻辑构造 (i,j)
        self.combinations_per_scene = {}  # (scene_name, seq_id) -> [(i, j), ...]
        max_gap = 30
        step_gap = 5
        for scene_key in self.scene_list:
            # scene_key = (scene_name, seq_id)
            frame_ids = self.scenes[scene_key]  # e.g. [1, 2, 3, 4, 5]
            n = len(frame_ids)
            if n < 2:
                self.combinations_per_scene[scene_key] = []
                continue

            # 在 "帧下标" 空间里做组合（0..n-1），gap 按 Co3d 规则
            # 直接存储真实 frame_id
            combos = [
                (frame_ids[i], frame_ids[j])
                for i, j in itertools.combinations(range(n), 2)
                if 0 < abs(i - j) <= max_gap and abs(i - j) % step_gap == 0
            ]
            self.combinations_per_scene[scene_key] = combos

        # 为了 __len__ 和 idx 映射方便，做一个全局的 (scene_key, i_local, j_local) 列表
        self.global_pairs = []
        for scene_key in self.scene_list:
            for (i, j) in self.combinations_per_scene[scene_key]:
                self.global_pairs.append((scene_key, i, j))

        # 无效缓存，逻辑模仿 Co3d
        self.invalidate = {scene_key: {} for scene_key in self.scene_list}

    def __len__(self):
        return len(self.global_pairs)

    # -------------------- 路径构造：保持 TartanAir 自己的规则 -------------------- #
    def _get_metadatapath(self, scene, view_idx):
        #print("scene name", scene)
        # 根据你原先 tartanair.py 的实现来：
        # 原码里是 _get_metadatapath(self, obj, instance, view_idx)
        # 如果你的 key = "obj/instance" 这样的字符串，可以在这里拆分
        obj, instance =scene[0], scene[1]
        return osp.join(self.ROOT, obj, instance, 'images', f'frame{view_idx:06n}.npz')

    def _get_impath(self, scene, view_idx):
        
        obj, instance =scene[0], scene[1]
        return osp.join(self.ROOT, obj, instance, 'images_rgb', f'frame{view_idx:06n}.png')

    def _get_depthpath(self, scene, view_idx):
        
        obj, instance =scene[0], scene[1]
        # 先尝试 TartanAir 的 png 深度
        path = osp.join(self.ROOT, obj, instance, 'depths', f'frame{view_idx:06n}.jpg.geometric.png')
        if os.path.exists(path):
            return path
        # 再尝试 exr（你原代码里的 fallback）
        path = osp.join(self.ROOT, obj, instance, 'depths', f'{view_idx:04n}.exr')
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"Depth file not found for frame {view_idx} in {scene}")

    def _get_maskpath(self, scene, view_idx):

        obj, instance = scene[0], scene[1]
        return osp.join(self.ROOT, obj, instance, 'masks', f'frame{view_idx:06n}.png')

    def _read_depthmap(self, depthpath, input_metadata):
        depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
        depthmap = (depthmap.astype(np.float32) / 65535) * np.nan_to_num(input_metadata['maximum_depth'])
        return depthmap

    # -------------------- _get_views：整体逻辑完全模仿 Co3d -------------------- #
    def _get_views(self, idx, resolution, rng):
        """
        idx -> (scene, frame_id_1, frame_id_2)
        直接使用 frame_id 读取数据
        """
        scene, frame_id_1, frame_id_2 = self.global_pairs[idx]

        # 不使用 jitter，直接使用 frame_id
        imgs_idxs = deque([frame_id_2, frame_id_1])

        # decide now if we mask the bg（和 Co3d 一致）
        mask_bg = (self.mask_bg is True) or (self.mask_bg == 'rand' and rng.choice(2))

        views = []

        while len(imgs_idxs) > 0:
            frame_id = imgs_idxs.popleft()

            # 如果之前这个 resolution 下这帧被判 invalid，就跳过
            if resolution in self.invalidate[scene] and frame_id in self.invalidate[scene][resolution]:
                continue

            # 1. 读 metadata（TartanAir 自己的 npz 格式）
            metadata_path = self._get_metadatapath(scene, frame_id)
            try:
                input_metadata = np.load(metadata_path)
            except Exception as e:
                # 读取失败，标记 invalid，跳过
                self.invalidate[scene].setdefault(resolution, set()).add(frame_id)
                continue

            # camera_pose: 按你 tartanair 的 npz 结构来
            # 你之前示例里是 camera_pose / camera_intrinsics
            # 如果你有 "相邻帧 bias" 的逻辑，也可以在这里加回去
            camera_pose = input_metadata['camera_pose'].astype(np.float32)
            intrinsics = input_metadata['camera_intrinsics'].astype(np.float32)

            # 2. RGB
            impath = self._get_impath(scene, frame_id)
            try:
                rgb_image = imread_cv2(impath)
            except Exception:
                self.invalidate[scene].setdefault(resolution, set()).add(frame_id)
                continue

            # 3. Depth
            depthpath = self._get_depthpath(scene, frame_id)
            try:
                depthmap = self._read_depthmap(depthpath, input_metadata)
            except Exception:
                self.invalidate[scene].setdefault(resolution, set()).add(frame_id)
                continue

            # 4. 可选 mask_bg（如果你 TartanAir 有 mask）
            if mask_bg:
                maskpath = self._get_maskpath(scene, frame_id)
                if osp.exists(maskpath):
                    mask = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED)
                    # 这里简单假设：非零为前景
                    fg = (mask != 0)
                    depthmap = depthmap * fg.astype(np.float32)

            # 5. 和其它 dataset 一样，做 crop + resize
            try:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=(scene, frame_id))
            except Exception:
                self.invalidate[scene].setdefault(resolution, set()).add(frame_id)
                continue

            # 6. 构造 view dict，字段名和其它 dataset 对齐
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),      # cam2world
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='TartanAir',
                label=scene,
                instance=f'{scene}_{frame_id:06d}',
            ))

        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    # 替换为你自己的 ROOT
    dataset = TartanAir(split='train',
                        ROOT="/run/determined/workdir/home/data/feed_forward_event/Tartanair_tmp/indoor",
                        resolution=512, aug_crop=16)

    import numpy as np
    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        print(view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[0]['camera_pose'], views[1]['camera_pose']]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx * 255, (1 - idx) * 255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()

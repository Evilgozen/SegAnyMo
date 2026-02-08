# edit from https://github.com/vye16/shape-of-motion/blob/main/preproc/compute_tracks_torch.py
import argparse
import glob
import io
import os

import imageio.v2 as imageio
import mediapy as media
import numpy as np
import torch
from tapnet_torch import tapir_model, transforms
from tqdm import tqdm

import sys
sys.path.append(f'/mnt/afs/yanghongbo/My_Work/hajimi/utils')
from redis_help import RedisHelper
from aoss_help import AOSS_Client

AOSS_CONF_PATH = '/mnt/afs/yanghongbo/My_Work/hajimi/utils/aoss.conf'
S3_BUCKET_PREFIX = 's3://yanghongbo/ylr-data/P02/SAM_M'

_aoss_client = None

def get_aoss_client():
    """全局单例，懒初始化 AOSS_Client（避免序列化问题）"""
    global _aoss_client
    if _aoss_client is None:
        _aoss_client = AOSS_Client(AOSS_CONF_PATH, 'aoss')
    return _aoss_client


def aoss_list_files(dir_path):
    """从 AOSS 列举指定目录下的文件路径列表
    get_Bucket_list 返回的是子项名称，需要和目录前缀拼接成完整路径
    """
    s3_dir = f"yanghongbo/ylr-data/P02/SAM_M/{dir_path}"
    items = get_aoss_client().get_Bucket_list(s3_dir)
    prefix = s3_dir.rstrip('/') + '/'
    return sorted([prefix + item for item in items])


def aoss_read_image_np(s3_full_path):
    """从 AOSS 读取图片并返回 numpy array (HWC, uint8, RGB)
    s3_full_path: 完整的 S3 路径（不含 s3:// 前缀）
    """
    img_bytes = get_aoss_client().get_path2data(f"s3://{s3_full_path}")
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    import cv2
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_model(args):
    ## Load model
    ckpt_file = (
        "tapir_checkpoint_panning.pt"
        if args.model_type == "tapir"
        else "bootstapir_checkpoint_v2.pt"
    )
    ckpt_path = os.path.join(args.ckpt_dir, ckpt_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tapir_model.TAPIR(pyramid_level=1)
    model.load_state_dict(torch.load(ckpt_path))
    model = model.to(device)
    return model,device

def read_video(folder_path):
    frame_paths = aoss_list_files(folder_path)
    video = np.stack([aoss_read_image_np(frame_path) for frame_path in frame_paths])
    print(f"{video.shape=} {video.dtype=} {video.min()=} {video.max()=}")
    video = media._VideoArray(video)
    return video


def preprocess_frames(frames):
    """Preprocess frames to model inputs.

    Args:
      frames: [num_frames, height, width, 3], [0, 255], np.uint8

    Returns:
      frames: [num_frames, height, width, 3], [-1, 1], np.float32
    """
    frames = frames.float()
    frames = frames / 255 * 2 - 1
    if frames.shape[-1] == 4:  
        frames = frames[..., :3] 
    return frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="current-data-dir/HOI4D/images/00000", help="image dir")
    parser.add_argument("--train",action='store_true', help="image dir")
    parser.add_argument("--out_dir", type=str, default="current-data-dir/HOI4D/bootstapir/00000", help="out dir")
    parser.add_argument("--grid_size", type=int, default=None, help="grid size")
    parser.add_argument("--resize_height", type=int, default=256, help="resize height")
    parser.add_argument("--resize_width", type=int, default=256, help="resize width")
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument(
        "--model_type", type=str, choices=["tapir", "bootstapir"], help="model type"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="current_work_dir/preproc/checkpoints",
        help="checkpoint dir",
    )

    # [Fix]: 需要跑一个批量，传入对应的input_dir
    parser.add_argument("--input_dir",type=str,default=None,help="directory contains multiple videos")
    args = parser.parse_args()

    # 封装一个load_model
    model,device = load_model(args)
    
    # 创建redis
    redis_client = RedisHelper(redis_host='10.119.7.77', db_num=0, port=6379, redis_passwd='')

    # 修改逻辑，需要将本来的直接传入img_dir 和 out_dir的逻辑变为传入input_dir然后类似dino中的处理变为loop给out_dir
    if args.input_dir is not None:
        # 因为历史原因，我这里选择直接修改对应的args.image_dir,因为太多使用了这个变量
        for path in glob.glob(os.path.join(args.input_dir,'images','*')):
            args.image_dir = path
            
            if not redis_client.set(f"{path.split('/')[-1]}0205-02-bootstap",1):
                print(f'{path.split("/")[-1]}已经处理过')
                continue

            # 修改对应的out_raw_dir
            # 类似dino处理获取大的save_dir
            args.out_dir = os.path.join(os.path.dirname(os.path.dirname(args.image_dir)), "bootstapir", os.path.basename(args.image_dir))
            folder_path = args.image_dir
            # mask_dir = args.mask_dir
            frame_names = [
                os.path.basename(f) for f in aoss_list_files(folder_path)
            ]
            out_dir = args.out_dir
            os.makedirs(out_dir, exist_ok=True)

            # 检查 S3 上是否已完成
            s3_out_check = f"yanghongbo/ylr-data/P02/SAM_M/{out_dir}"
            existing_npys = get_aoss_client().get_Bucket_list(s3_out_check)
            expected_count = len(frame_names) * len(frame_names)
            npy_count = len([f for f in existing_npys if f.endswith('.npy')]) if existing_npys else 0
            done = (npy_count >= expected_count)
            print(f"{done=}")
            if done:
                print("Already done")
                return

            resize_height = args.resize_height
            resize_width = args.resize_width
            grid_size = args.grid_size

            video = read_video(folder_path)
            num_frames, height, width = video.shape[0:3]
            # masks = read_video(mask_dir)
            # masks = (masks.reshape((num_frames, height, width, -1)) > 0).any(axis=-1)
            masks = np.ones((num_frames, height, width), dtype=float)
            print(f"{video.shape=} {masks.shape=} {masks.max()=} {masks.sum()=}")

            frames = media.resize_video(video, (resize_height, resize_width))
            print(f"{frames.shape=}")
            frames = torch.from_numpy(frames).to(device)
            frames = preprocess_frames(frames)[None]
            print(f"preprocessed {frames.shape=}")

            if grid_size is None:
                max_grid_points = 9000
                grid_size = max(1, int(np.sqrt((height * width) / max_grid_points)))

            y, x = np.mgrid[0:height:grid_size, 0:width:grid_size]
            y_resize, x_resize = y / (height - 1) * (resize_height - 1), x / (width - 1) * (
                resize_width - 1
            )

            # step = 4
            # q_ts = list(range(8, 17, step))
            q_ts = list(range(0, num_frames, args.step))
            
            for t in tqdm(q_ts, desc="query frames"):
                name_t = os.path.splitext(frame_names[t])[0]
                # 检查 S3 上该 query frame 的 track 是否已完成
                s3_qt_dir = f"yanghongbo/ylr-data/P02/SAM_M/{out_dir}"
                qt_files = get_aoss_client().get_Bucket_list(s3_qt_dir) or []
                qt_matches = [f for f in qt_files if f.startswith(f"{name_t}_") and f.endswith('.npy')]
                if len(qt_matches) == num_frames:
                    print(f"Already computed tracks with query {t=} {name_t=}")
                    continue

                all_points = np.stack([t * np.ones_like(y), y_resize, x_resize], axis=-1)
                mask = masks[t]
                in_mask = mask[y, x] > 0.5
                all_points_t = all_points[in_mask]
                print(f"{all_points.shape=} {all_points_t.shape=} {t=}")
                outputs = []
                if len(all_points_t) > 0:
                    num_chunks = max(1, len(all_points_t) // 128)
                    for points in tqdm(
                        np.array_split(all_points_t, axis=0, indices_or_sections=num_chunks),
                        leave=False,
                        desc="points",
                    ):
                        points = torch.from_numpy(points.astype(np.float32))[None].to(
                            device
                        )  # Add batch dimension
                        with torch.inference_mode():
                            preds = model(frames, points)
                        tracks, occlusions, expected_dist = (
                            preds["tracks"][0].detach().cpu().numpy(),
                            preds["occlusion"][0].detach().cpu().numpy(),
                            preds["expected_dist"][0].detach().cpu().numpy(),
                        )
                        tracks = transforms.convert_grid_coordinates(
                            tracks, (resize_width - 1, resize_height - 1), (width - 1, height - 1)
                        )
                        outputs.append(
                            np.concatenate(
                                [tracks, occlusions[..., None], expected_dist[..., None]], axis=-1
                            )
                        )
                    outputs = np.concatenate(outputs, axis=0)
                else:
                    outputs = np.zeros((0, num_frames, 4), dtype=np.float32)

            for j in range(num_frames):
                if j == t:
                    original_query_points = np.stack([x[in_mask], y[in_mask]], axis=-1)
                    outputs[:, j, :2] = original_query_points
                name_j = os.path.splitext(frame_names[j])[0]
                out_path = f"{out_dir}/{name_t}_{name_j}.npy"
                buf = io.BytesIO()
                np.save(buf, outputs[:, j])
                s3_path = f"{S3_BUCKET_PREFIX}/{out_path}"
                get_aoss_client().put_data(s3_path, buf.getvalue())
        
if __name__ == "__main__":
    main()
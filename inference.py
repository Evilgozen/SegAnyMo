import os
import io
import shutil
import argparse
import torch
import torchvision
from core.utils.utils import cls_iou, load_config_file, get_feat
from core.dataset.kubric import Kubric_dataset, find_traj_label
from core.network.traj_oa_depth import traj_oa_depth
from core.network import loss_func
from core.dataset.data_utils import normalize_point_traj_torch
from glob import glob
import numpy as np
from PIL import Image
from core.utils.visualize import read_video_from_path, Visualizer, read_imgs_from_path
from core.dataset.kubric import parse_tapir_track_info, load_target_tracks
import json
from core.network.transfomer import traj_seg
from filelock import FileLock
from train_seq import setup_model
import h5py

import sys
sys.path.append('/mnt/afs/yanghongbo/My_Work/hajimi/utils')
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


def aoss_list_images(image_dir):
    """从 AOSS 列举指定目录下的图片文件路径列表
    get_Bucket_list 返回的是子项名称，需要和目录前缀拼接成完整路径
    """
    s3_dir = f"yanghongbo/ylr-data/P02/SAM_M/{image_dir}"
    items = get_aoss_client().get_Bucket_list(s3_dir)
    valid_exts = ('.png', '.jpg', '.jpeg')
    prefix = s3_dir.rstrip('/') + '/'
    paths = sorted([
        prefix + item for item in items
        if any(item.lower().endswith(ext) for ext in valid_exts)
    ])
    return paths


def aoss_read_image_pil(s3_full_path):
    """从 AOSS 读取图片并返回 PIL Image
    s3_full_path: 完整的 S3 路径（不含 s3:// 前缀），如 yanghongbo/ylr-data/P02/SAM_M/xxx.png
    """
    img_bytes = get_aoss_client().get_path2data(f"s3://{s3_full_path}")
    return Image.open(io.BytesIO(img_bytes))


def aoss_read_npy(s3_full_path):
    """从 AOSS 读取 npy 文件并返回 numpy array
    s3_full_path: 完整的 S3 路径（不含 s3:// 前缀）
    """
    npy_bytes = get_aoss_client().get_path2data(f"s3://{s3_full_path}")
    return np.load(io.BytesIO(npy_bytes))


def aoss_read_imgs(img_dir):
    """从 AOSS 读取图片目录，替代 read_imgs_from_path"""
    img_paths = aoss_list_images(img_dir)
    img_list = []
    for path in img_paths:
        pil_img = aoss_read_image_pil(path)
        img = np.array(pil_img)
        img_tensor = torch.from_numpy(img)
        img_list.append(img_tensor)
    imgs = torch.stack(img_list, dim=0).permute(0, 3, 1, 2).unsqueeze(0)
    if imgs.shape[2] == 4:
        imgs = imgs[:, :, :3, :, :]
    return imgs


def to_s3_path(local_path):
    """将本地相对路径转为 S3 完整路径（不含 s3:// 前缀）"""
    return f"yanghongbo/ylr-data/P02/SAM_M/{local_path}"


def aoss_load_target_tracks(q_name, tracks_dir, num_frames, frame_names, dim=1):
    """从 AOSS 读取 track npy 文件，替代 load_target_tracks"""
    target_indices = range(num_frames)
    all_tracks = []
    for ti in target_indices:
        t_name = frame_names[ti]
        path = f"{tracks_dir}/{q_name}_{t_name}.npy"
        tracks = aoss_read_npy(to_s3_path(path)).astype(np.float32)
        all_tracks.append(tracks)
    return torch.from_numpy(np.stack(all_tracks, axis=dim))

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
torch.autograd.set_detect_anomaly(True)

def main(cfg):
    if "dynamic_stereo" in cfg.imgs_dir:
        seq_name = os.path.basename(os.path.dirname(cfg.imgs_dir))
    else:
        seq_name = os.path.basename(cfg.imgs_dir)
    # read data
    img_dir = cfg.imgs_dir
    depth_dir = cfg.depths_dir

    imgs = aoss_read_imgs(img_dir)

    depth_paths = aoss_list_images(depth_dir)
    
    frame_names = [os.path.splitext(os.path.basename(p))[0] for p in depth_paths]
    num_frames = len(depth_paths)
    
    depth_list = []
    for path in depth_paths:
        depth_img = np.array(aoss_read_image_pil(path).convert('L'))
        depth_image_normalized = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
        depth_tensor = torch.from_numpy(depth_image_normalized)
        depth_list.append(depth_tensor)
    depths = torch.stack(depth_list, dim=0).permute(1,2,0).unsqueeze(0).unsqueeze(0)
    
    _,_,H,W,_ = depths.shape
    
    if cfg.gt_dir is not None:
        mask_dir = cfg.gt_dir
        mask_paths = sorted(glob(os.path.join(mask_dir, "*.png"))) + sorted(
            glob(os.path.join(mask_dir, "*.jpg")) + sorted(glob(os.path.join(mask_dir, "*.jpeg")))
        )
        mask_list = []
        for path in mask_paths:
            mask_img = np.array(Image.open(path).convert('L'))
            mask_img = (mask_img > 0).astype(np.uint8)
            mask_tensor = torch.from_numpy(mask_img)
            mask_list.append(mask_tensor)
        dynamic_mask = torch.stack(mask_list, dim=0)
    
    traj_label = None
    track_methods = os.path.basename(os.path.dirname(cfg.track_dir))
    if track_methods == "cotracker":
        track_path = os.path.join(cfg.track_dir,"pred_tracks.npy")
        visible_path = track_path.replace('tracks','visibility')
        
        track = torch.from_numpy(aoss_read_npy(to_s3_path(track_path))).permute(2,1,0).unsqueeze(0) # (200, 256, 2)
        visible_mask = torch.from_numpy(aoss_read_npy(to_s3_path(visible_path))).permute(1,0)
        # confi_value, visib_value, confidences = 1, 1, 1
        q_t = 0

    elif track_methods == "bootstapir":
        tracks_dir = cfg.track_dir
        q_ts = list(range(0, num_frames, cfg.step))
        ratio = 1 / len(q_ts)
        sampled_tracks = []
        sampled_visible_mask = []
        sampled_confidences = []
        sampled_visible_v = []
        sampled_confi_v = []
        traj_labels = []
        dinos = []
        
        for q_t in q_ts:
            tracks_2d = aoss_load_target_tracks(frame_names[q_t], tracks_dir, num_frames, frame_names)
            track_2d, occs, dists = (
                tracks_2d[..., :2],
                tracks_2d[..., 2],
                tracks_2d[..., 3],
            )
            visibles, _, confidences, visib_value, confi_value = parse_tapir_track_info(occs, dists, 0.5)

            # randomly pick pts to get a new track
            pts_num = track_2d.shape[0]
            num_sample = int(pts_num * ratio)
            indices = torch.randperm(pts_num)
            selected_indices = indices[:num_sample]
            
            sampled_track = track_2d[selected_indices]
            sample_visibles = visibles[selected_indices]
            sample_con = confidences[selected_indices]
            sample_v = visib_value[selected_indices]
            sample_c = confi_value[selected_indices]

            sampled_tracks.append(sampled_track)
            sampled_visible_mask.append(sample_visibles)
            sampled_confidences.append(sample_con)
            sampled_visible_v.append(sample_v)
            sampled_confi_v.append(sample_c)
            #  gt : (48,)
            if cfg.gt_dir is not None:
                gt_label = find_traj_label(sampled_track.permute(2,0,1).unsqueeze(0), ~sample_visibles.unsqueeze(0).unsqueeze(0), dynamic_mask, q_t)
                traj_labels.append(gt_label)
            if cfg.dino:
                # Load DINO feature from AOSS
                dino_dir = os.path.join(os.path.dirname(os.path.dirname(args.imgs_dir)), "dinos", os.path.basename(args.imgs_dir))
                dino_path = os.path.join(dino_dir, f'{frame_names[q_t]}.npy')
                features = aoss_read_npy(to_s3_path(dino_path))
                dino_list = torch.from_numpy(features).unsqueeze(0).unsqueeze(0)
                    
                featmap_downscale_factor = (
                    dino_list[0].shape[1]/ H,
                    dino_list[0].shape[2]/ W,
                )
                # b,c,n,t
                gt_dino = get_feat(featmap_downscale_factor, sampled_track.permute(2,0,1)[..., q_t:q_t+1], dino_list)
                if not cfg.dino_later:
                    gt_dino = gt_dino.repeat(1, 1, 1, num_frames)

                dinos.append(gt_dino)
        
        track_2d = torch.cat(sampled_tracks, dim=0)
        visibles = torch.cat(sampled_visible_mask, dim=0)
        confidences = torch.cat(sampled_confidences, dim=0)
        visib_value = torch.cat(sampled_visible_v, dim=0)
        confi_value = torch.cat(sampled_confi_v, dim=0)  
        if cfg.gt_dir is not None:      
            traj_label = np.concatenate(traj_labels)
        if cfg.dino:
            dino = torch.cat(dinos, dim=2) 
        
        total_num = 5000
        pts_num = track_2d.shape[0]
        indices = torch.randperm(pts_num)
        selected_indices = indices[:total_num]
        
        track_2d = track_2d[selected_indices]
        visibles = visibles[selected_indices]
        confidences = confidences[selected_indices]
        visib_value, confi_value = visib_value[selected_indices], confi_value[selected_indices]
        if cfg.gt_dir is not None:
            traj_label = traj_label[selected_indices]
        if cfg.dino:
            dino = dino[:, :, selected_indices, :]

        rows_all_false = torch.all((~visibles), dim=1)
        track_2d = track_2d[~rows_all_false,:,:]
        visibles = visibles[~rows_all_false,:]
        confidences = confidences[~rows_all_false,:]
        if cfg.gt_dir is not None:
            traj_label = traj_label[~rows_all_false]
        if cfg.dino:
            dino = dino[:, :, ~rows_all_false, :]
        visib_value = visib_value[~rows_all_false,:]
        confi_value = confi_value[~rows_all_false,:]
        
        cols_all_false = torch.all(~visibles.permute(1,0), dim=1)
        for t in range(cols_all_false.size(0)):
            if cols_all_false[t]:
                max_v, max_conf_index = torch.max(confidences[:, t], dim=0)
                
                visibles[max_conf_index, t] = True

        track = track_2d.permute(2,0,1).unsqueeze(0)
        visible_mask = visibles
    
    B,_,N,L = track.shape
    
    mask = (~visible_mask).unsqueeze(0).unsqueeze(0)

    if traj_label is None and cfg.gt_dir is not None:
        traj_label = find_traj_label(track,mask,dynamic_mask, q_t)
    if traj_label is not None:
        traj_label_t = torch.from_numpy(traj_label).unsqueeze(0).unsqueeze(1).float()
    if cfg.dino:
        dino_t = dino.float().cuda()

    traj = normalize_point_traj_torch(track,[H,W])

    depth_t, traj_t, mask_t = depths.float().cuda(),traj.float().cuda(), mask.float().cuda()
    
    if cfg.extra_info:
        visib_value = visib_value.unsqueeze(0).unsqueeze(0)
        confi_value = confi_value.unsqueeze(0).unsqueeze(0)
        visib_value_t, confi_value_t = visib_value.float().cuda(), confi_value.float().cuda()
        input_batch = {"traj": traj_t, "mask": mask_t, "depth": depth_t,
                        "visib_value": visib_value_t, "confi_value": confi_value_t,
                        }
    else:
        input_batch = {"traj": traj_t, "mask": mask_t, "depth": depth_t}
    
    if cfg.dino:
        input_batch["dino"] = dino_t

    # initialize model
    model = setup_model(cfg)
    model.cuda()

    if cfg.resume_path:
        print('resuming from checkpoint')
        checkpoint = torch.load(cfg.resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Load model from {}'.format(cfg.resume_path))
    else:
        print('training from scratch')

    # vis original
    # video: torch.Tensor, (B,T,C,H,W)
    # tracks: torch.Tensor, (B,T,N,2)
    # visibility: torch.Tensor, (B, T, N, 1)
    seq_save_dir = os.path.join(cfg.save_dir, seq_name)
    os.makedirs(seq_save_dir, exist_ok=True)
    vis = Visualizer(seq_save_dir,pad_value=120, linewidth=3)
    vis.visualize(imgs, track.permute(0,3,2,1), visible_mask.permute(1,0).unsqueeze(0).unsqueeze(-1), filename="original")
    
    # # better visualization
    # vis = Visualizer(
    #     save_dir="./saved_videos",
    #     pad_value=120,
    #     linewidth=3,
    #     tracks_leave_trace=-1,
    # )
    # mask_path = "exp_res/sam_res/main/last_res/initial_preds/blackswan/00000.png"
    # segm_mask = np.array(Image.open(os.path.join(mask_path)))
    # segm_mask = torch.from_numpy(segm_mask)[None, None]
    # segm_mask_true = torch.rand_like(segm_mask, dtype=torch.float32) < 0.2

    # vis.visualize(
    #         imgs,
    #         track.permute(0,3,2,1).cpu(),
    #         query_frame=0,
    #         segm_mask=segm_mask_true,
    #         compensate_for_camera_motion=True,
    #         filename='original'
    #     )

    # inference
    with torch.no_grad():
        # traj: [B, 2, N, L], depth: [B, 1, H, W, L], mask: [B, 1, N, L]
        model.eval()
        pred = model(input_batch).detach().cpu()
        
        # 需要后景的光流
        pred = 1 - pred
    
    thres = 0.7
    if cfg.gt_dir is not None:
        cur_iou = cls_iou(pred, traj_label_t, thres).item()
        print(f"IOU for {seq_name} is {cur_iou}")
        doc_path = os.path.join(cfg.save_dir, "iou.json")
        lock_path = doc_path + ".lock"

        with FileLock(lock_path):
            # 尝试从 S3 读取已有的 iou.json
            try:
                s3_doc = f"{S3_BUCKET_PREFIX}/{doc_path}"
                doc_bytes = get_aoss_client().get_path2data(s3_doc)
                existing_data = json.loads(doc_bytes.decode('utf-8'))
            except Exception:
                existing_data = {}
            
            existing_data[seq_name] = cur_iou

            mean_iou = sum(existing_data.values()) / len(existing_data)
            existing_data['mean_iou'] = mean_iou

            json_bytes = json.dumps(existing_data).encode('utf-8')
            s3_path = f"{S3_BUCKET_PREFIX}/{doc_path}"
            get_aoss_client().put_data(s3_path, json_bytes)

        pred_np = pred.cpu().numpy()
        label_np = traj_label_t.cpu().numpy()

        roc_file_path = os.path.join(cfg.save_dir, "res.npz")
        lock_roc_path = roc_file_path + ".lock"

        with FileLock(lock_roc_path):
            # 尝试从 S3 读取已有的 res.npz
            try:
                s3_roc = f"{S3_BUCKET_PREFIX}/{roc_file_path}"
                roc_bytes = get_aoss_client().get_path2data(s3_roc)
                with np.load(io.BytesIO(roc_bytes)) as data:
                    all_preds = data['all_preds']
                    all_labels = data['all_labels']
            except Exception:
                all_preds, all_labels = [], []

            new_pred = pred_np.flatten()
            new_label = label_np.flatten()

            all_preds = np.concatenate((all_preds, new_pred))
            all_labels = np.concatenate((all_labels, new_label))

            buf = io.BytesIO()
            np.savez(buf, all_preds=all_preds, all_labels=all_labels)
            s3_path = f"{S3_BUCKET_PREFIX}/{roc_file_path}"
            get_aoss_client().put_data(s3_path, buf.getvalue())
    
    min_num = 3
    thresholds = [0.95, 0.93, 0.9, 0.85, 0.8, 0.75, 0.7]

    if L > 300:
        thresholds = [0.99, 0.98, 0.97, 0.96, 0.95]
    for thres in thresholds:
        pred_thresh_mask = (pred > thres)
        
        if pred_thresh_mask.sum() >= 10:
            # print(f"{L}, {thres}, {pred_thresh_mask.sum()}")
            pred = pred_thresh_mask
            break
    else:
        # If the number of points greater than the lowest threshold is still less than min_num
        # Find the top min_num values and their indices
        topk_vals, topk_indices = torch.topk(pred.view(-1), min_num)
        # Create a mask initialized to False
        pred = torch.zeros_like(pred, dtype=torch.bool)
        # Set the positions of the top min_num values to True
        pred.view(-1)[topk_indices] = True    
        
    d_mask = pred.squeeze(0).squeeze(0)
    confidence_mask = confidences > 0.9
    visible_mask = visible_mask & confidence_mask
    dynamic_traj = (track.squeeze(0)) [:, d_mask, :]
    d_visibility = (visible_mask)[d_mask, :]
    d_confidences = (confidences)[d_mask, :]
    for fname, data in [
        ("dynamic_confidences.npy", d_confidences),
        ("dynamic_traj.npy", dynamic_traj),
        ("dynamic_visibility.npy", d_visibility),
    ]:
        buf = io.BytesIO()
        np.save(buf, data)
        s3_path = f"{S3_BUCKET_PREFIX}/{os.path.join(seq_save_dir, fname)}"
        get_aoss_client().put_data(s3_path, buf.getvalue())

    # vis
    # vis = Visualizer(save_dir=seq_save_dir, pad_value=100)
    # (B,T,C,H,W), (B,T,N,2)
    if d_visibility.shape[0] > 0:
        dynamic_traj = dynamic_traj.permute(2,1,0).unsqueeze(0)
        d_visibility = d_visibility.permute(1,0).unsqueeze(0).unsqueeze(-1)
        vis.visualize(imgs, dynamic_traj, d_visibility, filename='dynamic')

        # vis.visualize(
        #     imgs,
        #     dynamic_traj.cpu(),
        #     query_frame=0,
        #     segm_mask=segm_mask,
        #     compensate_for_camera_motion=True,
        #     filename='dynamic'
        # )
    else:
                print(f'[WARN]: {seq_name} don\'t have dynamic objects!' )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train trajectory-based motion segmentation network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str,default="exp_res/tracks_seg/rainbow")
    parser.add_argument('--imgs_dir', type=str,default="current-data-dir/davis/DAVIS/eval/480p/blackswan")
    parser.add_argument('--depths_dir', type=str,default="current-data-dir/davis/DAVIS/eval/depth_anything_v2/blackswan")
    parser.add_argument('--track_dir', type=str,default="current-data-dir/davis/DAVIS/eval/bootstapir/blackswan")    
    parser.add_argument('--gt_dir', type=str,default=None)    
    parser.add_argument('--step', type=int,default=10)    
    parser.add_argument('--config_file', metavar='DIR',default="configs/para_train3.yaml")
    args = parser.parse_args()
    cfg = load_config_file(args.config_file)
    args = parser.parse_args()
    for key, value in vars(cfg).items():
        setattr(args, key, value)
    
    main(args)
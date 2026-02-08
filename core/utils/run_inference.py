import io
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

import argparse
import glob
import cv2
import numpy as np
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

def aoss_list_files(dir_path):
    """从 AOSS 列举指定目录下的所有文件路径
    get_Bucket_list 返回的是子项名称，需要和目录前缀拼接成完整路径
    """
    s3_dir = f"yanghongbo/ylr-data/P02/SAM_M/{dir_path}"
    items = get_aoss_client().get_Bucket_list(s3_dir)
    prefix = s3_dir.rstrip('/') + '/'
    return sorted([prefix + item for item in items])

def aoss_read_image_cv2(s3_full_path):
    """从 AOSS 读取图片并返回 cv2 BGR numpy array
    s3_full_path: 完整的 S3 路径（不含 s3:// 前缀）
    """
    img_bytes = get_aoss_client().get_path2data(f"s3://{s3_full_path}")
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

def resize_images(input_dir, output_dir):    
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    
    # 从 S3 列举 input_dir 下所有文件，按子目录分组
    all_files = aoss_list_files(input_dir)
    # 按第一级子目录分组
    seq_files = {}
    prefix = input_dir.rstrip('/') + '/'
    for f in all_files:
        rel = f[len(prefix):] if f.startswith(prefix) else os.path.basename(f)
        parts = rel.split('/')
        if len(parts) >= 2:
            seq = parts[0]
            filename = parts[-1]
            seq_files.setdefault(seq, []).append((filename, f))
    
    for seq, files in seq_files.items():
        seq_out_dir = os.path.join(output_dir, seq)
        for filename, full_path in files:
            if filename.lower().endswith(valid_exts):
                output_path = os.path.join(seq_out_dir, filename)
                
                img = aoss_read_image_cv2(full_path)
                if img is None:
                    continue
                    
                h, w = img.shape[:2]
                
                max_dim = max(h, w)
                if max_dim > 1000:
                    scale = 1000 / max_dim
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                else:
                    new_w = w
                    new_h = h
                    
                resized_img = cv2.resize(img, (new_w, new_h), 
                                    interpolation=cv2.INTER_AREA)
                 
                # 上传到 AOSS
                success, buf = cv2.imencode('.png', resized_img)
                if success:
                    s3_out = f"{S3_BUCKET_PREFIX}/{output_path}"
                    get_aoss_client().put_data(s3_out, buf.tobytes())

def video_to_images(video_path, output_dir, efficiency):
    if video_path is not None:
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if efficiency:
            target_frames = min(total_frames, 100)
            frame_interval = total_frames // target_frames if total_frames > target_frames else 1
            # frame_interval = 1
        else:
            target_frames = total_frames
            frame_interval = 1
        
        frame_count = 0
        saved_frame_count = 0

        print(f'存储的结果:{target_frames}')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # [Fix]: 直接将抽帧和Resize合并
            if efficiency:
                h,w = frame.shape[:2] 
                max_dim = max(h,w)
                if max_dim > 1000:
                    scale = 1000 / max_dim
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                else:
                    new_w = w
                    new_h = h
                frame = cv2.resize(frame,(new_w,new_h),interpolation=cv2.INTER_AREA)

            
            if (efficiency and frame_count % frame_interval == 0 and saved_frame_count < target_frames) \
                or (not efficiency and saved_frame_count < target_frames):
                
                image_path = os.path.join(output_dir, f"{saved_frame_count:05d}.png")
                success, buf = cv2.imencode('.png', frame)
                if success:
                    s3_path = f"{S3_BUCKET_PREFIX}/{image_path}"
                    get_aoss_client().put_data(s3_path, buf.tobytes())
                saved_frame_count += 1
            
            frame_count += 1
        
        cap.release()
            
def main(
    args,
    depth_model: str = "depth-anything-v2",
    track_model: str = "bootstapir"
):
    gpus = args.gpus
    abs_dir = os.path.dirname(os.path.abspath(__file__))
    current_work_dir = os.path.dirname(os.path.dirname(abs_dir))
     
    stereo = False
    waymo = False
    if "stereo" in args.data_dir:
        stereo = True
        dataset = "dynamic_stereo"
        data_dir = args.data_dir
        # 从 S3 列举子目录名
        all_s3 = aoss_list_files(data_dir)
        prefix = data_dir.rstrip('/') + '/'
        names = set()
        for f in all_s3:
            rel = f[len(prefix):] if f.startswith(prefix) else f
            parts = rel.split('/')
            if parts[0] and not parts[0].endswith('.json'):
                names.add(os.path.splitext(parts[0])[0])
        img_names = sorted(names)
    elif "waymo" in args.data_dir:
        waymo = True
        data_dir = args.data_dir
        all_s3 = aoss_list_files(data_dir)
        prefix = data_dir.rstrip('/') + '/'
        names = set()
        for f in all_s3:
            rel = f[len(prefix):] if f.startswith(prefix) else f
            parts = rel.split('/')
            if parts[0] and not parts[0].endswith('.json'):
                names.add(os.path.splitext(parts[0])[0])
        img_names = sorted(names)
    # davis, kubric, HOI4D and in-the-wild data
    else:
        img_dirs_root = args.data_dir
        data_dir = os.path.dirname(img_dirs_root)
        # 从 S3 列举 images/ 下的子目录名
        all_s3 = aoss_list_files(img_dirs_root)
        prefix = img_dirs_root.rstrip('/') + '/'
        seq_names = set()
        for f in all_s3:
            rel = f[len(prefix):] if f.startswith(prefix) else f
            parts = rel.split('/')
            if len(parts) >= 2:
                seq_names.add(parts[0])
        img_names = sorted(seq_names)
    
    # 多机分片：每台机器只处理自己分到的 img_names
    img_names = img_names[args.rank::args.world_size]
    print(f'[机器 {args.rank}/{args.world_size}] main阶段分配任务数: {len(img_names)}')
    
    with ProcessPoolExecutor(max_workers=len(gpus)) as exe:
        # for i, img_name in enumerate(img_names):  # 是否为所有的img的处理结果？
        for i in range(len(gpus)): # 因为思路准确直接硬编码
            # if stereo or waymo:
            #     img_dir = os.path.join(data_dir, img_name, "images")
            # else:
            #     img_dir = os.path.join(img_dirs_root, img_name)
            # if not os.path.exists(img_dir):
            #     print(f"Skipping {img_dir} as it is not a directory")
            #     continue
            dev_id = gpus[i % len(gpus)]
            # extract DINO feature
            if args.dinos:
                cmd = ( 
                    f"CUDA_VISIBLE_DEVICES={dev_id} python {current_work_dir}/core/utils/dino_feat.py "
                    # f"--image_dir {img_dir} "
                    f"--input_dir {args.input_dir} "
                    f"--step {args.step} "
                )
                print(cmd)
                exe.submit(subprocess.call, cmd, shell=True)                

            # process dynamic mask
            if args.dynamic_mask:
                sequence_dir = os.path.join(data_dir, img_name)

                cmd = (
                    f"CUDA_VISIBLE_DEVICES={dev_id} python {current_work_dir}/core/utils/cal_dynamic_mask.py "
                    f"--data_dir {sequence_dir} --dataset {dataset} "
                )
                exe.submit(subprocess.call, cmd, shell=True)                
            
            # run depth anything
            # depth_name = depth_model.replace("-", "_")
            # if stereo or waymo:
            #     depth_dir = os.path.join(data_dir, img_name, depth_name)
            # else:
            #     depth_dir = os.path.join(data_dir, depth_name, img_name)
            if args.depths:
                cmd = (
                    f"CUDA_VISIBLE_DEVICES={dev_id} python {current_work_dir}/core/utils/run_depth.py "
                    # f"--img_dir {img_dir} --out_raw_dir {depth_dir} "
                    f"--input_dir {args.input_dir} "
                    f"--step {args.step} "
                    f"--model {depth_model}"
                )
                exe.submit(subprocess.call, cmd, shell=True)

            # run tracks model
            # if stereo or waymo:
            #     track_dir = os.path.join(data_dir, img_name, f'{track_model}')
            # else:
            #     track_dir = os.path.join(data_dir, f'{track_model}', img_name)

            if args.tracks and track_model == "cotracker":
                cmd = (
                    f"CUDA_VISIBLE_DEVICES={dev_id} python {current_work_dir}/core/utils/cotracker.py "
                    f"--imgs_dir {img_dir} --save_dir {track_dir} "
                )
                exe.submit(subprocess.call, cmd, shell=True)
            elif args.tracks and track_model == "bootstapir":
                cmd = (
                    f"CUDA_VISIBLE_DEVICES={dev_id} python {current_work_dir}/preproc/run_tapir.py "
                    f"--model_type bootstapir " 
                    # f"--image_dir {img_dir} "
                    # f"--out_dir {track_dir} "
                    f"--input_dir {args.input_dir} "
                    f"--step {args.step} "
                    f"--ckpt_dir {current_work_dir}/preproc/checkpoints "
                )
                exe.submit(subprocess.run, cmd, shell=True)
            
            # clean preprocess data
            if args.clean:
                cmd = (
                    f"CUDA_VISIBLE_DEVICES={dev_id} python {current_work_dir}/core/utils/clean_data.py "
                    f"--data_dir {img_dir} "
                )
                if waymo:
                    cmd += "--waymo"
                elif stereo:
                    cmd += "--stereo "
                exe.submit(subprocess.call, cmd, shell=True)
            
            # run inference
            gt_dir = None
            if "davis" in args.data_dir:
                gt_root = "current-data-dir/davis/DAVIS/Annotations/480p"
                gt_dir = os.path.join(gt_root, img_name)
                
            motin_seg_dir = args.motin_seg_dir
            if args.motion_seg_infer:
                cmd = (
                    f"CUDA_VISIBLE_DEVICES={dev_id} python {current_work_dir}/inference.py "
                    # f"--imgs_dir {img_dir} --save_dir {motin_seg_dir} "
                    # f"--depths_dir {depth_dir} --track_dir {track_dir} "
                    f"--input_dir {args.input_dir} "
                    f"--step {args.step} "
                    f"--config_file {args.config_file} "
                )
                print(f'motion_seg_infer:{cmd}')
                if gt_dir is not None:
                    cmd += f"--gt_dir {gt_dir} "

                exe.submit(subprocess.call, cmd, shell=True)
                
            # run SAM2
            if args.sam2:
                dynamic_dir = os.path.join(motin_seg_dir, img_name)
                cmd = (
                    f"CUDA_VISIBLE_DEVICES={dev_id} python {current_work_dir}/sam2/run_sam2.py "
                    f"--video_dir {img_dir} --dynamic_dir {dynamic_dir} "
                    f"--output_mask_dir {args.sam2dir} "
                    f"--gt_dir {gt_dir} "
                )
                exe.submit(subprocess.call, cmd, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--video_path", type=str, default=None, help="images")
    parser.add_argument("--data_dir", type=str, default="current-data-dir/baseline/SegTrackv2/JPEGImages", help="images")
    parser.add_argument('--gpus', nargs='+', type=int, default=[0], help='GPU ID')
    parser.add_argument('--track_model', type=str, default="bootstapir")
    parser.add_argument("--e", default = 'True' ,help="efficiency mode")
    parser.add_argument('--step', type=int,default=10)    
    # data process
    parser.add_argument("--depths", action='store_true')
    parser.add_argument("--tracks", action='store_true')
    parser.add_argument("--dynamic_mask", action='store_true')
    parser.add_argument("--dinos", action='store_true')
    parser.add_argument("--clean", action='store_true')
    # motion segmentation inference
    parser.add_argument("--motion_seg_infer", action='store_true')
    parser.add_argument("--motin_seg_dir", type=str, default="./test/tennis_res3", help="save motion seg pred")
    parser.add_argument('--config_file', metavar='DIR',default="configs/example.yaml")
    # sam2 inference
    parser.add_argument("--sam2", action='store_true')
    parser.add_argument("--sam2dir", type=str, default="./output/sam2/sintel", help="save sam2 pred")

    # [Fix]: 提供对应的批量处理方法
    parser.add_argument("--input_dir",type=str,default=None,help="directory contains multiple videos")
    # [Fix]: 多机分片参数
    parser.add_argument("--rank", type=int, default=0, help="当前机器编号 (0-indexed)")
    parser.add_argument("--world_size", type=int, default=1, help="总机器数量")
    args = parser.parse_args()

    sys.path.append('/mnt/afs/yanghongbo/My_Work/hajimi/utils')
    from redis_help import RedisHelper

    print(f'[Debug]:{args.e}')
    if args.e:
        print(f'已经开启了efficient模式')
        
    if args.input_dir is not None and args.video_path is not None:
        print(f'Both have input_dir and video_path')
        exit()

    # [Fix]: 修改对应的输入，直接输入一个input_dir然后处理其中的所有的文件
    if args.input_dir is not None:
        # [Fix]: 获取其中的所有的文件然后放入文件夹中
        valid_video_exts = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
        # 尝试构建一个 '/image/{name}' 的结构进行批量结果的存储
        img_root = os.path.join(args.input_dir,'images')
        os.makedirs(img_root,exist_ok=True)

        # 创建redis用于去重校验
        redis_client = RedisHelper(redis_host='10.119.7.77', db_num=0, port=6379, redis_passwd='')

        # 收集所有视频文件并排序（保证每台机器看到的顺序一致）
        all_videos = sorted([
            path for path in glob.glob(os.path.join(args.input_dir, '*'))
            if os.path.isfile(path) and path.lower().endswith(valid_video_exts)
        ])

        # 按 rank/world_size 静态分片，每台机器只处理自己的那份
        my_videos = all_videos[args.rank::args.world_size]
        print(f'[机器 {args.rank}/{args.world_size}] 总视频数: {len(all_videos)}, 本机分配: {len(my_videos)}')

        # 收集需要处理的视频任务（Redis 做二次去重，可选）
        tasks = []
        for path in my_videos:
            seq_name = os.path.splitext(os.path.basename(path))[0]
            output_dir = os.path.join(img_root, seq_name)
            if not redis_client.set(f"{seq_name}0205-02-frames", 1):
                print(f'{seq_name} 已经处理过，跳过拆帧')
                continue
            tasks.append((path, output_dir, args.e))

        # 多进程并行拆帧
        if tasks:
            num_workers = min(len(tasks), os.cpu_count() or 4)
            print(f'使用 {num_workers} 个进程并行处理 {len(tasks)} 个视频')
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(video_to_images, vpath, odir, eff): vpath
                    for vpath, odir, eff in tasks
                }
                for future in futures:
                    try:
                        future.result()
                        print(f'完成: {os.path.basename(futures[future])}')
                    except Exception as e:
                        print(f'失败: {os.path.basename(futures[future])}, 错误: {e}')

        # 最终传递的是一个root的路径，下面有对应的
        args.data_dir = img_root
        

    # if input is video
    if args.video_path is not None:
        seq_name = os.path.splitext(os.path.basename(args.video_path))[0]
        img_dir = os.path.join(os.path.dirname(args.video_path), 'images')
        output_dir = os.path.join(img_dir, seq_name)
        # 检查 S3 上是否已有拆帧结果
        s3_check = aoss_list_files(output_dir)
        if len(s3_check) == 0:
            video_to_images(args.video_path, output_dir, args.e)
        args.data_dir = img_dir
        # if efficiency, change resolution
        if args.e:
            resize_dir = os.path.join(os.path.dirname(args.data_dir),"resize_images")
            resize_images(args.data_dir, resize_dir)
            args.data_dir = resize_dir

    main(args, track_model=args.track_model)

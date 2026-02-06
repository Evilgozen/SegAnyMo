import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

import argparse
import glob
import cv2

def resize_images(input_dir, output_dir):    
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    
    for seq in os.listdir(input_dir):
        seq_in_dir = os.path.join(input_dir, seq)
        seq_out_dir = os.path.join(output_dir, seq)
        if not os.path.exists(seq_out_dir):
            os.makedirs(seq_out_dir, exist_ok=True)

            for filename in os.listdir(seq_in_dir):
                if filename.lower().endswith(valid_exts):
                    input_path = os.path.join(seq_in_dir, filename)
                    output_path = os.path.join(seq_out_dir, filename)
                    
                    img = cv2.imread(input_path)
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
                     
                    cv2.imwrite(output_path, resized_img)

def video_to_images(video_path, output_dir, efficiency):
    if video_path is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
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
                cv2.imwrite(image_path, frame)
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
        img_names = sorted([os.path.splitext(f)[0] for f in os.listdir(data_dir) if not f.endswith('.json')])
    elif "waymo" in args.data_dir:
        waymo = True
        data_dir = args.data_dir
        img_names = sorted([os.path.splitext(f)[0] for f in os.listdir(data_dir) if not f.endswith('.json')])
    # davis, kubric, HOI4D and in-the-wild data
    else:
        img_dirs_root = args.data_dir
        data_dir = os.path.dirname(img_dirs_root)
        img_names = sorted(os.listdir(img_dirs_root))
    
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
    args = parser.parse_args()

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
        # print(f'[Debug]:是否有对应的运行')
        os.makedirs(img_root,exist_ok=True)

        # 需要获取其中的所有文件然后进行后续的存储的重构
        for path in glob.glob(os.path.join(args.input_dir,'*')):
            if os.path.isfile(path) and path.lower().endswith(valid_video_exts):
                seq_name = os.path.splitext(os.path.basename(path))[0]
                output_dir = os.path.join(img_root, seq_name)

                if not os.path.exists(output_dir):
                    video_to_images(path, output_dir, args.e)
        
        # 最终传递的是一个root的路径，下面有对应的
        args.data_dir = img_root
        

    # if input is video
    if args.video_path is not None:
        seq_name = os.path.splitext(os.path.basename(args.video_path))[0]
        img_dir = os.path.join(os.path.dirname(args.video_path), 'images')
        output_dir = os.path.join(img_dir, seq_name)
        if not os.path.exists(output_dir):
            video_to_images(args.video_path, output_dir, args.e)
        args.data_dir = img_dir
        # if efficiency, change resolution
        if args.e:
            resize_dir = os.path.join(os.path.dirname(args.data_dir),"resize_images")
            resize_images(args.data_dir, resize_dir)
            args.data_dir = resize_dir

    main(args, track_model=args.track_model)

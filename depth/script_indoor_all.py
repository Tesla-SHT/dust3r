import os
import json
import csv
import argparse
import torch
from rich.console import Console
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from depth_CUT3R import run_pair, AsymmetricCroCo3DStereo

console = Console()

def generate_pairs(frame_list, pair_gap, pair_step):
    """
    从帧列表生成 (start, end) 对。
    frame_list: 已存在帧编号列表 (int)
    pair_gap: 成对视角间隔
    pair_step: 下一对起点推进步长
    """
    frames_set = set(frame_list)
    frames_sorted = sorted(frame_list)
    pairs = []
    for f in frames_sorted:
        end_f = f + pair_gap
        if end_f in frames_set:
            pairs.append((f, end_f))
    sampled = []
    used_indices = range(0, len(pairs), pair_step)
    for idx in used_indices:
        sampled.append(pairs[idx])
    return sampled

def main():
    parser = argparse.ArgumentParser(description="批量运行 TartanAir indoor 测试集多场景深度估计")
    parser.add_argument("--base_path", type=str,
                        default="/run/determined/workdir/data/feed_forward_event/Tartanair_tmp/indoor",
                        #default = "/run/user/1001/gvfs/sftp:host=login.cvgl.lab,port=22332/datasets/feed_forward_event/Tartanair_tmp/indoor",
                        help="包含 selected_seqs_test.json 的根目录")
    parser.add_argument("--json_file", type=str, default="selected_seqs_train.json",
                        help="JSON 文件名（位于 base_path 下）")
    parser.add_argument("--output_dir", type=str,
                        default="/home/w/Documents/project/data/dust3r_event_output/indoor_batch",
                        help="输出根目录")
    parser.add_argument("--model_ckpt", type=str,
                        default="checkpoints/checkpoint-best-300.pth",
                        help="模型权重")
    parser.add_argument("--input_type", type=str, default="frame_voxel",
                        choices=["frame", "voxel", "frame_voxel"])
    parser.add_argument("--size", type=int, default=512, help="图像输入尺寸")
    parser.add_argument("--pair_gap", type=int, default=4,
                        help="同一对视角内第二帧 = 第一帧 + pair_gap")
    parser.add_argument("--pair_step", type=int, default=1,
                        help="下一对起始帧步长（对起点采样）")
    parser.add_argument("--limit_pairs", type=int, default=-1,
                        help="每个场景最多处理多少对（-1 不限制）")
    parser.add_argument("--niter", type=int, default=300, help="全局对齐迭代次数")
    parser.add_argument("--lr", type=float, default=0.01, help="对齐学习率")
    parser.add_argument("--schedule", type=str, default="cosine")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    json_path = os.path.join(args.base_path, args.json_file)
    if not os.path.exists(json_path):
        console.print(f"[red]JSON 文件不存在: {json_path}[/red]")
        return
    with open(json_path, "r") as f:
        seqs = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    console.print(f"[green]加载模型 {args.model_ckpt} ...[/green]")
    model = AsymmetricCroCo3DStereo.from_pretrained(args.model_ckpt, input_type=args.input_type).to(device)
    model.eval()

    summary_rows = []
    failed_pairs = []  # ⭐ 记录失败的 pair
    summary_csv = os.path.join(args.output_dir, "summary.csv")
    failed_csv = os.path.join(args.output_dir, "failed_pairs.csv")  # ⭐ 失败记录
    
    header = ["scene", "frame_start", "frame_end",
              "rmse1", "abs_rel1", "sq_rel1", "delta1_25_1", "delta1_3_1",
              "rmse2", "abs_rel2", "sq_rel2", "delta1_25_2", "delta1_3_2"]
    
    failed_header = ["scene", "frame_start", "frame_end", "reason"]

    total_pairs = 0
    success_count = 0
    fail_count = 0

    for scene_name, seq_dict in seqs.items():
        #if ("outdoor" not in scene_name.lower()):
        #    continue
        if ("japanesealley" not in scene_name.lower()):
            continue
        if "1" not in seq_dict:
            console.print(f"[yellow]Scene {scene_name} 没有 '1' 轨迹，跳过[/yellow]")
            continue
        frame_list = seq_dict["1"]
        if not frame_list:
            continue
        console.rule(f"[bold cyan]Scene {scene_name} 共有 {len(frame_list)} 帧[/bold cyan]")

        pairs = generate_pairs(frame_list, args.pair_gap, args.pair_step)
        if args.limit_pairs > 0:
            pairs = pairs[:args.limit_pairs]

        console.print(f"[blue]生成 {len(pairs)} 对 (gap={args.pair_gap}, step={args.pair_step})[/blue]")
        total_pairs += len(pairs)

        scene_out_dir = os.path.join(args.output_dir, scene_name)
        os.makedirs(scene_out_dir, exist_ok=True)

        for (fs, fe) in pairs:
            console.print(f"[magenta]处理 {scene_name} : {fs}-{fe}[/magenta]")
            
            # ⭐ 关键修改：即使失败也继续
            try:
                res = run_pair(model=model,
                               scene_name=scene_name,
                               base_path=args.base_path,
                               frame_start_int=fs,
                               frame_end_int=fe,
                               output_dir=scene_out_dir,
                               input_type=args.input_type,
                               size=args.size,
                               niter=args.niter,
                               schedule=args.schedule,
                               lr=args.lr)
                
                if res:
                    summary_rows.append(res)
                    success_count += 1
                    console.print(f"[green]✓ 成功 ({success_count}/{total_pairs})[/green]")
                else:
                    fail_count += 1
                    failed_pairs.append({
                        "scene": scene_name,
                        "frame_start": fs,
                        "frame_end": fe,
                        "reason": "run_pair returned None"
                    })
                    console.print(f"[yellow]⚠ 跳过 ({fail_count} 失败)[/yellow]")
                
                # 及时写入成功记录
                with open(summary_csv, "w", newline='') as cf:
                    writer = csv.DictWriter(cf, fieldnames=header)
                    writer.writeheader()
                    writer.writerows(summary_rows)
                
                # 及时写入失败记录
                with open(failed_csv, "w", newline='') as ff:
                    writer = csv.DictWriter(ff, fieldnames=failed_header)
                    writer.writeheader()
                    writer.writerows(failed_pairs)
                    
            except Exception as e:
                # ⭐ 捕获脚本级别的异常（双重保险）
                fail_count += 1
                console.print(f"[red]❌ 脚本异常: {type(e).__name__}: {e}[/red]")
                failed_pairs.append({
                    "scene": scene_name,
                    "frame_start": fs,
                    "frame_end": fe,
                    "reason": f"{type(e).__name__}: {str(e)[:100]}"
                })
                continue  # 继续下一对

    # 最终汇总
    console.rule("[bold green]批处理完成[/bold green]")
    console.print(f"[cyan]总对数: {total_pairs}[/cyan]")
    console.print(f"[green]成功: {success_count}[/green]")
    console.print(f"[red]失败: {fail_count}[/red]")
    console.print(f"[blue]成功率: {success_count/total_pairs*100:.2f}%[/blue]")
    console.print(f"\n[yellow]结果汇总: {summary_csv}[/yellow]")
    console.print(f"[yellow]失败记录: {failed_csv}[/yellow]")


if __name__ == "__main__":
    main()

#MVSEC ckpt
'''
python eval/depth/script/script_indoor_all.py   --pair_gap 15 --pair_step 10 --limit_pairs 500   --output_dir output/depth_MVSEC_batch_final_gap15step10 --base_path /run/determined/workdir/data/feed_forward_event/MVSEC_all --model_ckpt checkpoints/MVSEC_indoor_finetune/checkpoint-best-mvsec-outdoor-50.pth
'''

#Tartanair
'''
python eval/depth/script/script_indoor_all.py   --pair_gap 8 --pair_step 2 --limit_pairs 200   --output_dir output/depth_JapaneseAlley --model_ckpt checkpoints/indoor_dpt512/checkpoint-best.pth --base_path /run/user/1001/gvfs/sftp:host=login.cvgl.lab,port=22332/datasets/feed_forward_event/Tartanair_tmp/indoor
'''
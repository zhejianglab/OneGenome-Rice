import pyBigWig
import numpy as np
import os
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from pathlib import Path

def calculate_total_signal(bw_path):
    """
    更准确地计算BigWig文件的总信号值
    """
    try:
        with pyBigWig.open(bw_path) as bw:
            total_signal = 0.0
            chromosomes = bw.chroms()
            
            for chrom, length in chromosomes.items():
                intervals = bw.intervals(chrom)
                if intervals is None:
                    continue
                
                chrom_signal = 0.0
                for start, end, value in intervals:
                    chrom_signal += value * (end - start)
                
                total_signal += chrom_signal
            
            return total_signal
            
    except Exception as e:
        print(f"计算总信号值时出错 {bw_path}: {e}")
        return None

def calculate_original_read_length(bw_path):
    """
    通过计算BigWig文件的总信号值来推断原始读长
    """
    total_signal = calculate_total_signal(bw_path)
    
    if total_signal is None:
        return None
    
    # 根据ENCODE规则：总信号 = 1,000,000 × 原始读长
    original_read_length = total_signal / 1e6
    
    return original_read_length

def process_single_file(args):
    """
    处理单个文件的函数，用于多进程
    """
    input_bw_path, output_bw_path, common_read_length = args
    
    print(f"开始处理: {input_bw_path}\n")
    start_time = time.time()
    
    # 推断原始读长
    original_read_length = calculate_original_read_length(input_bw_path)
    
    if original_read_length is None:
        print(f"无法推断原始读长，跳过: {input_bw_path}")
        return False
    
    print(f"文件: {os.path.basename(input_bw_path)}, 原始读长: {original_read_length:.2f} bp")
    
    # 计算缩放因子
    scale_factor = common_read_length / original_read_length
    
    try:
        # 打开输入的BigWig文件
        with pyBigWig.open(input_bw_path) as bw_in:
            # 获取染色体信息
            chromosomes = bw_in.chroms()
            # 创建输出BigWig文件
            with pyBigWig.open(output_bw_path, "w") as bw_out:
                # 添加头信息（染色体名称和大小）
                bw_out.addHeader(list(chromosomes.items()))
                
                # 处理每条染色体
                for chrom, length in chromosomes.items():
                    intervals = bw_in.intervals(chrom)
                    if intervals is None:
                        continue
                    
                    # 提取区间和值
                    starts = [interval[0] for interval in intervals]
                    ends = [interval[1] for interval in intervals]
                    values = [interval[2] * scale_factor for interval in intervals]
                    
                    # 将缩放后的值添加到输出文件
                    bw_out.addEntries(
                        [chrom] * len(starts),
                        starts,
                        ends=ends,
                        values=values
                    )
        
        # 验证输出文件
        output_total = calculate_total_signal(output_bw_path)
        if output_total is not None:
            expected_total = common_read_length * 1e6
            error_percent = abs(output_total - expected_total) / expected_total * 100
            print(f"完成: {os.path.basename(input_bw_path)} -> {os.path.basename(output_bw_path)}, "
                  f"误差: {error_percent:.4f}%, 耗时: {time.time() - start_time:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"处理文件时出错 {input_bw_path}: {e}")
        if os.path.exists(output_bw_path):
            os.remove(output_bw_path)
        return False

def batch_process_bigwigs(bigwig_paths, output_dir, common_read_length=100, num_processes=None):
    """
    批量处理BigWig文件，使用多进程加速
    
    参数:
    bigwig_paths: BigWig文件路径列表
    output_dir: 输出目录
    common_read_length: 共同读长（默认为100）
    num_processes: 进程数，默认为CPU核心数
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置进程数
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"开始批量处理 {len(bigwig_paths)} 个文件，使用 {num_processes} 个进程")
    
    # 准备参数列表
    tasks = []
    for input_path in bigwig_paths:
        if not os.path.exists(input_path):
            print(f"文件不存在: {input_path}")
            continue
        
        # 生成输出文件名
        input_filename = os.path.basename(input_path)
        output_filename = f"re-normalized_{input_filename}"
        output_path = os.path.join(output_dir, output_filename)
        
        tasks.append((input_path, output_path, common_read_length))
    
    # 使用进程池处理
    success_count = 0
    failed_count = 0
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_single_file, task): task for task in tasks}
        
        # 处理完成的任务
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                if result:
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                print(f"任务执行出错: {e}")
                failed_count += 1
    
    print(f"\n处理完成! 成功: {success_count}, 失败: {failed_count}")

def read_bigwig_list(file_path):
    """
    从文本文件中读取BigWig文件路径列表
    每行一个文件路径
    """
    bigwig_paths = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # 跳过空行和注释行
                bigwig_paths.append(line)
    return bigwig_paths

def main():
    parser = argparse.ArgumentParser(description='批量重新标准化ENCODE BigWig文件')
    parser.add_argument('--input_list', '-i', required=True, 
                       help='包含BigWig文件路径列表的文本文件，每行一个路径')
    parser.add_argument('--output_dir', '-o', required=True, 
                       help='输出目录')
    parser.add_argument('--common_length', '-c', type=int, default=100,
                       help='共同读长（默认为100）')
    parser.add_argument('--processes', '-p', type=int, default=None,
                       help='进程数，默认为CPU核心数')
    
    args = parser.parse_args()
    
    # 读取文件列表
    if not os.path.exists(args.input_list):
        print(f"输入列表文件不存在: {args.input_list}")
        return
    
    bigwig_paths = read_bigwig_list(args.input_list)
    
    if not bigwig_paths:
        print("未找到有效的BigWig文件路径")
        return
    
    print(f"找到 {len(bigwig_paths)} 个BigWig文件")
    
    # 批量处理
    batch_process_bigwigs(
        bigwig_paths=bigwig_paths,
        output_dir=args.output_dir,
        common_read_length=args.common_length,
        num_processes=args.processes 
    )


if __name__ == "__main__":
    main()
import sys
import os

sPathFolderUtilVrf = "D:/users/xiaoyaopan/PxyAI/DataSet/Vrf"
assert os.path.exists(sPathFolderUtilVrf), f"Folder does not exist: {sPathFolderUtilVrf}"
sys.path.append(sPathFolderUtilVrf)
import utilVrf
import utilDng
import utilRaw

import torch

# ----- jin1 sensor calibration result -----
jin1DataFolder = "D:/image_database/jn1_mfnr_bestshot"
assert os.path.exists(jin1DataFolder), f"Folder {jin1DataFolder} doesn't exist"
if jin1DataFolder not in sys.path:
    sys.path.append(jin1DataFolder)

def print_gpu_memory_stats(device=torch.device('cuda:0')):
    # 基础信息
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    free = total - allocated
        
    print(f"GPU 总内存: {total:.2f} GiB")
    print(f"已分配内存: {allocated:.2f} GiB")
    print(f"已保留内存: {reserved:.2f} GiB")
    print(f"可用物理内存: {free:.2f} GiB")
        
    # 详细内存统计
    print("\n🔍 详细内存统计:")
    stats = torch.cuda.memory_stats(device)
        
    # 碎片相关指标
    largest_block = stats.get('largest_block', 0) / 1024**2
    num_alloc_retries = stats.get('num_alloc_retries', 0)
    num_ooms = stats.get('num_ooms', 0)
        
    print(f"最大连续空闲块: {largest_block:.2f} MB")
    print(f"内存分配重试次数: {num_alloc_retries}")
    print(f"OOM 发生次数: {num_ooms}")
        
    # 活跃和非活跃内存
    active_bytes = stats.get('active_bytes.all.current', 0) / 1024**2
    inactive_bytes = stats.get('inactive_split_bytes.all.current', 0) / 1024**2
        
    print(f"活跃内存: {active_bytes:.2f} MB")
    print(f"非活跃/碎片内存: {inactive_bytes:.2f} MB")
        
    # 内存池信息
    print("\n📦 内存池信息:")
    for key, value in stats.items():
        if 'pool' in key or 'segment' in key or 'block' in key:
            if isinstance(value, (int, float)) and value > 0:
                if 'bytes' in key:
                    print(f"  {key}: {value/1024**2:.2f} MB")
                else:
                    print(f"  {key}: {value}")

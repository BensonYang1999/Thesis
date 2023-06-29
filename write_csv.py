import csv

# 打开文本文件
with open('0530.txt', 'r') as f:
    lines = f.readlines()

# 创建CSV文件
with open('output.csv', 'w', newline='') as csvfile:
    fieldnames = ['iter', 'PSNR', 'SSIM', 'VFID']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for line in lines:
        # 检查这一行是否包含我们需要的值
        if 'PSNR' in line and 'SSIM' in line and 'VFID' in line:
            # 分割这一行并提取所需数据
            parts = line.split()
            iter_idx = parts.index('iter')
            psnr_idx = parts.index('PSNR')
            ssim_idx = parts.index('SSIM')
            vfid_idx = parts.index('VFID')
            # 写入CSV文件
            writer.writerow({'iter': parts[iter_idx+1], 'PSNR': parts[psnr_idx+1], 'SSIM': parts[ssim_idx+1], 'VFID': parts[vfid_idx+1]})

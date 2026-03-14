import os
from flowcontainer.extractor import extract

# ================= 配置区域 =================
# 填入你的数据集总路径 (例如 F:/project/ET-BERT/data)
DATASET_PATH = r'X:\\dataDisk\\TrafficDataset\\ton_iot_attack_3k' 
# 保持与你原程序一致的过滤参数
FILTER = 'tcp' 
EXTENSION = ['tls.record.content_type', 'tls.record.opaque_type', 'tls.handshake.type']
# ===========================================

def check_pcap_files(root_path):
    bad_files = []
    print(f"开始扫描路径: {root_path}")
    print("正在查找损坏或不兼容的 PCAP 文件...\n")

    # 遍历总路径下的所有子目录和文件
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.pcap') or file.endswith('.pcapng'):
                file_path = os.path.join(root, file)
                try:
                    # 尝试调用 flowcontainer 的核心解析函数
                    # 如果该文件会导致 ValueError，会被 except 捕获
                    extract(file_path, filter=FILTER, extension=EXTENSION)
                except Exception as e:
                    print(f"[错误] 发现问题文件: {file_path}")
                    print(f"      异常信息: {e}")
                    bad_files.append(file_path)
                else:
                    # 正常的文件可以不打印，或者打印点号表示进度
                    pass 

    print("\n" + "="*50)
    if bad_files:
        print(f"扫描完成！共发现 {len(bad_files)} 个有问题的文件：")
        for bf in bad_files:
            print(bf)
    else:
        print("扫描完成！未发现会导致该错误的文件。")
    print("="*50)

if __name__ == '__main__':
    check_pcap_files(DATASET_PATH)
#!/usr/bin/python3
#-*- coding:utf-8 -*-

import os
import sys
import copy
import xlrd
import json
import tqdm
import shutil
import pickle
import random
import binascii
import operator
import numpy as np
import pandas as pd
import scapy.all as scapy
from functools import reduce
from flowcontainer.extractor import extract
from concurrent.futures import ThreadPoolExecutor, as_completed # 导入线程池模块

random.seed(40)

word_dir = "I:/corpora/"
word_name = "encrypted_burst.txt"

def convert_pcapng_2_pcap(pcapng_path, pcapng_file, output_path):
    
    pcap_file = output_path + pcapng_file.replace('pcapng','pcap')
    cmd = "I:\\editcap.exe -F pcap %s %s"
    command = cmd%(pcapng_path+pcapng_file, pcap_file)
    os.system(command)
    return 0

def split_cap(pcap_path, pcap_file, pcap_name, pcap_label='', dataset_level = 'flow'):
    # 使用 exist_ok=True 防止多线程竞争创建目录报错
    base_split_path = os.path.join(pcap_path, "splitcap")
    os.makedirs(base_split_path, exist_ok=True)

    if pcap_label != '':
        output_path = os.path.join(base_split_path, pcap_label)
    else:
        output_path = os.path.join(base_split_path, pcap_name)
    
    os.makedirs(output_path, exist_ok=True)

    if dataset_level == 'flow':
        # 注意：SplitCap.exe 路径如果包含空格，建议用引号包裹 %s
        cmd = 'E:\\SplitCap.exe -r "%s" -s session -o "%s"' % (pcap_file, output_path)
    elif dataset_level == 'packet':
        cmd = 'E:\\SplitCap.exe -r "%s" -s packets 1 -o "%s"' % (pcap_file, output_path)
    
    os.system(cmd)
    return output_path

def cut(obj, sec):
    result = [obj[i:i+sec] for i in range(0,len(obj),sec)]
    try:
        remanent_count = len(result[0])%4
    except Exception as e:
        remanent_count = 0
        print("cut datagram error!")
    if remanent_count == 0:
        pass
    else:
        result = [obj[i:i+sec+remanent_count] for i in range(0,len(obj),sec+remanent_count)]
    return result

def bigram_generation(packet_datagram, packet_len = 64, flag=True):
    result = ''
    generated_datagram = cut(packet_datagram,1)
    token_count = 0
    for sub_string_index in range(len(generated_datagram)):
        if sub_string_index != (len(generated_datagram) - 1):
            token_count += 1
            if token_count > packet_len:
                break
            else:
                merge_word_bigram = generated_datagram[sub_string_index] + generated_datagram[sub_string_index + 1]
        else:
            break
        result += merge_word_bigram
        result += ' '
    
    return result

def get_burst_feature(label_pcap, payload_len):
    feature_data = []
    
    packets = scapy.rdpcap(label_pcap)
    
    packet_direction = []
    feature_result = extract(label_pcap)
    for key in feature_result.keys():
        value = feature_result[key]
        packet_direction = [x // abs(x) for x in value.ip_lengths]

    if len(packet_direction) == len(packets):
        
        burst_data_string = ''
        
        burst_txt = ''
        
        for packet_index in range(len(packets)):
            packet_data = packets[packet_index].copy()
            data = (binascii.hexlify(bytes(packet_data)))
            
            packet_string = data.decode()[:2*payload_len]
            
            if packet_index == 0:
                burst_data_string += packet_string
            else:
                if packet_direction[packet_index] != packet_direction[packet_index - 1]:
                    
                    length = len(burst_data_string)
                    for string_txt in cut(burst_data_string, int(length / 2)):
                        burst_txt += bigram_generation(string_txt, packet_len=len(string_txt))
                        burst_txt += '\n'
                    burst_txt += '\n'
                    
                    burst_data_string = ''
                
                burst_data_string += packet_string
                if packet_index == len(packets) - 1:
                    
                    length = len(burst_data_string)
                    for string_txt in cut(burst_data_string, int(length / 2)):
                        burst_txt += bigram_generation(string_txt, packet_len=len(string_txt))
                        burst_txt += '\n'
                    burst_txt += '\n'
        
        with open(word_dir + word_name,'a') as f:
            f.write(burst_txt)
    return 0

def get_feature_packet(label_pcap,payload_len):
    feature_data = []

    try:
        # 尝试读取 PCAP 文件
        packets = scapy.rdpcap(label_pcap)
    except Exception as e:
        # 如果读取失败，打印文件名并返回空列表，避免程序中断
        print(f"\n[!] 警告: 跳过损坏文件: {label_pcap}")
        print(f"[!] 错误原因: {e}")
        return []
    packet_data_string = ''  

    for packet in packets:
            packet_data = packet.copy()
            data = (binascii.hexlify(bytes(packet_data)))
            packet_string = data.decode()
            new_packet_string = packet_string[76:]
            packet_data_string += bigram_generation(new_packet_string, packet_len=payload_len, flag = True)
            break

    feature_data.append(packet_data_string)
    return feature_data

def get_feature_flow(label_pcap, payload_len, payload_pac):
    
    feature_data = []
    packets = scapy.rdpcap(label_pcap)
    packet_count = 0  
    flow_data_string = '' 

    feature_result = extract(label_pcap, filter='tcp', extension=['tls.record.content_type', 'tls.record.opaque_type', 'tls.handshake.type'])
    if len(feature_result) == 0:
        feature_result = extract(label_pcap, filter='udp')
        if len(feature_result) == 0:
            return -1
        extract_keys = list(feature_result.keys())[0]
        if len(feature_result[label_pcap, extract_keys[1], extract_keys[2]].ip_lengths) < 3:
            print("preprocess flow %s but this flow has less than 3 packets." % label_pcap)
            return -1
    elif len(packets) < 3:
        print("preprocess flow %s but this flow has less than 3 packets." % label_pcap)
        return -1
    try:
        if len(feature_result[label_pcap, 'tcp', '0'].ip_lengths) < 3:
            print("preprocess flow %s but this flow has less than 3 packets." % label_pcap)
            return -1
    except Exception as e:
        print("*** this flow begings from 1 or other numbers than 0.")
        for key in feature_result.keys():
            if len(feature_result[key].ip_lengths) < 3:
                print("preprocess flow %s but this flow has less than 3 packets." % label_pcap)
                return -1

    if feature_result.keys() == {}.keys():
        return -1
    
    if feature_result == {}:
        return -1
    feature_result_lens = len(feature_result.keys())
    for key in feature_result.keys():
        value = feature_result[key]

    packet_index = 0
    for packet in packets:
        packet_count += 1
        if packet_count == payload_pac:
            packet_data = packet.copy()
            data = (binascii.hexlify(bytes(packet_data)))
            packet_string = data.decode()[76:]
            flow_data_string += bigram_generation(packet_string, packet_len=payload_len, flag = True)
            break
        else:
            packet_data = packet.copy()
            data = (binascii.hexlify(bytes(packet_data)))
            packet_string = data.decode()[76:]
            flow_data_string += bigram_generation(packet_string, packet_len=payload_len, flag = True)
    feature_data.append(flow_data_string)

    return feature_data

def generation(pcap_path, samples, features, splitcap = False, payload_length = 128, payload_packet = 5, dataset_save_path = "I:\\ex_results\\", dataset_level = "flow"):
    if os.path.exists(dataset_save_path + "dataset.json"):
        print("the pcap file of %s is finished generating."%pcap_path)
        
        clean_dataset = 0
        
        re_write = 0

        if clean_dataset:
            with open(dataset_save_path + "\\dataset.json", "r") as f:
                new_dataset = json.load(f)
            pop_keys = ['1','10','16','23','25','71']
            print("delete domains.")
            for p_k in pop_keys:
                print(new_dataset.pop(p_k))
            
            change_keys = [str(x) for x in range(113, 119)]
            relation_dict = {}
            for c_k_index in range(len(change_keys)):
                relation_dict[change_keys[c_k_index]] = pop_keys[c_k_index]
                new_dataset[pop_keys[c_k_index]] = new_dataset.pop(change_keys[c_k_index])
            with open(dataset_save_path + "\\dataset.json", "w") as f:
                json.dump(new_dataset, fp=f, ensure_ascii=False, indent=4)
        elif re_write:
            with open(dataset_save_path + "\\dataset.json", "r") as f:
                old_dataset = json.load(f)
            os.renames(dataset_save_path + "\\dataset.json", dataset_save_path + "\\old_dataset.json")
            with open(dataset_save_path + "\\new-samples.txt", "r") as f:
                source_samples = f.read().split('\n')
            new_dataset = {}
            samples_count = 0
            for i in range(len(source_samples)):
                current_class = source_samples[i].split('\t')
                if int(current_class[1]) > 9:
                    new_dataset[str(samples_count)] = old_dataset[str(i)]
                    samples_count += 1
                    print(old_dataset[str(i)]['samples'])
            with open(dataset_save_path + "\\dataset.json", "w") as f:
                json.dump(new_dataset, fp=f, ensure_ascii=False, indent=4)
        X, Y = obtain_data(pcap_path, samples, features, dataset_save_path)
        return X,Y

    # --- 下面是特征生成的多线程改造 ---
    dataset = {}
    label_name_list = []
    session_pcap_path = {}

    for parent, dirs, files in os.walk(pcap_path):
        if label_name_list == []:
            label_name_list.extend(dirs)
        for dir in label_name_list:
            for p,dd,ff in os.walk(parent + "\\" + dir):
                
                if splitcap:
                    for file in ff:
                        session_path = (split_cap(pcap_path, p + "\\" + file, file.split(".")[-2], dir, dataset_level = dataset_level))
                    session_pcap_path[dir] = session_path
                else:
                    session_pcap_path[dir] = pcap_path + dir
        break

    label_id = {}
    for index in range(len(label_name_list)):
        label_id[label_name_list[index]] = index

    r_file_record = []
    print("\nBegin to generate features (Multi-threaded).")

    # 定义一个内部函数，用于线程执行的具体任务
    def _extract_task(f_path, lid, level, p_len, p_pac):
        if level == "flow":
            data = get_feature_flow(f_path, payload_len=p_len, payload_pac=p_pac)
        else:
            data = get_feature_packet(f_path, payload_len=p_len)
        return f_path, lid, data

    label_count = 0
    all_tasks = []
    
    # 建立线程池
    with ThreadPoolExecutor(max_workers=8) as executor:
        # 第一步：遍历并分发任务
        for key in tqdm.tqdm(session_pcap_path.keys(), desc="Scheduling"):
            # 这里的 dataset 初始化保持原样
            if label_id[key] not in dataset:
                if dataset_level == "flow":
                    dataset[label_id[key]] = {"samples": 0, "payload": {}, "length": {}, "time": {}, "direction": {}, "message_type": {}}
                else:
                    dataset[label_id[key]] = {"samples": 0, "payload": {}}

            # 获取该类别下所有文件
            target_all_files = [x[0] + "\\" + y for x in [(p, f) for p, d, f in os.walk(session_pcap_path[key])] for y in x[1]]
            
            # 采样
            num_samples = min(len(target_all_files), samples[label_count])
            r_files = random.sample(target_all_files, num_samples)
            label_count += 1
            
            for r_f in r_files:
                # 提交任务到线程池，不阻塞，直接进入下一个循环
                task = executor.submit(_extract_task, r_f, label_id[key], dataset_level, payload_length, payload_packet)
                all_tasks.append(task)

        # 第二步：回收结果
        for future in tqdm.tqdm(as_completed(all_tasks), total=len(all_tasks), desc="Extracting"):
            r_f, lid, feature_data = future.result()
            
            if feature_data == -1 or not feature_data:
                continue
            
            r_file_record.append(r_f)
            dataset[lid]["samples"] += 1
            # 这里的 payload 赋值逻辑保留你原来的 str 索引方式
            dataset[lid]["payload"][str(dataset[lid]["samples"])] = feature_data[0]

    # --- 结尾统计和保存逻辑保持不变 ---
    all_data_number = 0
    for index in range(len(label_name_list)):
        print("%s\t%s\t%d"%(label_id[label_name_list[index]], label_name_list[index], dataset[label_id[label_name_list[index]]]["samples"]))
        all_data_number += dataset[label_id[label_name_list[index]]]["samples"]
    print("all\t%d"%(all_data_number))

    with open(dataset_save_path + "\\picked_file_record","w") as p_f:
        for i in r_file_record:
            p_f.write(i)
            p_f.write("\n")
    with open(dataset_save_path + "\\dataset.json", "w") as f:
        json.dump(dataset,fp=f,ensure_ascii=False,indent=4)

    X,Y = obtain_data(pcap_path, samples, features, dataset_save_path, json_data = dataset)
    return X,Y

def read_data_from_json(json_data, features, samples):
    X,Y = [], []
    ablation_flag = 0
    for feature_index in range(len(features)):
        x = []
        label_count = 0
        for label in json_data.keys():
            sample_num = json_data[label]["samples"]
            if X == []:
                if not ablation_flag:
                    y = [label] * sample_num
                    Y.append(y)
                else:
                    if sample_num > 1500:
                        y = [label] * 1500
                    else:
                        y = [label] * sample_num
                    Y.append(y)
            if samples[label_count] < sample_num:
                x_label = []
                for sample_index in random.sample(list(json_data[label][features[feature_index]].keys()),1500):
                    x_label.append(json_data[label][features[feature_index]][sample_index])
                x.append(x_label)
            else:
                x_label = []
                for sample_index in json_data[label][features[feature_index]].keys():
                    x_label.append(json_data[label][features[feature_index]][sample_index])
                x.append(x_label)
            label_count += 1
        X.append(x)
    return X,Y

def obtain_data(pcap_path, samples, features, dataset_save_path, json_data = None):
    
    if json_data:
        X,Y = read_data_from_json(json_data,features,samples)
    else:
        print("read dataset from json file.")
        with open(dataset_save_path + "\\dataset.json","r") as f:
            dataset = json.load(f)
        X,Y = read_data_from_json(dataset,features,samples)

    for index in range(len(X)):
        if len(X[index]) != len(Y):
            print("data and labels are not properly associated.")
            print("x:%s\ty:%s"%(len(X[index]),len(Y)))
            return -1
    return X,Y

def combine_dataset_json():
    dataset_name = "I:\\traffic_pcap\\splitcap\\dataset-"
    # dataset vocab
    dataset = {}
    # progress
    progress_num = 8
    for i in range(progress_num):
        dataset_file = dataset_name + str(i) + ".json"
        with open(dataset_file,"r") as f:
            json_data = json.load(f)
        for key in json_data.keys():
            if i > 1:
                new_key = int(key) + 9*1 + 6*(i-1)
            else:
                new_key = int(key) + 9*i
            print(new_key)
            if new_key not in dataset.keys():
                dataset[new_key] = json_data[key]
    with open("I:\\traffic_pcap\\splitcap\\dataset.json","w") as f:
        json.dump(dataset, fp=f, ensure_ascii=False, indent=4)
    return 0

def pretrain_dataset_generation(pcap_path):
    output_split_path = "I:\\dataset\\"
    pcap_output_path = "I:\\dataset\\"
    
    if not os.listdir(pcap_output_path):
        print("Begin to convert pcapng to pcap.")
        for _parent,_dirs,files in os.walk(pcap_path):
            for file in files:
                if 'pcapng' in file:
                    #print(_parent + file)
                    convert_pcapng_2_pcap(_parent, file, pcap_output_path)
                else:
                    shutil.copy(_parent+"\\"+file, pcap_output_path+file)
    
    if not os.path.exists(output_split_path + "splitcap"):
        print("Begin to split pcap as session flows in parallel.")
        
        all_files = []
        for _p, _d, files in os.walk(pcap_output_path):
            for file in files:
                if file.endswith(".pcap"): # 确保只处理 pcap
                    all_files.append((_p, file))

        # 使用线程池并行执行 split_cap
        # max_workers 建议设置为 CPU 核心数的 1-2 倍，或者根据磁盘 I/O 能力调整（例如 4-8）
        with ThreadPoolExecutor(max_workers=8) as executor:
            for _p, file in all_files:
                executor.submit(split_cap, output_split_path, _p + file, file)

    print("Begin to generate burst dataset.")
    # burst sample
    for _p,_d,files in os.walk(output_split_path + "splitcap"):
        for file in files:
            get_burst_feature(_p+"\\"+file, payload_len=64)
    return 0

def size_format(size):
    # 'KB'
    file_size = '%.3f' % float(size/1000)
    return file_size

if __name__ == '__main__':
    # pretrain
    pcap_path = "I:\\pcaps\\"
    # tls 13 downstream
    #pcap_path, samples, features = "I:\\dataset\\labeled\\", 500, ["payload","length","time","direction","message_type"]
    #X,Y = generation(pcap_path, samples, features, splitcap=False)
    # pretrain data
    pretrain_dataset_generation(pcap_path)
    #print("X:%s\tx:%s\tY:%s"%(len(X),len(X[0]),len(Y)))
    # combine dataset.json
    #combine_dataset_json()

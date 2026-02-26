#!/usr/bin/python3
#-*- coding:utf-8 -*-

import numpy as np
import json
import os
import time
import xlrd
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from scipy.stats import skew,kurtosis
import sys
import csv
import copy
import tqdm
import random
import shutil
import dataset_generation
import multiprocessing
from functools import partial

import data_preprocess
import open_dataset_deal

_category = 2 # dataset class
dataset_dir = "E:\\tsv\\zero-shot_unseen_1_5\\" # the path to save dataset for dine-tuning

pcap_path, dataset_save_path, samples, features, dataset_level = "X:\\dataDisk\\TrafficDataset\\zero-shot\\unseen_1_5\\", "E:\\npy\\zero-shot_unseen_1_5\\", [20000], ["payload"], "flow"

if not os.path.exists(dataset_dir):
    print(f"Directory {dataset_dir} does not exist. Creating it...")
    os.makedirs(dataset_dir, exist_ok=True)

if not os.path.exists(dataset_save_path):
    print(f"Directory {dataset_save_path} does not exist. Creating it...")
    os.makedirs(dataset_save_path, exist_ok=True)

def _single_class_worker(label_id, pcap_path, samples, features, dataset_save_path, dataset_level):
    # 重点 1: 为每个进程创建一个独立的临时保存目录，防止 JSON 写入冲突
    temp_save_path = os.path.join(dataset_save_path, f"temp_proc_{label_id}")
    if not os.path.exists(temp_save_path):
        os.makedirs(temp_save_path)

    # 重点 2: 构造 current_samples，确保只处理当前类
    current_samples = [0] * _category
    current_samples[label_id] = samples[label_id]
    
    # 重点 3: 调用 generation，指向临时目录
    X, Y = dataset_generation.generation(
        pcap_path, 
        current_samples, 
        features, 
        splitcap=False, 
        dataset_save_path=temp_save_path, # 关键：不共用目录
        dataset_level=dataset_level
    )
    
    # 清理临时目录（可选）
    # shutil.rmtree(temp_save_path) 
    
    return X, Y

def dataset_extract(model):
    
    X_dataset = {}
    Y_dataset = {}

    try:
        if os.listdir(dataset_save_path):
            print("Reading dataset from %s ..." % (dataset_save_path))
            
            x_payload_train, x_payload_test, x_payload_valid,\
            y_train, y_test, y_valid = \
                np.load(dataset_save_path + "x_payload_train.npy",allow_pickle=True), np.load(dataset_save_path + "x_payload_test.npy",allow_pickle=True), np.load(dataset_save_path + "x_payload_valid.npy",allow_pickle=True),\
                np.load(dataset_save_path + "y_train.npy",allow_pickle=True), np.load(dataset_save_path + "y_test.npy",allow_pickle=True), np.load(dataset_save_path + "y_valid.npy",allow_pickle=True)
            
            X_dataset, Y_dataset = models_deal(model, X_dataset, Y_dataset,
                                               x_payload_train, x_payload_test,
                                               x_payload_valid,
                                               y_train, y_test, y_valid)

            return X_dataset, Y_dataset
    except Exception as e:
        print(e)
        print("Dataset directory %s not exist.\nBegin to obtain new dataset."%(dataset_save_path))
    
    print(f"开始并行处理，最大进程数: 6")
    pool_size = min(6, _category)
    
    with multiprocessing.Pool(processes=pool_size) as pool:
        # 使用 partial 固定不变的参数
        worker_func = partial(_single_class_worker, 
                            pcap_path=pcap_path, 
                            samples=samples, 
                            features=features, 
                            dataset_save_path=dataset_save_path, 
                            dataset_level=dataset_level)
        
        # 并行执行，传入类别列表 [0, 1, 2, 3]
        results = pool.map(worker_func, range(_category))

    # 3. 合并多进程返回的数据
    # 初始化与原始格式一致的结构
    X_payload = []
    Y_all = []
    
    # 重新整合每个进程的结果
    # results 的结构是 [(X1, Y1), (X2, Y2), ...]
    for label_id in range(_category):
        res_X, res_Y = results[label_id]
        # 提取当前进程中对应的 label 数据 (res_X[feature_idx][label_idx])
        for f_idx in range(len(features)):
            if features[f_idx] == "payload":
                # 获取该 label 下的所有样本
                samples_of_label = res_X[f_idx][label_id]
                X_payload.extend(samples_of_label)
        
        # 提取对应的 Y 标签
        Y_all.extend(res_Y[label_id])

    # 4. 统计与打印 (逻辑不变)
    dataset_statistic = [0] * _category
    for label in Y_all:
        dataset_statistic[int(label)] += 1
    print("category flow statistic:")
    for index, count in enumerate(dataset_statistic):
        print(f"{index}\t{count}")

    split_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.6, random_state=41) 
    split_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=42) 

    x_payload = np.array(X_payload)
    dataset_label = np.array(Y_all)

    x_payload_train = []
    y_train = []

    x_payload_valid = []
    y_valid = []

    x_payload_test = []
    y_test = []

    for train_index, test_index in split_1.split(x_payload, dataset_label):
        x_payload_train, y_train = x_payload[train_index], dataset_label[train_index]
        x_payload_test, y_test = x_payload[test_index], dataset_label[test_index]
    for test_index, valid_index in split_2.split(x_payload_test, y_test):
        x_payload_valid, y_valid = x_payload_test[valid_index], y_test[valid_index]
        x_payload_test, y_test = x_payload_test[test_index], y_test[test_index]

    output_x_payload_train = os.path.join(dataset_save_path, 'x_payload_train.npy')

    output_x_payload_test = os.path.join(dataset_save_path, 'x_payload_test.npy')

    output_x_payload_valid = os.path.join(dataset_save_path, 'x_payload_valid.npy')

    output_y_train = os.path.join(dataset_save_path,'y_train.npy')
    output_y_test = os.path.join(dataset_save_path, 'y_test.npy')
    output_y_valid = os.path.join(dataset_save_path, 'y_valid.npy')

    np.save(output_x_payload_train, x_payload_train)
    np.save(output_x_payload_test, x_payload_test)
    np.save(output_x_payload_valid, x_payload_valid)

    np.save(output_y_train, y_train)
    np.save(output_y_test, y_test)
    np.save(output_y_valid, y_valid)

    X_dataset, Y_dataset = models_deal(model, X_dataset, Y_dataset,
                                       x_payload_train, x_payload_test, x_payload_valid,
                                       y_train, y_test, y_valid)

    return X_dataset,Y_dataset

def models_deal(model, X_dataset, Y_dataset, x_payload_train, x_payload_test, x_payload_valid, y_train, y_test, y_valid):
    for index in range(len(model)):
        print("Begin to model %s dealing..."%model[index])
        x_train_dataset = []
        x_test_dataset = []
        x_valid_dataset = []

        if model[index] == "pre-train":
            save_dir = dataset_dir
            write_dataset_tsv(x_payload_train, y_train, save_dir, "train")
            write_dataset_tsv(x_payload_test, y_test, save_dir, "test")
            write_dataset_tsv(x_payload_valid, y_valid, save_dir, "valid")
            print("finish generating pre-train's payload dataset.\nPlease check in %s" % save_dir)
            unlabel_data(dataset_dir + "test_dataset.tsv")

        X_dataset[model[index]] = {"train": [], "valid": [], "test": []}
        Y_dataset[model[index]] = {"train": [], "valid": [], "test": []}

        X_dataset[model[index]]["train"], Y_dataset[model[index]]["train"] = x_train_dataset, y_train
        X_dataset[model[index]]["valid"], Y_dataset[model[index]]["valid"] = x_valid_dataset, y_valid
        X_dataset[model[index]]["test"], Y_dataset[model[index]]["test"] = x_test_dataset, y_test

    return X_dataset, Y_dataset

def write_dataset_tsv(data,label,file_dir,type):
    dataset_file = [["label", "text_a"]]
    for index in range(len(label)):
        dataset_file.append([label[index], data[index]])
    with open(file_dir + type + "_dataset.tsv", 'w',newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        tsv_w.writerows(dataset_file)
    return 0

def unlabel_data(label_data):
    nolabel_data = ""
    with open(label_data,newline='') as f:
        data = csv.reader(f,delimiter='\t')
        for row in data:
            nolabel_data += row[1] + '\n'
    nolabel_file = label_data.replace("test_dataset","nolabel_test_dataset")
    #nolabel_file = label_data.replace("train_dataset", "nolabel_train_dataset")
    with open(nolabel_file, 'w',newline='') as f:
        f.write(nolabel_data)
    return 0

def cut_byte(obj, sec):
    result = [obj[i:i+sec] for i in range(0,len(obj),sec)]
    remanent_count = len(result[0])%2
    if remanent_count == 0:
        pass
    else:
        result = [obj[i:i+sec+remanent_count] for i in range(0,len(obj),sec+remanent_count)]
    return result

def pickle_save_data(path_file, data):
    with open(path_file, "wb") as f:
        pickle.dump(data, f)
    return 0

def count_label_number(samples):
    new_samples = samples * _category
    
    if 'splitcap' in pcap_path:
        dataset_length, labels = open_dataset_deal.statistic_dataset_sample_count(pcap_path + 'splitcap\\')
    else:
        dataset_length, labels = open_dataset_deal.statistic_dataset_sample_count(pcap_path)

    for index in range(len(dataset_length)):
        if dataset_length[index] < samples[0]:
            print("label %s has less sample's number than defined samples %d" % (labels[index], samples[0]))
            new_samples[index] = dataset_length[index]
    return new_samples

if __name__ == '__main__':
    open_dataset_not_pcap = 0
    
    if open_dataset_not_pcap:
        #open_dataset_deal.dataset_file2dir(pcap_path)
        for p,d,f in os.walk(pcap_path):
            for file in f:
                target_file = file.replace('.','_new.')
                open_dataset_deal.file_2_pcap(p+"\\"+file, p+"\\"+target_file)
                if '_new.pcap' not in file:
                    os.remove(p+"\\"+file)

    file2dir = 0
    if file2dir:
        open_dataset_deal.dataset_file2dir(pcap_path)

    splitcap_finish = 1
    if splitcap_finish:
        samples = count_label_number(samples)
    else:
        samples = samples * _category

    train_model = ["pre-train"]
    ml_experiment = 0

    dataset_extract(train_model)

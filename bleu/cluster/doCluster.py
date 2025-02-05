from bleu.cluster.dbscan import *
from bleu.cluster.spectral import *
from bleu.cluster.kmeans import *
from bleu.utils.commonUtils import *
import numpy as np
import pandas as pd


def ArtiLogTest():
    k = 22
    all_traces, traces_dict, traces_num, ori2dev_dict, flag = init_json_data()
    # all_traces, traces_dict, traces_num, ori2dev_dict, flag = load_json_data()
    print("==========数据状况==========")
    print("日志集共有", len(all_traces), "种轨迹,共有:", traces_num, "条轨迹")
    fit = sum(traces_dict[key] for key in ori2dev_dict.keys())
    print("拟合轨迹:", fit, "条,异常轨迹:", traces_num - fit, "条")
    print("----事件:", sum(len(trace.split(',')) * traces_dict[trace] for trace in all_traces), "个-----")
    if flag: return

    # dbscan约束轨迹聚类
    print("--------------------------------------------------------")
    init_center = 5
    min_clust_size = 100
    len_eps = 0.5
    bleu_data = []
    for i in range(11):
        bleu_eps = round(i * 0.1, 2)
        correct, wrong, noise, rate, rate_with_noise, cost_time = doConsCluster(init_center, bleu_eps,
                                                                                len_eps,
                                                                                min_clust_size,
                                                                                all_traces,
                                                                                traces_dict, traces_num,
                                                                                ori2dev_dict)
        bleu_data.append([bleu_eps, rate, rate_with_noise, cost_time, noise, correct, wrong])
    df_bleu = pd.DataFrame(bleu_data,
                           columns=['bleu_eps', 'rate', 'rate_with_noise', 'cost_time', 'noise', 'correct', 'wrong'])
    df_bleu.to_excel("Bleu_Data.xlsx", sheet_name='Bleu', index=False)
    # 谱聚类
    # print("--------------------------------------------------------")
    # edit_data = []
    # for i in range(11):
    #     edit_eps = round(i * 0.1, 2)
    #     correct, wrong, noise, rate, rate_with_noise, cost_time = doSpecCluster(edit_eps, k, all_traces,
    #                                                                             traces_dict,
    #                                                                             traces_num,
    #                                                                             ori2dev_dict)
    #     edit_data.append([edit_eps, rate, rate_with_noise, cost_time, noise, correct, wrong])
    # df_edit = pd.DataFrame(edit_data,
    #                        columns=['edit_eps', 'rate', 'rate_with_noise', 'cost_time', 'noise', 'correct', 'wrong'])
    # df_edit.to_excel("Edit_Data.xlsx", sheet_name='Edit', index=False)
    # K-means聚类
    # print("--------------------------------------------------------")
    # cos_data = []
    # for i in range(101):
    #     cos_eps = round(i * 0.01, 2)
    #     correct, wrong, noise, rate, rate_with_noise, cost_time = doKmeansCluster(cos_eps, k, all_traces,
    #                                                                               traces_dict,
    #                                                                               traces_num,
    #                                                                               ori2dev_dict)
    #     cos_data.append([cos_eps, rate, rate_with_noise, cost_time, noise, correct, wrong])
    # df_cos = pd.DataFrame(cos_data,
    #                       columns=['cos_eps', 'rate', 'rate_with_noise', 'cost_time', 'noise', 'correct', 'wrong'])
    # df_cos.to_excel("Cos_Data.xlsx", sheet_name='Cos', index=False)


def doConsCluster(init_center, bleu_eps, len_eps, min_clust_size, all_traces, traces_dict, traces_num, ori2dev_dict):
    print("Bleu+dbscan修复开始,bleu_eps:", bleu_eps)
    cluster_groups, cost_time = dbscan_bleu(traces_dict, all_traces, init_center, bleu_eps, len_eps, min_clust_size)
    correct, wrong, noise, rate, rate_with_noise = compare(cluster_groups, traces_dict, traces_num, ori2dev_dict)
    print("修复成功:", correct, "条,修复失败:", wrong, "条,准确率:", rate,
          "%,含噪准确率", rate_with_noise, "%,噪音数:",
          noise, "条,运算耗时:",
          cost_time, "秒")
    return correct, wrong, noise, rate, rate_with_noise, cost_time


def doSpecCluster(edit_eps, k, all_traces, traces_dict, traces_num, ori2dev_dict):
    print("编辑距离+谱聚类修复开始,edit_eps:", edit_eps)
    cluster_groups, cost_time = spectral_clustering(all_traces, traces_dict, edit_eps, k)
    correct, wrong, noise, rate, rate_with_noise = compare(cluster_groups, traces_dict, traces_num, ori2dev_dict)
    print("修复成功:", correct, "条,修复失败:", wrong, "条,准确率:", rate,
          "%,含噪准确率", rate_with_noise, "%,噪音数:",
          noise, "条,运算耗时:",
          cost_time, "秒")
    return correct, wrong, noise, rate, rate_with_noise, cost_time


def doKmeansCluster(cos_eps, k, all_traces, traces_dict, traces_num, ori2dev_dict):
    print("余弦相似度+kmeans修复开始,cos_eps:", cos_eps)
    min_cost = traces_num
    best_correct = 0
    best_wrong = 0
    best_noise = 0
    best_rate = 0.0
    best_rate_with_noise = 0.0
    best_cost_time = 0
    for i in range(10):
        cluster_groups, cost_time = kmeans_cos(all_traces, traces_dict, cos_eps, k)
        correct, wrong, noise, rate, rate_with_noise = compare(cluster_groups, traces_dict, traces_num, ori2dev_dict)
        temp_min_cost = noise + wrong
        if temp_min_cost < min_cost:
            best_correct = correct
            best_wrong = wrong
            best_noise = noise
            best_rate = rate
            best_rate_with_noise = rate_with_noise
            best_cost_time = cost_time

    print("修复成功:", best_correct, "条,修复失败:", best_wrong, "条,准确率:", best_rate,
          "%,含噪准确率", best_rate_with_noise, "%,噪音数:",
          best_noise, "条,运算耗时:",
          best_cost_time, "秒")
    return best_correct, best_wrong, best_noise, best_rate, best_rate_with_noise, best_cost_time


def doRealLogCluster(filepath, case, event):
    print(">>>>>>>>>>>>>>正在处理日志：", filepath, ">>>>>>>>>>>>>>>>")
    all_traces, traces_dict, traces_num = init_finale_data(filepath, case, event)
    avgFreq = sum(traces_dict.values()) / len(traces_dict)
    len_eps = min(max((np.log(avgFreq) / np.log(max(traces_dict.values()))), 0.3), 0.5)
    min_clust_size = min(max(avgFreq, 3), 10)
    result = dbscan_bleu(traces_dict, all_traces, 5, 0.6, len_eps, min_clust_size)
    csv_result_print(all_traces, traces_dict, traces_num, result)
    print(">>>>>>>>>>>>>>日志：", filepath, "处理完成>>>>>>>>>>>>>>>>")


def assessAlignResult(result):
    all_traces, traces_dict, traces_num, ori2dev_dict, flag = load_json_data()
    cost_time = result.get("time")
    correct, wrong, noise, rate, rate_with_noise = compare(result.get("logs"), traces_dict, traces_num,
                                                           ori2dev_dict)
    print("修复成功:", correct, "条,修复失败:", wrong, "条,准确率:", rate,
          "%,含噪准确率", rate_with_noise, "%,噪音数:",
          noise, "条,运算耗时:",
          cost_time, "秒")
    return correct, wrong, noise, rate, rate_with_noise, cost_time


def assessAlignData():
    file1 = open("bleu/data/syncResult.json")
    syncResult = json.loads(file1.read())
    file1.close()
    assessAlignResult(syncResult)
    file2 = open("bleu/data/prodResult.json")
    prodResult = json.loads(file2.read())
    file2.close()
    assessAlignResult(prodResult)


def compare(cluster_groups, traces_dict, traces_num, ori2dev_dict):
    correct = 0
    wrong = 0
    center = sum(traces_dict[key] for key in ori2dev_dict.keys())
    for cluster in cluster_groups:
        if ori2dev_dict.get(cluster[0]) is None:
            for trace in cluster:
                # print("--------------簇中心错误---------------")
                if ori2dev_dict.get(trace) is None:
                    wrong += traces_dict[trace]
        else:
            correct += traces_dict[cluster[0]]
            standard = ori2dev_dict.get(cluster[0])
            for i in range(len(cluster)):
                if i == 0:
                    continue
                if cluster[i] in standard:
                    correct += traces_dict[cluster[i]]
                elif ori2dev_dict.get(cluster[i]) is None:
                    wrong += traces_dict[cluster[i]]
    noise = traces_num - correct - wrong
    if (correct + wrong) > 0:
        rate = round((correct / (correct + wrong)) * 100, 2)
        rate_with_noise = round((correct / (correct + wrong + noise)) * 100, 2)
    else:
        rate = 0
        rate_with_noise = 0
    return correct, wrong, noise, rate, rate_with_noise

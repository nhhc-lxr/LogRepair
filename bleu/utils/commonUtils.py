import numpy as np
import json
import random
import math
import collections
import pandas as pd
from typing import List


def init_json_data():
    file = open("bleu/data/generatedTraces.json")
    ori2dev_dict = json.loads(file.read())
    file.close()
    # 创建数据变量
    all_traces: List[str] = []
    traces_dict: dict = {}
    traces_num = 0
    for key in ori2dev_dict:
        all_traces.append(key)
        ori_trace_num = random.randint(1500, 2500)
        traces_dict[key] = ori_trace_num
        traces_num += ori_trace_num
        for t in ori2dev_dict[key]:
            dev_trace_num = random.randint(1, 20)
            traces_num += dev_trace_num
            if traces_dict.get(t) is None:
                all_traces.append(t)
                traces_dict[t] = dev_trace_num
            else:
                traces_dict[t] += dev_trace_num
    all_traces.sort(key=lambda trace: traces_dict[trace], reverse=True)
    file2 = open("bleu/data/tracesDict.json", "w+")
    json.dump(traces_dict, file2)
    file2.close
    return all_traces, traces_dict, traces_num, ori2dev_dict, True


def load_json_data():
    file1 = open("bleu/data/generatedTraces.json")
    ori2dev_dict = json.loads(file1.read())
    file1.close()
    file2 = open("bleu/data/tracesDict.json")
    traces_dict = json.loads(file2.read())
    file2.close()
    all_traces = list(traces_dict.keys())
    all_traces.sort(key=lambda trace: traces_dict[trace], reverse=True)
    traces_num = sum(value for value in traces_dict.values())
    return all_traces, traces_dict, traces_num, ori2dev_dict, False


def init_finale_data(filename, case_id, activity):
    event_log = pd.read_csv(filename).loc[:, [case_id, activity]]
    # 创建数据变量
    all_traces: List[str] = []
    traces_dict: dict = {}
    activity_dict: dict = {}
    traces_num = 0
    case_dict: dict = {}

    for i in range(len(event_log)):
        # 统计活动数
        if activity_dict.get(event_log[activity][i]) is None:
            activity_dict[event_log[activity][i]] = 0
        else:
            activity_dict[event_log[activity][i]] += 1
        # 将活动串联成轨迹
        if case_dict.get(event_log[case_id][i]) is None:
            case_dict[event_log[case_id][i]] = []
        case_dict[event_log[case_id][i]].append(event_log[activity][i])
    for val in case_dict.values():
        if len(val) >= 3:
            activities = ','.join(val)
            if traces_dict.get(activities) is None:
                all_traces.append(activities)
                traces_dict[activities] = 1
            else:
                traces_dict[activities] += 1
            traces_num += 1
    all_traces.sort(key=lambda trace: traces_dict[trace], reverse=True)
    print('共有轨迹', sum(traces_dict.values()), '条，去重轨迹', len(all_traces), '条，活动', len(activity_dict.keys()),
          '个，事件',
          sum(activity_dict.values()),
          '次')
    return all_traces, traces_dict, traces_num


# 轮盘赌算法，返回随机得到的字典里的key字符串
def dict_rand(traces, k=1):
    traces_now = traces.copy()
    result = []
    for i in range(k):
        traces_num = sum(traces_now.values())
        index = random.randint(0, math.ceil(traces_num))
        for key in traces_now:
            if index < traces_now[key]:
                result.append(key)
                del traces_now[key]
                break
            else:
                index -= traces_now[key]
    return result


def get_edit_distance(trace1, trace2):
    trace1 = trace1.split(",")
    trace2 = trace2.split(",")
    matrix_ed = np.zeros((len(trace1) + 1, len(trace2) + 1), dtype=int)
    matrix_ed[0] = np.arange(len(trace2) + 1)
    matrix_ed[:, 0] = np.arange(len(trace1) + 1)
    for i in range(1, len(trace1) + 1):
        for j in range(1, len(trace2) + 1):
            # 表示删除a_i
            dist_1 = matrix_ed[i - 1, j] + 1
            # 表示插入b_i
            dist_2 = matrix_ed[i, j - 1] + 1
            # 表示替换b_i
            dist_3 = matrix_ed[i - 1, j - 1] + (1 if trace1[i - 1] != trace2[j - 1] else 0)
            # 取最小距离
            matrix_ed[i, j] = np.min([dist_1, dist_2, dist_3])
    return matrix_ed[-1, -1]


def get_cos_similarity(trace1, trace2):
    temp_trace1 = trace1.split(",")
    temp_trace2 = trace2.split(",")
    trace_union = list(set(temp_trace1).union(set(temp_trace2)))
    vector1 = np.zeros(len(trace_union))
    vector2 = np.zeros(len(trace_union))
    for i in range(len(trace_union)):
        vector1[i] = temp_trace1.count(trace_union[i])
        vector2[i] = temp_trace2.count(trace_union[i])
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


def calBleu(cand, refer, n=3):
    if cand == refer:
        return 1.0
    tmp_cand = cand.split(',')
    temp_ref = refer.split(',')
    cand_len = len(tmp_cand)
    ref_len = len(temp_ref)
    score = 0.0
    for i in range(n):
        times = 0
        ref_dict = collections.defaultdict(int)
        for k in range(ref_len - i):
            w = ",".join(temp_ref[k: k + i + 1])
            if ref_dict[w] is None:
                ref_dict[w] = 1
            else:
                ref_dict[w] += 1
        for k in range(cand_len - i):
            w = ",".join(tmp_cand[k: k + i + 1])
            if ref_dict.get(w) is not None and ref_dict[w] > 0:
                times += 1
                ref_dict[w] -= 1
        gram = times / (cand_len - i)
        if gram == 0:
            return 0.0
        score += math.log(gram)
    score /= n
    bp = math.exp(min(0, 1 - ref_len / cand_len))
    return math.exp(score) * bp


def find_center(vectors_dict, cluster, matrix):
    min_dist_val, min_dist_index = float("inf"), -1
    for i in range(len(cluster)):
        vector = cluster[i]
        dist_val = sum(
            [euclid_dist(vector, cluster[j]) * vectors_dict[matrix.index(cluster[j])] for j in range(len(cluster))])
        if min_dist_val > dist_val:
            min_dist_val = dist_val
            min_dist_index = i
    return min_dist_index


def find_center_cos(traces_dict, cluster):
    temp_k, temp_trace_index = max(
        (([sum([get_cos_similarity(cluster[i], trace) * traces_dict[trace] for trace in cluster])], i) for i in
         range(len(cluster))), key=lambda x: x[0])
    return temp_trace_index


# 多对多，返回中心点对应的字符串
def find_center_bleu(traces_dict, cluster, n=3):
    temp_k, temp_trace_index = max(
        (([sum([calBleu(cluster[i], trace, n) * traces_dict[trace] for trace in cluster])], i) for i in
         range(len(cluster))), key=lambda x: x[0])
    return temp_trace_index


def normalization(matrix):  # 归一化
    return matrix / np.sqrt(np.sum(matrix ** 2, axis=1, keepdims=True))  # 求数组的正平方根


def euclid_dist(vector_a, vector_b, sqrt_flag=False):
    dist = sum([(vector_a[i] - vector_b[i]) ** 2 for i in range(len(vector_a))])
    if sqrt_flag:
        dist = np.sqrt(dist)
    return dist


def csv_result_print(all_traces, traces_dict, traces_num, result):
    print("约束轨迹聚类完成")
    print(len(result), "个聚类")
    available_trace = 0
    repair_num = 0
    for cluster in result:
        size = 0
        for trace in cluster:
            size += traces_dict[trace]
        available_trace += size
        repair_num += size - traces_dict[cluster[0]]
        # print(size, traces_dict[cluster[0]], cluster[0])
    print(available_trace, "条有效轨迹，", traces_num - available_trace, "条噪音，修复异常轨迹", repair_num, "条")

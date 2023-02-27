from bleu.utils.commonUtils import *
from datetime import datetime
from multiprocessing import Process, cpu_count
import multiprocessing


def spectral_clustering(all_traces, traces_dict, edit_eps, k=500):
    ssss = datetime.now()
    H = process_data_(all_traces, k).tolist()
    result = []
    kmeans_result = kmeans_edit_dist(all_traces, traces_dict, H, edit_eps, 22)
    top = 0
    for cluster in kmeans_result:
        result.append([])
        for vector in cluster:
            result[top].append(all_traces[H.index(vector)])
        top += 1
    eeee = datetime.now()
    return result, round((eeee - ssss).total_seconds(), 2)


def run_faster(all_traces, start, end, total, k, result):
    neighbour_matrix = np.zeros((total, total))
    for i in range(start, end):
        for j in range(total):
            if i != j:  # 计算不重复点的距离
                neighbour_matrix[i][j] = neighbour_matrix[j][i] = 1 - get_edit_distance(all_traces[i],
                                                                                        all_traces[j]) / max(
                    len(all_traces[i]), len(all_traces[j]))
        t = np.argsort(neighbour_matrix[i, :])
        for x in range(total - k):
            neighbour_matrix[i][t[x]] = 0
        result.append(neighbour_matrix[i])


def process_data_(all_traces, k=500):
    # ssss = datetime.now()
    traces_type_num = len(all_traces)
    core_num = cpu_count()
    size = int(traces_type_num / core_num)
    remain = traces_type_num % core_num
    result_list = []
    process_list = []
    for i in range(core_num):
        result_list.append(multiprocessing.Manager().list())
    for i in range(core_num):
        if i == core_num - 1:
            process_list.append(
                Process(target=run_faster,
                        args=(all_traces, i * size, (i + 1) * size + remain, traces_type_num, k, result_list[i])))
            continue
        process_list.append(
            Process(target=run_faster, args=(all_traces, i * size, (i + 1) * size, traces_type_num, k, result_list[i])))
    for i in range(core_num):
        process_list[i].start()
    for i in range(core_num):
        process_list[i].join()
    for i in range(core_num):
        result_list[i] = np.array(result_list[i])
    result0 = np.vstack(result_list)
    neighbour_matrix = (result0 + result0.T) / 2
    # 创建拉普拉斯矩阵
    degree_matrix = np.sum(neighbour_matrix, axis=1)
    laplacian_matrix = np.diag(degree_matrix) - neighbour_matrix
    sqrt_degree_matrix = np.diag(1.0 / (degree_matrix ** 0.5))
    laplacian_matrix = np.dot(np.dot(sqrt_degree_matrix, laplacian_matrix), sqrt_degree_matrix)
    # 特征值、特征向量分解
    eigen_values, eigen_vectors = np.linalg.eig(laplacian_matrix)
    e_i = np.argsort(eigen_values)
    H = normalization(np.vstack([eigen_vectors[:, i] for i in e_i[:100]]).T)
    # eeee = datetime.now()
    # print("耗时", (eeee - ssss).seconds, "秒")
    return H


def kmeans_edit_dist(all_traces, traces_dict, matrix, edit_eps, k=22):
    # ssss = datetime.now()
    is_center_changed = True
    iter_time = 1
    inertia, inertia_after_iter = 0.0, 0.0
    centers = [matrix[i] for i in range(k)]
    vectors_dict: dict = {}
    for i in range(len(matrix)):
        vectors_dict[i] = traces_dict[all_traces[i]]
    while is_center_changed or inertia_after_iter != inertia or iter_time <= 3:
        inertia, inertia_after_iter = inertia_after_iter, 0.0
        clusters = [[val] for val in centers]
        is_center_changed = False
        for vector in matrix:
            if vector in centers:
                continue
            temp, index = min([(euclid_dist(vector, centers[i]), i) for i in range(k)], key=lambda x: x[0])
            if temp <= edit_eps:
                inertia_after_iter += temp
                clusters[index].append(vector)
        for i in range(k):
            new_center_index = find_center(vectors_dict, clusters[i], matrix)
            if new_center_index == 0:
                continue
            is_center_changed = True
            clusters[i][0], clusters[i][new_center_index] = clusters[i][new_center_index], clusters[i][0]
            centers[i] = clusters[i][0]
        iter_time += 1
    # eeee = datetime.now()
    return clusters

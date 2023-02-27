from bleu.utils.commonUtils import *
from datetime import datetime


def kmeans_cos(all_traces, traces_dict, cos_eps, k):
    ssss = datetime.now()
    is_center_changed = True
    iter_time = 1
    # centers = [all_traces[i] for i in range(k)]
    centers = random.sample(all_traces, k)
    clusters = []
    inertia, inertia_after_iter = 0.0, 0.0
    while is_center_changed or inertia_after_iter < inertia or iter_time <= 10:
        inertia, inertia_after_iter = inertia_after_iter, 0.0
        clusters = [[val] for val in centers]
        is_center_changed = False
        for trace in all_traces:
            if trace in centers:
                continue
            temp, index = max([(get_cos_similarity(trace, centers[i]), i) for i in range(k)], key=lambda x: x[0])
            if temp >= cos_eps:
                inertia_after_iter += temp
                clusters[index].append(trace)
        for i in range(k):
            new_center_index = find_center_cos(traces_dict, clusters[i])
            if new_center_index == 0:
                continue
            is_center_changed = True
            clusters[i][0], clusters[i][new_center_index] = clusters[i][new_center_index], clusters[i][0]
            centers[i] = clusters[i][0]
        iter_time += 1
    eeee = datetime.now()
    return clusters, round((eeee - ssss).total_seconds(), 2)

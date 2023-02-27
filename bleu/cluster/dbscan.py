from bleu.utils.commonUtils import *
from datetime import datetime


# DBSCAN算法(bleu_project)
def dbscan_bleu(traces_dict, all_traces,
                init_center, bleu_eps, len_eps,
                min_clust_size):
    ssss = datetime.now()
    cluster_groups = [[all_traces[i]] for i in range(init_center)]
    noise_traces = []
    for i in range(init_center, len(all_traces)):
        trace = all_traces[i]
        best_bleu, best_index = max((calBleu(trace, cluster_groups[i][0]), i) for i in range(len(cluster_groups)))
        is_bleu_sati = best_bleu >= bleu_eps
        is_len_sati = traces_dict[cluster_groups[best_index][0]] * len_eps >= traces_dict[trace]
        if is_bleu_sati and is_len_sati:
            cluster_groups[best_index].append(trace)
            new_center_index = find_center_bleu(traces_dict, cluster_groups[best_index])
            cluster_groups[best_index][0], cluster_groups[best_index][new_center_index] = \
                cluster_groups[best_index][new_center_index], cluster_groups[best_index][0]
            continue
        if ((is_bleu_sati and not is_len_sati) or (not is_bleu_sati)) and traces_dict[trace] >= min_clust_size:
            cluster_groups.append([trace])
            continue
        noise_traces.append(trace)
    changed = True
    while changed:
        changed = False
        for trace in noise_traces:
            best_bleu = 0
            best_index = 0
            for cluster in cluster_groups:
                temp_bleu = calBleu(trace, cluster[0])
                if temp_bleu > best_bleu:
                    best_bleu = temp_bleu
                    best_index = cluster_groups.index(cluster)
            is_bleu_sati = best_bleu >= bleu_eps
            is_len_sati = traces_dict[cluster_groups[best_index][0]] * len_eps >= traces_dict[trace]
            if is_bleu_sati and is_len_sati:
                cluster_groups[best_index].append(trace)
                new_center_index = find_center_bleu(traces_dict, cluster_groups[best_index])
                cluster_groups[best_index][0], cluster_groups[best_index][new_center_index] = \
                    cluster_groups[best_index][new_center_index], cluster_groups[best_index][0]
                noise_traces.remove(trace)
                changed = True
                continue
            if ((is_bleu_sati and not is_len_sati) or (not is_bleu_sati)) and traces_dict[trace] >= min_clust_size:
                cluster_groups.append([trace])
                noise_traces.remove(trace)
                changed = True
                continue
    eeee = datetime.now()
    return cluster_groups, round((eeee - ssss).total_seconds(), 2)

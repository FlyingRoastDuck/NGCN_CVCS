import torch
import collections
import numpy as np
from tqdm import tqdm
import infomap
import faiss
import math
import multiprocessing as mp
import time
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from .faiss_rerank import compute_jaccard_distance


class TextColors:
    HEADER = '\033[35m'
    OKBLUE = '\033[34m'
    OKGREEN = '\033[32m'
    WARNING = '\033[33m'
    FATAL = '\033[31m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Timer():
    def __init__(self, name='task', verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print('[Time] {} consumes {:.4f} s'.format(
                self.name,
                time.time() - self.start))
        return exc_type is None


def l2norm(vec):
    """
    归一化
    :param vec:
    :return:
    """
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


def intdict2ndarray(d, default_val=-1):
    arr = np.zeros(len(d)) + default_val
    for k, v in d.items():
        arr[k] = v
    return arr


def read_meta(fn_meta, start_pos=0, verbose=True):
    """
    idx2lb：每一个顶点对应一个类
    lb2idxs：每个类对应一个id
    """
    lb2idxs = {}
    idx2lb = {}
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[start_pos:]):
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            idx2lb[idx] = lb

    inst_num = len(idx2lb)
    cls_num = len(lb2idxs)
    if verbose:
        print('[{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))
    return lb2idxs, idx2lb


def knns2ordered_nbrs(knns, sort=True):
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :].astype(np.int32)
    dists = knns[:, 1, :]
    if sort:
        # sort dists from low to high
        nb_idx = np.argsort(dists, axis=1)
        idxs = np.arange(nb_idx.shape[0]).reshape(-1, 1)
        dists = dists[idxs, nb_idx]
        nbrs = nbrs[idxs, nb_idx]
    return dists, nbrs


# def get_links(single, links, nbrs, dists, min_sim):
#     for i in tqdm(range(nbrs.shape[0])):
#         count = 0
#         for j in range(0, len(nbrs[i])):
#             if i == nbrs[i][j]:
#                 pass
#             elif dists[i][j] <= 1 - min_sim:
#                 count += 1
#                 links[(i, nbrs[i][j])] = float(1 - dists[i][j])
#             else:
#                 break
#         if count == 0:
#             single.append(i)
#     return single, links

def get_links(single, links, nbrs, dists, min_sim):
    nbrs, dists = nbrs.todense(), dists.todense()

    temp_nbrs = nbrs + nbrs.T
    temp_dists = dists + dists.T
    
    for ii in range(temp_nbrs.shape[0]):
        locs = [idx for idx in range(temp_nbrs.shape[1]) if (temp_nbrs[ii,idx]!=0 and ii!=idx)]

        if len(locs)==0: single.append(ii)
        else:
            for val in locs:
                links[(ii, val)] = float(temp_dists[ii,val])
    return single, links


def get_links_with_label(single, links, nbrs, dists, min_sim, label_mark, if_labeled, args=None):
    if args is not None and args.max_sim:
        for i in tqdm(range(nbrs.shape[0])):
            count = 0
            for j in range(0, len(nbrs[i])):
                # 排除本身节点
                if i == nbrs[i][j]:
                    pass
                elif dists[i][j] <= 1 - min_sim:
                    count += 1
                    if label_mark[i] == label_mark[j] and if_labeled[i] == True:
                        links[(i, nbrs[i][j])] = float(
                            1 - max(min(dists[i][1:]), 0))
                    else:
                        links[(i, nbrs[i][j])] = float(1 - dists[i][j])
                else:
                    break
            # 统计孤立点
            if count == 0:
                single.append(i)
        return single, links
        # for i in tqdm(range(nbrs.shape[0])):
        #     count = 0
        #     for j in range(0, len(nbrs[i])):
        #         # 排除本身节点
        #         if i == nbrs[i][j]:
        #             pass
        #         elif dists[i][j] <= 1 - min_sim:
        #             count += 1
        #             links[(i, nbrs[i][j])] = float(1 - dists[i][j])
        #         else:
        #             break
        #     # 统计孤立点
        #     if count == 0:
        #         single.append(i)
        # ref_link = copy.deepcopy(links)
        # for i in tqdm(range(nbrs.shape[0])):
        #     count = 0
        #     for j in range(0, len(nbrs[i])):
        #         # 排除本身节点
        #         if i == nbrs[i][j]:
        #             pass
        #         elif dists[i][j] <= 1 - min_sim:
        #             count += 1
        #             if label_mark[i] == label_mark[j] and if_labeled[i] == True:
        #                 links[(i, nbrs[i][j])] = max([ref_link[(i, pp)] for pp in nbrs[i, 1:]])
        #             else:
        #                 links[(i, nbrs[i][j])] = float(1 - dists[i][j])
        #         else:
        #             break
        #     # 统计孤立点
        #     if count == 0:
        #         single.append(i)
    else:
        for i in tqdm(range(nbrs.shape[0])):
            count = 0
            for j in range(0, len(nbrs[i])):
                # 排除本身节点
                if i == nbrs[i][j]:
                    pass
                elif dists[i][j] <= 1 - min_sim:
                    count += 1
                    if label_mark[i] == label_mark[j] and if_labeled[i] == True:
                        links[(i, nbrs[i][j])] = float(0.999999999)
                    else:
                        links[(i, nbrs[i][j])] = float(1 - dists[i][j])
                else:
                    break
            # 统计孤立点
            if count == 0:
                single.append(i)
        return single, links

@torch.no_grad()
def get_centers(pseudo_labels, features):
    useful = features[pseudo_labels!=-1].clone()
    labels = pseudo_labels[pseudo_labels!=-1].clone()
    max_pids = pseudo_labels.max()+1
    centers = torch.zeros(max_pids, features.shape[1]).cuda()
    counters = torch.zeros(max_pids, 1).cuda()
    counters.index_add_(0, labels, torch.ones(max_pids).cuda())
    centers.index_add_(0, labels, useful)
    centers /= counters
    return centers
    

def cluster_by_semi_infomap(
    nbrs, dists, min_sim, cluster_num=2,
    label_mark=None, if_labeled=None, args=None
):
    if label_mark is None:
        single, links = [], {}
        with Timer('get links', verbose=True):
            single, links = get_links(
                single=single, links=links, nbrs=nbrs, dists=dists, min_sim=min_sim)

        infomapWrapper = infomap.Infomap("--two-level --directed")

        for (i, j), sim in tqdm(links.items()):
            _ = infomapWrapper.addLink(int(i), int(j), sim)

        infomapWrapper.run()
        label2idx, idx2label = {}, {}
        # 聚类结果统计
        for node in infomapWrapper.iterTree():
            if node.moduleIndex() not in label2idx:
                label2idx[node.moduleIndex()] = []
            label2idx[node.moduleIndex()].append(node.physicalId)

        node_count = 0
        for k, v in label2idx.items():
            if k == 0:
                each_index_list = v[2:]
                node_count += len(each_index_list)
                label2idx[k] = each_index_list
            else:
                each_index_list = v[1:]
                node_count += len(each_index_list)
                label2idx[k] = each_index_list
                
            for each_index in each_index_list:
                idx2label[each_index] = k
                
        keys_len = len(list(label2idx.keys()))  # total clusters w/o single


        # 孤立点放入到结果中
        for single_node in single:
            idx2label[single_node] = keys_len
            label2idx[keys_len] = [single_node]
            keys_len += 1
            node_count += 1

        # 孤立点个数
        print("孤立点数：{}".format(len(single)))

        unknown = [val for val in range(5994) if val not in idx2label]
        for single_node in unknown:
            idx2label[single_node] = keys_len
            label2idx[keys_len] = [single_node]
            keys_len += 1
            node_count += 1
        idx_len = len(list(idx2label.keys()))
        assert idx_len == node_count, 'idx_len not equal node_count!'

        print("总节点数：{}".format(idx_len))

        old_label_container = set()
        for each_label, each_index_list in label2idx.items():
            if len(each_index_list) <= cluster_num:
                for each_index in each_index_list:
                    idx2label[each_index] = -1
            else:
                old_label_container.add(each_label)

        old2new = {old_label: new_label for new_label,
                   old_label in enumerate(old_label_container)}

        for each_index, each_label in idx2label.items():
            if each_label == -1: continue
            # filter out single
            idx2label[each_index] = old2new[each_label]

        pre_labels = intdict2ndarray(idx2label)  # pseudo labels
        # all clusters v.s. labelled classes
        print("总类别数：{}/{}".format(keys_len, len(set(pre_labels)) -
              (1 if -1 in pre_labels else 0)))
        return pre_labels
    else:
        # this way
        single, links = [], {}
        with Timer('get links', verbose=True):
            single, links = get_links_with_label(
                single=single, links=links, nbrs=nbrs, dists=dists,
                min_sim=min_sim, label_mark=label_mark, if_labeled=if_labeled, args=args
            )

        infomapWrapper = infomap.Infomap("--two-level --directed")
        for (i, j), sim in tqdm(links.items()):
            _ = infomapWrapper.addLink(int(i), int(j), sim)

        infomapWrapper.run()
        label2idx, idx2label = {}, {}
        for node in infomapWrapper.iterTree():
            if node.moduleIndex() not in label2idx:
                label2idx[node.moduleIndex()] = []
            label2idx[node.moduleIndex()].append(node.physicalId)

        node_count = 0
        for k, v in label2idx.items():
            if k == 0:
                each_index_list = v[2:]
                node_count += len(each_index_list)
                label2idx[k] = each_index_list
            else:
                each_index_list = v[1:]
                node_count += len(each_index_list)
                label2idx[k] = each_index_list

            for each_index in each_index_list:
                idx2label[each_index] = k

        keys_len = len(list(label2idx.keys()))
        for single_node in single:
            idx2label[single_node] = keys_len
            label2idx[keys_len] = [single_node]
            keys_len += 1
            node_count += 1

        print("孤立点数：{}".format(len(single)))

        idx_len = len(list(idx2label.keys()))
        assert idx_len == node_count, 'idx_len not equal node_count!'

        print("总节点数：{}".format(idx_len))

        old_label_container = set()
        for each_label, each_index_list in label2idx.items():
            if len(each_index_list) <= cluster_num:
                for each_index in each_index_list:
                    idx2label[each_index] = -1
            else:
                old_label_container.add(each_label)

        old2new = {old_label: new_label for new_label,
                   old_label in enumerate(old_label_container)}

        for each_index, each_label in idx2label.items():
            if each_label == -1:
                continue
            idx2label[each_index] = old2new[each_label]

        pre_labels = intdict2ndarray(idx2label)

        print("总类别数：{} / 类别数：{}".format(keys_len,
              len(set(pre_labels)) - (1 if -1 in pre_labels else 0)))

        return pre_labels


def cluster_by_dbscan(features, k1, k2):
    rerank_dist = compute_jaccard_distance(features, k1=k1, k2=k2)
    tri_mat = np.triu(rerank_dist, 1)
    tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
    tri_mat = np.sort(tri_mat, axis=None)
    rho = 1.6e-3
    top_num = np.round(rho*tri_mat.size).astype(int)
    eps = tri_mat[:top_num].mean()
    cluster_func = DBSCAN(
        eps=eps, min_samples=4, metric='precomputed', n_jobs=-1
    )
    pseudo_labels = cluster_func.fit_predict(rerank_dist)
    return pseudo_labels


def get_dist_nbr(features, k=80, knn_method='faiss-cpu', device=0):
    index = knn_faiss(feats=features, k=k,
                      knn_method=knn_method, device=device)
    knns = index.get_knns()
    dists, nbrs = knns2ordered_nbrs(knns)
    return dists, nbrs


# generate new dataset and calculate cluster centers


@torch.no_grad()
def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)

    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])

    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]

    centers = torch.stack(centers, dim=0)
    return centers


def label_reassign(pseudo_labels, features, centers):
    non_features = features[pseudo_labels == -1].cuda()
    scores = F.normalize(non_features, p=2, dim=1) @ centers.t()
    new_labels = scores.argmax(1)
    pseudo_labels[pseudo_labels == -1] = new_labels.cpu().numpy()


def l2norm(vec):
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


def intdict2ndarray(d, default_val=-1):
    arr = np.zeros(len(d)) + default_val
    for k, v in d.items():
        arr[k] = v
    return arr


def read_meta(fn_meta, start_pos=0, verbose=True):
    """
    idx2lb：每一个顶点对应一个类
    lb2idxs：每个类对应一个id
    """
    lb2idxs = {}
    idx2lb = {}
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[start_pos:]):
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            idx2lb[idx] = lb

    inst_num = len(idx2lb)
    cls_num = len(lb2idxs)
    if verbose:
        print('[{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))
    return lb2idxs, idx2lb


class knn_faiss():
    def __init__(self, feats, k, knn_method='faiss-cpu', device=0, verbose=True):
        self.verbose = verbose

        with Timer('[{}] build index {}'.format(knn_method, k), verbose):
            feats = feats.astype('float32')
            size, dim = feats.shape
            if knn_method == 'faiss-gpu':
                i = math.ceil(size / 1000000)
                if i > 1:
                    i = (i - 1) * 4
                gpu_config = faiss.GpuIndexFlatConfig()
                # gpu_config.device = device
                res = faiss.StandardGpuResources()
                res.setTempMemory(i * 1024 * 1024 * 1024)
                index = faiss.GpuIndexFlatIP(res, dim, gpu_config)
            else:
                index = faiss.IndexFlatIP(dim)
            index.add(feats)

        with Timer('[{}] query topk {}'.format(knn_method, k), verbose):
            sims, nbrs = index.search(feats, k=k)
            self.knns = [(np.array(nbr, dtype=np.int32),
                          1 - np.array(sim, dtype=np.float32))
                         for nbr, sim in zip(nbrs, sims)]

    def filter_by_th(self, i):
        th_nbrs = []
        th_dists = []
        nbrs, dists = self.knns[i]
        for n, dist in zip(nbrs, dists):
            if 1 - dist < self.th:
                continue
            th_nbrs.append(n)
            th_dists.append(dist)
        th_nbrs = np.array(th_nbrs)
        th_dists = np.array(th_dists)
        return th_nbrs, th_dists

    def get_knns(self, th=None):
        if th is None or th <= 0.:
            return self.knns
        # TODO: optimize the filtering process by numpy
        # nproc = mp.cpu_count()
        nproc = 1
        with Timer('filter edges by th {} (CPU={})'.format(th, nproc),
                   self.verbose):
            self.th = th
            self.th_knns = []
            tot = len(self.knns)
            if nproc > 1:
                pool = mp.Pool(nproc)
                th_knns = list(
                    tqdm(pool.imap(self.filter_by_th, range(tot)), total=tot))
                pool.close()
            else:
                th_knns = [self.filter_by_th(i) for i in range(tot)]
            return th_knns


def cluster_by_infomap(nbrs, dists, min_sim, cluster_num=2):
    """
    基于infomap的聚类
    :param nbrs:
    :param dists:
    :param pred_label_path:
    :return:
    """
    single = []
    links = {}
    with Timer('get links', verbose=True):
        single, links = get_links(
            single=single, links=links, nbrs=nbrs, dists=dists, min_sim=min_sim)

    infomapWrapper = infomap.Infomap("--two-level --directed")
    for (i, j), sim in tqdm(links.items()):
        _ = infomapWrapper.addLink(int(i), int(j), sim)

    # 聚类运算
    infomapWrapper.run()

    label2idx = {}
    idx2label = {}

    # 聚类结果统计
    for node in infomapWrapper.iterTree():
        # node.physicalId 特征向量的编号
        # node.moduleIndex() 聚类的编号
        if node.moduleIndex() not in label2idx:
            label2idx[node.moduleIndex()] = []
        label2idx[node.moduleIndex()].append(node.physicalId)

    node_count = 0
    for k, v in label2idx.items():
        if k == 0:
            each_index_list = v[2:]
            node_count += len(each_index_list)
            label2idx[k] = each_index_list
        else:
            each_index_list = v[1:]
            node_count += len(each_index_list)
            label2idx[k] = each_index_list

        for each_index in each_index_list:
            idx2label[each_index] = k

    keys_len = len(list(label2idx.keys()))
    # 孤立点放入到结果中
    for single_node in single:
        idx2label[single_node] = keys_len
        label2idx[keys_len] = [single_node]
        keys_len += 1
        node_count += 1

    # 孤立点个数
    print("孤立点数：{}".format(len(single)))

    idx_len = len(list(idx2label.keys()))
    assert idx_len == node_count, 'idx_len not equal node_count!'

    print("总节点数：{}".format(idx_len))

    old_label_container = set()
    for each_label, each_index_list in label2idx.items():
        if len(each_index_list) <= cluster_num:
            for each_index in each_index_list:
                idx2label[each_index] = -1
        else:
            old_label_container.add(each_label)

    old2new = {old_label: new_label for new_label,
               old_label in enumerate(old_label_container)}

    for each_index, each_label in idx2label.items():
        if each_label == -1:
            continue
        idx2label[each_index] = old2new[each_label]

    pre_labels = intdict2ndarray(idx2label)

    print("总类别数：{}/{}".format(keys_len, len(set(pre_labels)) -
          (1 if -1 in pre_labels else 0)))

    return pre_labels


# generate new dataset and calculate cluster centers

@torch.no_grad()
def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)

    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])
    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]
    centers = torch.stack(centers, dim=0)
    return centers

# generate centroids and formulate -1 samples
@torch.no_grad()
def generate_hybrid_center(labels, features):
    centers = collections.defaultdict(list)
    outliers, outlier_index = [], []

    for idx, label in enumerate(labels):
        if label == -1:
            outliers.append(features[idx])
            outlier_index.append(idx)
        else:
            centers[labels[idx]].append(features[idx])
    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]
    # head--centers, rear--outliers
    centers = torch.stack(centers, dim=0)
    if len(outliers) != 0:
        outliers = torch.stack(outliers, dim=0)
        centers = torch.cat([centers, outliers], dim=0)
    index2pid = {
        index: ind + labels.max() + 1 for (ind, index) in enumerate(outlier_index)
    }
    return centers, index2pid

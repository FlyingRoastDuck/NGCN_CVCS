import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
import numpy as np
from project_utils.finch import FINCH
from project_utils.infomap_cluster_utils import cluster_by_semi_infomap
import scipy.sparse as sp
from sklearn import metrics


def train_gcn(gcn_e, gcn_e_dataset, epochs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gcn_e = gcn_e.to(device)
    gcn_e.eval()
    # constrauct optimizer
    optimizer_gcn_e = SGD(
        gcn_e.parameters(), lr=0.01, weight_decay=1e-5, momentum=0.9
    )
    lr_sche_edge = MultiStepLR(
        optimizer_gcn_e, milestones=[20, ], gamma=0.5
    )
    acc_list = []
    # start training
    for epoch in range(epochs):
        loss_e, loss_v = [], []
        for idx, data_e in enumerate(gcn_e_dataset):
            # handle data
            sub_g_features, sub_g_adj, sub_g_lab = data_e # data for graph certainty
            sub_g_features, sub_g_adj, sub_g_lab = torch.from_numpy(sub_g_features).to(device),\
                torch.from_numpy(sub_g_adj.toarray()).to(device).float(), torch.from_numpy(sub_g_lab).to(device)

            # sampling sub_g
            link_estimate = gcn_e([sub_g_features, sub_g_adj])
            loss_gcn_e = F.cross_entropy(link_estimate, sub_g_lab)
            # link_estimate[:, 1] > link_estimate[:, 0] -- link it
            acc_list.extend(
                ((link_estimate[:, 1] > link_estimate[:, 0]).squeeze().float() == sub_g_lab.float()).cpu().numpy()
            )
            # optimize
            optimizer_gcn_e.zero_grad()
            loss_gcn_e.backward()
            optimizer_gcn_e.step()
            loss_e.append(loss_gcn_e.item())
        # train gcn_e
        if len(acc_list)!=0:
            acc = sum(acc_list) / len(acc_list)
            print(f"Epoch: {epoch}. gcn_e loss:{sum(loss_e)/len(loss_e):4.2f}, "
                f"Pred Acc: {acc*100: 4.2f}%")
        lr_sche_edge.step(epoch)
    return gcn_e


def test_gcn_e(
    gcn_e, gcn_e_dataset, max_conn=1, 
    target_pids=None, if_labelled=None, train_pid_count=None,
    eexit=False
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gcn_e = gcn_e.to(device)
    gcn_e = gcn_e.eval()
    pred_conns = []
    
    # confidence for each sample, and their dists
    pred_peaks = gcn_e_dataset.peaks
    pred_dist2peak = gcn_e_dataset.dist2peak
        
    # generate confidence for each node
    acc_list = []
    for i, data in enumerate(gcn_e_dataset):
        with torch.no_grad():
            sub_feat, sub_adj, real_pids = data
            sub_feat, sub_adj = torch.from_numpy(sub_feat).to(device),\
                torch.from_numpy(sub_adj.toarray()).to(device)
            # support features and support set
            output = gcn_e((sub_feat, sub_adj)).squeeze()
            output = F.softmax(output, 1)[:, 1].detach().cpu().numpy()
            pred_ind = (-output).argpartition(max_conn)[:max_conn]
            pred_pos = output[pred_ind] > 0.5
            # filter out low confi samples
            pred_ind = pred_ind[pred_pos]
                
        if len(pred_ind) > 0:
            pred_conns.append(pred_ind) 
            if -1 not in real_pids:
                for ind in range(len(pred_ind)): 
                    acc_list.append(real_pids[pred_ind[ind]]==1)
        else:
            pred_conns.append([])
        
    if -1 not in real_pids and len(acc_list)!=0:
        acc = sum(acc_list) / len(acc_list)
        print(f"Linkage Acc: {acc*100:4.2f}%")
        
    # final prediction on nbrs for chosen samples
    for pred_rel_nbr, nbr, dist, idx in zip(
        pred_conns, gcn_e_dataset.subset_nbrs, 
        gcn_e_dataset.subset_dists, gcn_e_dataset.subset_idxs
    ):
        # no neighbor, pass
        if len(pred_rel_nbr)==0: continue
        # samples need to be predicted
        pred_abs_nbr = nbr[pred_rel_nbr]
        # add predicted to chose pred sample
        pred_peaks[idx].extend(pred_abs_nbr)
        pred_dist2peak[idx].extend(dist[pred_rel_nbr])
        
    row, col, dis = [], [], []
    for idx, key in enumerate(pred_peaks):
        nbr, distance = pred_peaks[key], pred_dist2peak[key]
        if isinstance(nbr, list) and len(nbr)!=0:
            row.extend([idx,]*len(nbr))
            col.extend(nbr)
            dis.extend(distance)
        elif not isinstance(nbr, list):
            row.append(idx)
            col.append(nbr)
            dis.append(distance)
    vals = [1, ]*len(row)
    
    adj = sp.csr_matrix(
        (
            vals, (row, col)
        ), shape=(len(pred_peaks), len(pred_peaks))
    )
    ori_dist = sp.csr_matrix(
        (
            dis, (row, col)
        ), shape=(len(pred_peaks), len(pred_peaks))
    )
    pse_labels = FINCH(gcn_e_dataset.features, adj, ori_dist, ensure_early_exit=eexit)
    
    # using target pids
    if target_pids is not None:
        delta = abs(target_pids-pse_labels.max(0))
        chosen = np.argmin(delta)
    else:
        # using training pids
        delta = abs(train_pid_count-pse_labels[if_labelled, :].max(0))
        chosen = np.argmin(delta)
    
    predicted = pse_labels[:, chosen]
    print(f'NMI Score: {metrics.normalized_mutual_info_score(predicted, gcn_e_dataset.labels.numpy()): 4.3f}')
    return predicted

def test_gcn_infomap(
    gcn_e, gcn_e_dataset, max_conn=1, 
    target_pids=None, if_labelled=None, train_pid_count=None,
    eexit=False
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gcn_e = gcn_e.to(device)
    gcn_e = gcn_e.eval()
    pred_conns = []
    
    # confidence for each sample, and their dists
    pred_peaks = gcn_e_dataset.peaks
    pred_dist2peak = gcn_e_dataset.dist2peak
        
    # generate confidence for each node
    acc_list = []
    for i, data in enumerate(gcn_e_dataset):
        with torch.no_grad():
            sub_feat, sub_adj, real_pids = data
            sub_feat, sub_adj = torch.from_numpy(sub_feat).to(device),\
                torch.from_numpy(sub_adj.toarray()).to(device)
            # support features and support set
            output = gcn_e((sub_feat, sub_adj)).squeeze()
            output = F.softmax(output, 1)[:, 1].detach().cpu().numpy()
            pred_ind = (-output).argpartition(max_conn)[:max_conn]
            pred_pos = output[pred_ind] > 0.5
            # filter out low confi samples
            pred_ind = pred_ind[pred_pos]
                
        if len(pred_ind) > 0:
            pred_conns.append(pred_ind) 
            if -1 not in real_pids:
                for ind in range(len(pred_ind)): 
                    acc_list.append(real_pids[pred_ind[ind]]==1)
        else:
            pred_conns.append([])
        
    if -1 not in real_pids and len(acc_list)!=0:
        acc = sum(acc_list) / len(acc_list)
        print(f"Linkage Acc: {acc*100:4.2f}%")
        
    # final prediction on nbrs for chosen samples
    for pred_rel_nbr, nbr, dist, idx in zip(
        pred_conns, gcn_e_dataset.subset_nbrs, 
        gcn_e_dataset.subset_dists, gcn_e_dataset.subset_idxs
    ):
        # no neighbor, pass
        if len(pred_rel_nbr)==0: continue
        # samples need to be predicted
        pred_abs_nbr = nbr[pred_rel_nbr]
        # add predicted to chose pred sample
        pred_peaks[idx].extend(pred_abs_nbr)
        pred_dist2peak[idx].extend(dist[pred_rel_nbr])
        
    row, col, dis = [], [], []
    for idx, key in enumerate(pred_peaks):
        nbr, distance = pred_peaks[key], pred_dist2peak[key]
        if isinstance(nbr, list) and len(nbr)!=0:
            row.extend([idx,]*len(nbr))
            col.extend(nbr)
            dis.extend(distance)
        elif not isinstance(nbr, list):
            row.append(idx)
            col.append(nbr)
            dis.append(distance)
    vals = [1, ]*len(row)
    
    adj = sp.csr_matrix(
        (
            vals, (row, col)
        ), shape=(len(pred_peaks), len(pred_peaks))
    )
    ori_dist = sp.csr_matrix(
        (
            dis, (row, col)
        ), shape=(len(pred_peaks), len(pred_peaks))
    )
    pse_labels = cluster_by_semi_infomap(
        adj, ori_dist, 0.4, cluster_num=2,
        label_mark=None, if_labeled=None, args=None
    )
    return pse_labels

def test_1nn_e(
    features,  
    target_pids=None, if_labelled=None, train_pid_count=None, eexit=False
):
    sim = features @ features.T
    pred_conns = sim.argmax(1)
    row, col, dis = [], [], []
    for idx, key in enumerate(pred_conns):
        row.append(idx)
        col.append(key)
        dis.append(sim[idx][key])
    vals = [1, ]*len(row)
    
    adj = sp.csr_matrix(
        (
            vals, (row, col)
        ), shape=(len(features), len(features))
    )
    ori_dist = sp.csr_matrix(
        (
            dis, (row, col)
        ), shape=(len(features), len(features))
    )
    pse_labels = FINCH(features, adj, ori_dist, ensure_early_exit=eexit)
    # using target pids
    if target_pids is not None:
        delta = abs(target_pids-pse_labels.max(0))
        chosen = np.argmin(delta)
    else:
        # using training pids
        delta = abs(train_pid_count-pse_labels[if_labelled, :].max(0))
        chosen = np.argmin(delta)
        
    return pse_labels[:, chosen]

import numpy as np
from sklearn.cluster import KMeans
import torch
from project_utils.cluster_and_log_utils import log_accs_from_preds

from methods.clustering.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
from methods.estimate_k.estimate_k import scipy_optimise, binary_search

from sklearn.utils._joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils import check_random_state
from tqdm import tqdm

# TODO: Debug
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch.nn as nn
import time
import torch.nn.functional as F


def pairwise_distance(data1, data2, batch_size=None):
    #N*1*M
    A = data1.unsqueeze(dim=1)

    #1*N*M
    B = data2.unsqueeze(dim=0)

    if batch_size == None:
        dis = (A-B)**2
        #return N*N matrix for pairwise distance
        dis = dis.sum(dim=-1)
    else:
        i = 0
        dis = torch.zeros(data1.shape[0], data2.shape[0])
        while i < data1.shape[0]:
            if(i+batch_size < data1.shape[0]):
                dis_batch = (A[i:i+batch_size]-B)**2
                dis_batch = dis_batch.sum(dim=-1)
                dis[i:i+batch_size] = dis_batch
                i = i+batch_size
            elif(i+batch_size >= data1.shape[0]):
                dis_final = (A[i:] - B)**2
                dis_final = dis_final.sum(dim=-1)
                dis[i:] = dis_final
                break
    return dis

@torch.no_grad()
def test_kmeans_bs(
    model, test_loader, epoch, save_name, device, 
    args, logger_class=None, in_training=False, max_classes=250, min_classes=None
):
    K = binary_search(test_loader, args, max_classes, model, min_classes)
    logger_class(f"The best K for testing {K}")
    model.eval()
    all_feats = []
    targets = np.array([])
    mask_lab = np.array([])  # From all the data, which instances belong to the labelled set
    mask_cls = np.array([])  # From all the data, which instances belong to Old classes

    # First extract all features
    for batch_idx, (images, label, _, mask_lab_, attribute) in enumerate(test_loader):
        images = images.to(device)
        with torch.no_grad():
            feats = model(images)
        if args.use_l2_in_ssk:
            feats = torch.nn.functional.normalize(feats, dim=-1)
        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes))
                                                    else False for x in label]))
        mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())

    mask_lab = mask_lab.astype(bool)
    mask_cls = mask_cls.astype(bool)

    all_feats = np.concatenate(all_feats)

    l_feats = all_feats[mask_lab]  # Get labelled set
    u_feats = all_feats[~mask_lab]  # Get unlabelled set
    l_targets = targets[mask_lab]  # Get labelled targets
    u_targets = targets[~mask_lab]  # Get unlabelled targets
    kmeans = SemiSupKMeans(k=K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, 
                           init='k-means++', n_init=args.k_means_init, random_state=None, 
                           n_jobs=None, pairwise_batch_size=512, mode=None)
    l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for x in (l_feats, u_feats, l_targets, u_targets))
    kmeans.fit_mix(u_feats, l_feats, l_targets)
    all_preds = kmeans.labels_.cpu().numpy()
    u_targets = u_targets.cpu().numpy()
    preds = all_preds[~mask_lab]

    # Get portion of mask_cls which corresponds to the unlabelled set
    mask = mask_cls[~mask_lab]
    mask = mask.astype(bool)
    all_acc, old_acc, new_acc = log_accs_from_preds(
        y_true=u_targets, y_pred=preds, mask=mask, 
        eval_funcs=args.eval_funcs, save_name=save_name, T=epoch, print_output=True
    )

    if args.use_contrastive_cluster and args.contrastive_cluster_method == 'ssk':
        return all_acc, old_acc, new_acc, kmeans, all_feats
    else:
        return all_acc, old_acc, new_acc, kmeans, None

@torch.no_grad()
def test_kmeans_scipy(
    model, test_loader, epoch, save_name, device, 
    args, logger_class=None, in_training=False, max_classes=100
):
    """
    In this case, the test loader needs to have the labelled and unlabelled subsets of the training data
    """
    K = scipy_optimise(test_loader, args, max_classes, model)
    logger_class(f"The best K for testing {K}")
    
    if isinstance(model, (list, tuple, nn.ModuleList, nn.ModuleDict)) and len(model) >= 2:
        co_feat_extractor, att_feat_extractor = model[0], model[1]
        co_feat_extractor.eval()
        att_feat_extractor.eval()

        all_feats, all_co_feats = [], []
        targets = np.array([])
        mask_lab = np.array([])  # From all the data, which instances belong to the labelled set
        mask_cls = np.array([])  # From all the data, which instances belong to Old classes

        print('Collating features...')
        # First extract all features
        for batch_idx, (images, label, _, mask_lab_, attribute) in enumerate(test_loader):
            images = images.to(device)

            co_feats, att_embs = co_feat_extractor(images)
            att_feats = att_feat_extractor(att_embs)
            if args.use_l2_in_ssk:
                co_feats = torch.nn.functional.normalize(co_feats, dim=-1)
                att_feats = torch.nn.functional.normalize(att_feats, dim=-1)

            feats = torch.cat((co_feats, att_feats), dim=1)
            all_co_feats.append(co_feats.cpu().numpy())
            all_feats.append(feats.cpu().numpy())
            targets = np.append(targets, label.cpu().numpy())
            mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes))
                                                     else False for x in label]))
            mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())

        # -----------------------
        # K-MEANS
        # -----------------------
        mask_lab = mask_lab.astype(bool)
        mask_cls = mask_cls.astype(bool)

        all_feats = np.concatenate(all_feats)
        all_co_feats = np.concatenate(all_co_feats)

        l_feats = all_feats[mask_lab]  # Get labelled set
        u_feats = all_feats[~mask_lab]  # Get unlabelled set
        l_co_feats = all_co_feats[mask_lab]  # Get labelled set
        u_co_feats = all_co_feats[~mask_lab]  # Get unlabelled set
        l_targets = targets[mask_lab]  # Get labelled targets
        u_targets = targets[~mask_lab]  # Get unlabelled targets

        print('Fitting Semi-Supervised K-Means with concatenated_feature...')
        if in_training:
            max_kmeans_iter = args.train_max_kmeans_iter
        else:
            max_kmeans_iter = args.max_kmeans_iter
        logger_class('max_kmeans_iter: {max_kmeans_iter}!')
        # not based on 
        kmeans = SemiSupKMeans(
            k=K, tolerance=1e-4, max_iterations=max_kmeans_iter, 
            init='k-means++', n_init=args.k_means_init, random_state=None, 
            n_jobs=None, pairwise_batch_size=512, mode=None
        )

        l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                                  x in (l_feats, u_feats, l_targets, u_targets))

        kmeans.fit_mix(u_feats, l_feats, l_targets)
        all_preds = kmeans.labels_.cpu().numpy()
        u_targets = u_targets.cpu().numpy()

        # -----------------------
        # EVALUATE
        # -----------------------
        # Get preds corresponding to unlabelled set
        preds = all_preds[~mask_lab]

        # Get portion of mask_cls which corresponds to the unlabelled set
        mask = mask_cls[~mask_lab]
        mask = mask.astype(bool)

        # -----------------------
        # EVALUATE
        # -----------------------
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets, y_pred=preds, mask=mask,
                                                        eval_funcs=args.eval_funcs,
                                                        save_name=save_name, T=epoch, print_output=True)
        logger_class(
            'Using concatenated_feature ==> SS-K Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc,
                                                                                                          old_acc,
                                                                                                          new_acc))

        print('Using contrastive_feature Fitting Semi-Supervised K-Means...')
        kmeans = SemiSupKMeans(k=K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
                               n_init=args.k_means_init, random_state=None, n_jobs=None, pairwise_batch_size=512,
                               mode=None)

        l_c0_feats, u_co_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                                        x in (l_co_feats, u_co_feats, l_targets, u_targets))

        kmeans.fit_mix(u_co_feats, l_c0_feats, l_targets)
        all_preds = kmeans.labels_.cpu().numpy()
        u_targets = u_targets.cpu().numpy()

        # -----------------------
        # EVALUATE
        # -----------------------
        # Get preds corresponding to unlabelled set
        preds = all_preds[~mask_lab]

        # Get portion of mask_cls which corresponds to the unlabelled set
        mask = mask_cls[~mask_lab]
        mask = mask.astype(bool)

        # -----------------------
        # EVALUATE
        # -----------------------
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets, y_pred=preds, mask=mask,
                                                        eval_funcs=args.eval_funcs,
                                                        save_name=save_name, T=epoch, print_output=True)

        return all_acc, old_acc, new_acc, kmeans
    else:
        model.eval()
        all_feats = []
        targets = np.array([])
        mask_lab = np.array([])  # From all the data, which instances belong to the labelled set
        mask_cls = np.array([])  # From all the data, which instances belong to Old classes

        print('Collating features...')
        # First extract all features
        for batch_idx, (images, label, _, mask_lab_, attribute) in enumerate(test_loader):
            images = images.to(device)
            with torch.no_grad():
                feats = model(images)
            if args.use_l2_in_ssk:
                feats = torch.nn.functional.normalize(feats, dim=-1)
            all_feats.append(feats.cpu().numpy())
            targets = np.append(targets, label.cpu().numpy())
            mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes))
                                                     else False for x in label]))
            mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())

        # -----------------------
        # K-MEANS
        # -----------------------
        mask_lab = mask_lab.astype(bool)
        mask_cls = mask_cls.astype(bool)

        all_feats = np.concatenate(all_feats)

        l_feats = all_feats[mask_lab]  # Get labelled set
        u_feats = all_feats[~mask_lab]  # Get unlabelled set
        l_targets = targets[mask_lab]  # Get labelled targets
        u_targets = targets[~mask_lab]  # Get unlabelled targets
        if in_training:
            max_kmeans_iter = args.train_max_kmeans_iter
        else:
            max_kmeans_iter = args.max_kmeans_iter
        logger_class(f'Using estimated K for K-Means... max_kmeans_iter = {max_kmeans_iter}')
        kmeans = SemiSupKMeans(k=K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
                               n_init=args.k_means_init, random_state=None, n_jobs=None, pairwise_batch_size=512,
                               mode=None)

        l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                                  x in (l_feats, u_feats, l_targets, u_targets))

        kmeans.fit_mix(u_feats, l_feats, l_targets)
        all_preds = kmeans.labels_.cpu().numpy()
        u_targets = u_targets.cpu().numpy()

        # -----------------------
        # EVALUATE
        # -----------------------
        # Get preds corresponding to unlabelled set
        preds = all_preds[~mask_lab]

        # Get portion of mask_cls which corresponds to the unlabelled set
        mask = mask_cls[~mask_lab]
        mask = mask.astype(bool)
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets, y_pred=preds, mask=mask,
                                                        eval_funcs=args.eval_funcs,
                                                        save_name=save_name, T=epoch, print_output=True)

        if args.use_contrastive_cluster and args.contrastive_cluster_method == 'ssk':
            return all_acc, old_acc, new_acc, kmeans, all_feats
        else:
            return all_acc, old_acc, new_acc, kmeans, None

@torch.no_grad()
def test_kmeans_semi_sup(model, test_loader, epoch, save_name, 
                         device, args, K=None, logger_class=None, 
                         in_training=False):
    if K is None:
        K = args.num_labeled_classes + args.num_unlabeled_classes

    if isinstance(model, (list, tuple, nn.ModuleList, nn.ModuleDict)) and len(model) >= 2:
        co_feat_extractor, att_feat_extractor = model[0], model[1]
        co_feat_extractor.eval()
        att_feat_extractor.eval()

        all_feats, all_co_feats = [], []
        targets = np.array([])
        mask_lab = np.array([])  # From all the data, which instances belong to the labelled set
        mask_cls = np.array([])  # From all the data, which instances belong to Old classes

        print('Collating features...')
        # First extract all features
        for batch_idx, (images, label, _, mask_lab_, attribute) in enumerate(test_loader):
            images = images.to(device)

            co_feats, att_embs = co_feat_extractor(images)
            att_feats = att_feat_extractor(att_embs)
            if args.use_l2_in_ssk:
                co_feats = torch.nn.functional.normalize(co_feats, dim=-1)
                att_feats = torch.nn.functional.normalize(att_feats, dim=-1)

            feats = torch.cat((co_feats, att_feats), dim=1)
            all_co_feats.append(co_feats.cpu().numpy())
            all_feats.append(feats.cpu().numpy())
            targets = np.append(targets, label.cpu().numpy())
            mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes))
                                                     else False for x in label]))
            mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())

        # -----------------------
        # K-MEANS
        # -----------------------
        mask_lab = mask_lab.astype(bool)
        mask_cls = mask_cls.astype(bool)

        all_feats = np.concatenate(all_feats)
        all_co_feats = np.concatenate(all_co_feats)

        l_feats = all_feats[mask_lab]  # Get labelled set
        u_feats = all_feats[~mask_lab]  # Get unlabelled set
        l_co_feats = all_co_feats[mask_lab]  # Get labelled set
        u_co_feats = all_co_feats[~mask_lab]  # Get unlabelled set
        l_targets = targets[mask_lab]  # Get labelled targets
        u_targets = targets[~mask_lab]  # Get unlabelled targets

        print('Fitting Semi-Supervised K-Means with concatenated_feature...')
        if in_training:
            max_kmeans_iter = args.train_max_kmeans_iter
        else:
            max_kmeans_iter = args.max_kmeans_iter
        logger_class('max_kmeans_iter: {max_kmeans_iter}!')
        # not based on 
        kmeans = SemiSupKMeans(
            k=K, tolerance=1e-4, max_iterations=max_kmeans_iter, 
            init='k-means++', n_init=args.k_means_init, random_state=None, 
            n_jobs=None, pairwise_batch_size=512, mode=None
        )

        l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                                  x in (l_feats, u_feats, l_targets, u_targets))

        kmeans.fit_mix(u_feats, l_feats, l_targets)
        all_preds = kmeans.labels_.cpu().numpy()
        u_targets = u_targets.cpu().numpy()

        # -----------------------
        # EVALUATE
        # -----------------------
        # Get preds corresponding to unlabelled set
        preds = all_preds[~mask_lab]

        # Get portion of mask_cls which corresponds to the unlabelled set
        mask = mask_cls[~mask_lab]
        mask = mask.astype(bool)

        # -----------------------
        # EVALUATE
        # -----------------------
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets, y_pred=preds, mask=mask,
                                                        eval_funcs=args.eval_funcs,
                                                        save_name=save_name, T=epoch, print_output=True)
        logger_class(
            'Using concatenated_feature ==> SS-K Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc,
                                                                                                          old_acc,
                                                                                                          new_acc))

        print('Using contrastive_feature Fitting Semi-Supervised K-Means...')
        kmeans = SemiSupKMeans(k=K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
                               n_init=args.k_means_init, random_state=None, n_jobs=None, pairwise_batch_size=512,
                               mode=None)

        l_c0_feats, u_co_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                                        x in (l_co_feats, u_co_feats, l_targets, u_targets))

        kmeans.fit_mix(u_co_feats, l_c0_feats, l_targets)
        all_preds = kmeans.labels_.cpu().numpy()
        u_targets = u_targets.cpu().numpy()

        # -----------------------
        # EVALUATE
        # -----------------------
        # Get preds corresponding to unlabelled set
        preds = all_preds[~mask_lab]

        # Get portion of mask_cls which corresponds to the unlabelled set
        mask = mask_cls[~mask_lab]
        mask = mask.astype(bool)

        # -----------------------
        # EVALUATE
        # -----------------------
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets, y_pred=preds, mask=mask,
                                                        eval_funcs=args.eval_funcs,
                                                        save_name=save_name, T=epoch, print_output=True)

        return all_acc, old_acc, new_acc, kmeans
    else:
        model.eval()
        all_feats = []
        targets = np.array([])
        mask_lab = np.array([])  # From all the data, which instances belong to the labelled set
        mask_cls = np.array([])  # From all the data, which instances belong to Old classes

        print('Collating features...')
        # First extract all features
        for batch_idx, (images, label, _, mask_lab_, attribute) in enumerate(test_loader):
            images = images.to(device)
            feats = model(images)
            if args.use_l2_in_ssk:
                feats = torch.nn.functional.normalize(feats, dim=-1)
            all_feats.append(feats.cpu().numpy())
            targets = np.append(targets, label.cpu().numpy())
            mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes))
                                                     else False for x in label]))
            mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())

        # -----------------------
        # K-MEANS
        # -----------------------
        mask_lab = mask_lab.astype(bool)
        mask_cls = mask_cls.astype(bool)

        all_feats = np.concatenate(all_feats)

        l_feats = all_feats[mask_lab]  # Get labelled set
        u_feats = all_feats[~mask_lab]  # Get unlabelled set
        l_targets = targets[mask_lab]  # Get labelled targets
        u_targets = targets[~mask_lab]  # Get unlabelled targets

        if in_training:
            max_kmeans_iter = args.train_max_kmeans_iter
        else:
            max_kmeans_iter = args.max_kmeans_iter
        kmeans = SemiSupKMeans(k=K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
                               n_init=args.k_means_init, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                               mode=None)

        l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                                  x in (l_feats, u_feats, l_targets, u_targets))

        kmeans.fit_mix(u_feats, l_feats, l_targets)
        all_preds = kmeans.labels_.cpu().numpy()
        u_targets = u_targets.cpu().numpy()

        # -----------------------
        # EVALUATE
        # -----------------------
        # Get preds corresponding to unlabelled set
        preds = all_preds[~mask_lab]

        # Get portion of mask_cls which corresponds to the unlabelled set
        mask = mask_cls[~mask_lab]
        mask = mask.astype(bool)

        # -----------------------
        # EVALUATE
        # -----------------------
        all_acc, old_acc, new_acc = log_accs_from_preds(
            y_true=u_targets, y_pred=preds, mask=mask, 
            eval_funcs=args.eval_funcs, save_name=save_name, 
            T=epoch, print_output=True
        )

        if args.use_contrastive_cluster and args.contrastive_cluster_method == 'ssk':
            return all_acc, old_acc, new_acc, kmeans, all_feats
        else:
            return all_acc, old_acc, new_acc, kmeans, None
        
class K_Means:

    def __init__(self, k=3, tolerance=1e-4, max_iterations=100, init='k-means++',
                 n_init=10, random_state=None, n_jobs=None, pairwise_batch_size=None, mode=None):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.pairwise_batch_size = pairwise_batch_size
        self.mode = mode

    def split_for_val(self, l_feats, l_targets, val_prop=0.2):

        np.random.seed(0)

        # Reserve some labelled examples for validation
        num_val_instances = int(val_prop * len(l_targets))
        val_idxs = np.random.choice(range(len(l_targets)), size=(num_val_instances), replace=False)
        val_idxs.sort()
        remaining_idxs = list(set(range(len(l_targets))) - set(val_idxs.tolist()))
        remaining_idxs.sort()
        remaining_idxs = np.array(remaining_idxs)

        val_l_targets = l_targets[val_idxs]
        val_l_feats = l_feats[val_idxs]

        remaining_l_targets = l_targets[remaining_idxs]
        remaining_l_feats = l_feats[remaining_idxs]

        return remaining_l_feats, remaining_l_targets, val_l_feats, val_l_targets
    
    @torch.no_grad()
    def kpp(self, X, pre_centers=None, k=10, random_state=None):
        random_state = check_random_state(random_state)

        if pre_centers is not None:

            C = pre_centers

        else:

            C = X[random_state.randint(0, len(X))]

        C = C.view(-1, X.shape[1])

        while C.shape[0] < k:

            dist = pairwise_distance(X, C, self.pairwise_batch_size)
            dist = dist.view(-1, C.shape[0])
            d2, _ = torch.min(dist, dim=1)
            prob = d2/d2.sum()
            cum_prob = torch.cumsum(prob, dim=0)
            r = random_state.rand()

            if len((cum_prob >= r).nonzero()) == 0:
                debug = 0
            else:
                ind = (cum_prob >= r).nonzero()[0][0]
            C = torch.cat((C, X[ind].view(1, -1)), dim=0)

        return C
    
    @torch.no_grad()
    def fit_once(self, X, random_state):
        
        centers = torch.zeros(self.k, X.shape[1]).type_as(X)
        labels = -torch.ones(len(X))
        #initialize the centers, the first 'k' elements in the dataset will be our initial centers

        if self.init == 'k-means++':
            centers = self.kpp(X, k=self.k, random_state=random_state)

        elif self.init == 'random':

            random_state = check_random_state(self.random_state)
            idx = random_state.choice(len(X), self.k, replace=False)
            for i in range(self.k):
                centers[i] = X[idx[i]]

        else:
            for i in range(self.k):
                centers[i] = X[i]
                
        best_labels, best_inertia, best_centers = None, None, None
        for i in range(self.max_iterations):
            centers_old = centers.clone()
            dist = pairwise_distance(X, centers, self.pairwise_batch_size)
            mindist, labels = torch.min(dist, dim=1)
            inertia = mindist.sum()

            for idx in range(self.k):
                selected_ind = torch.nonzero(labels == idx).squeeze().to(X.device)
                selected = torch.index_select(X, 0, selected_ind)
                centers[idx] = selected.mean(dim=0)

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.clone()
                best_centers = centers.clone()
                best_inertia = inertia
                best_selected_ind = selected_ind.clone()

            center_shift = torch.sum(torch.sqrt(torch.sum((centers - centers_old) ** 2, dim=1)))
            if center_shift ** 2 < self.tolerance:
                #break out of the main loop if the results are optimal, ie. the centers don't change their positions much(more than our tolerance)
                break

        return best_labels, best_inertia, best_centers, i + 1
    
    def fit_mix_once(self, u_feats, l_feats, l_targets, random_state):

        def supp_idxs(c):
            return l_targets.eq(c).nonzero().squeeze(1)

        l_classes = torch.unique(l_targets)
        support_idxs = list(map(supp_idxs, l_classes))
        l_centers = torch.stack([l_feats[idx_list].mean(0) for idx_list in support_idxs])
        cat_feats = torch.cat((l_feats, u_feats))

        centers = torch.zeros([self.k, cat_feats.shape[1]]).type_as(cat_feats)
        centers[:len(l_classes)] = l_centers

        labels = -torch.ones(len(cat_feats)).type_as(cat_feats).long()

        l_classes = l_classes.cpu().long().numpy()
        l_targets = l_targets.cpu().long().numpy()
        l_num = len(l_targets)
        cid2ncid = {cid:ncid for ncid, cid in enumerate(l_classes)}  # Create the mapping table for New cid (ncid)
        for i in range(l_num):
            labels[i] = cid2ncid[l_targets[i]]

        #initialize the centers, the first 'k' elements in the dataset will be our initial centers
        centers = self.kpp(u_feats, l_centers, k=self.k, random_state=random_state)

        # Begin iterations
        best_labels, best_inertia, best_centers = None, None, None
        for it in range(self.max_iterations):
            centers_old = centers.clone()

            dist = pairwise_distance(u_feats, centers, self.pairwise_batch_size)
            u_mindist, u_labels = torch.min(dist, dim=1)
            u_inertia = u_mindist.sum()
            l_mindist = torch.sum((l_feats - centers[labels[:l_num]])**2, dim=1)
            l_inertia = l_mindist.sum()
            inertia = u_inertia + l_inertia
            labels[l_num:] = u_labels

            for idx in range(self.k):
                
                selected = torch.nonzero(labels == idx).squeeze()
                selected = torch.index_select(cat_feats, 0, selected)
                centers[idx] = selected.mean(dim=0)

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.clone()
                best_centers = centers.clone()
                best_inertia = inertia

            center_shift = torch.sum(torch.sqrt(torch.sum((centers - centers_old) ** 2, dim=1)))

            if center_shift ** 2 < self.tolerance:
                #break out of the main loop if the results are optimal, ie. the centers don't change their positions much(more than our tolerance)
                break

        return best_labels, best_inertia, best_centers, i + 1
    
    @torch.no_grad()
    def fit(self, X):
        random_state = check_random_state(self.random_state)
        best_inertia = None
        if effective_n_jobs(self.n_jobs) == 1:
            for it in range(self.n_init):
                labels, inertia, centers, n_iters = self.fit_once(X, random_state)
                if best_inertia is None or inertia < best_inertia:
                    self.labels_ = labels.clone()
                    self.cluster_centers_ = centers.clone()
                    best_inertia = inertia
                    self.inertia_ = inertia
                    self.n_iter_ = n_iters
        else:
            # parallelisation of k-means runs
            seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(delayed(self.fit_once)(X, seed) for seed in seeds)
            # Get results with the lowest inertia
            labels, inertia, centers, n_iters = zip(*results)
            best = np.argmin(inertia)
            self.labels_ = labels[best]
            self.inertia_ = inertia[best]
            self.cluster_centers_ = centers[best]
            self.n_iter_ = n_iters[best]
            
    def fit_mix(self, u_feats, l_feats, l_targets):

        random_state = check_random_state(self.random_state)
        best_inertia = None
        fit_func = self.fit_mix_once

        if effective_n_jobs(self.n_jobs) == 1:
            for it in range(self.n_init):

                labels, inertia, centers, n_iters = fit_func(u_feats, l_feats, l_targets, random_state)

                if best_inertia is None or inertia < best_inertia:
                    self.labels_ = labels.clone()
                    self.cluster_centers_ = centers.clone()
                    best_inertia = inertia
                    self.inertia_ = inertia
                    self.n_iter_ = n_iters

        else:
            seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(delayed(fit_func)(u_feats, l_feats, l_targets, seed)
                                                              for seed in seeds)
            # Get results with the lowest inertia

            labels, inertia, centers, n_iters = zip(*results)
            best = np.argmin(inertia)
            self.labels_ = labels[best]
            self.inertia_ = inertia[best]
            self.cluster_centers_ = centers[best]
            self.n_iter_ = n_iters[best]

def forward(x, model, projection_head=None, predict_token='cls', 
            return_z_features=False, num_prompts=5, mode='train', 
            num_cop=2, **kwargs):
    if predict_token=='cls':
        features = model(x)
        if projection_head is not None:
            z_features = features
            features = projection_head(z_features)
        features = F.normalize(features, dim=-1)
        if return_z_features is True:
            z_features = F.normalize(z_features, dim=-1)
            return features, z_features
        return features
    elif predict_token == 'cls-vptm':
        assert return_z_features==False, f'not implemented error'
        assert 'aux_projection_head' in kwargs.keys()
        features = model(x, True)
        shape_feature = features[:, 1:1+num_cop, :].size()
        features = [features[:, 0, :], features[:, 1:1+num_cop, :].view(shape_feature[0], -1, shape_feature[-1]).mean(dim=1)]
        if projection_head is not None: # training mode
            features[0] = projection_head(features[0])
            features[1] = kwargs['aux_projection_head'](features[1])
            features[0] = F.normalize(features[0], dim=-1)
            features[1] = F.normalize(features[1], dim=-1)
        feat = features[0]
        aux_feat = features[1]

        if projection_head is None:
            feat = F.normalize(feat, dim=-1)
            aux_feat = F.normalize(aux_feat, dim=-1)
        if mode=='train':
            return feat, aux_feat
        elif mode=='test':
            return feat
    elif predict_token in ['cop']:
        if mode=='train':
            features = model(x, True)
            features = [features[:, 0, :], features[:, 1:1+num_prompts, :]]
            
            z_features = features[0]
            z_features = F.normalize(z_features, dim=-1)
            
            features[0] = projection_head(features[0])
            features[0] = F.normalize(features[0], dim=-1)
            
            prompt_features = features[1][:, :num_cop, :].mean(dim=1)
            z_prompt_features = F.normalize(prompt_features, dim=-1)
            prompt_features = kwargs['aux_projection_head'](prompt_features)
            prompt_features = F.normalize(prompt_features, dim=-1)
            if return_z_features:
                features = [features[0], z_features, prompt_features, z_prompt_features]
            else:
                features = [features[0], prompt_features]
            return features
        elif mode=='test':
            with torch.no_grad():
                features = model(x)
                features = F.normalize(features, dim=-1)
            return features
        elif mode=='teacher':
            with torch.no_grad():
                features = model(x, True)
                features = [features[:, 0, :], features[:, 1:1+num_prompts, :]]
                
                z_features = features[0]
                z_features = F.normalize(z_features, dim=-1)
                
                features[0] = projection_head(features[0])
                features[0] = F.normalize(features[0], dim=-1)
                
                prompt_features = features[1][:, :num_cop, :].mean(dim=1)
                z_prompt_features = F.normalize(prompt_features, dim=-1)
                prompt_features = kwargs['aux_projection_head_t'](prompt_features)
                prompt_features = F.normalize(prompt_features, dim=-1)
            if return_z_features:
                features = [features[0], z_features, prompt_features, z_prompt_features]
            else:
                features = [features[0], prompt_features]             
            return features
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    return

@torch.no_grad()
def test_kmeans_prt(model, test_loader, 
                    epoch, save_name, 
                    args, use_fast_Kmeans=False, 
                    predict_token='cls'
                ):
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    print('Collating features...')
    # First extract all features
    with tqdm(total=len(test_loader)) as pbar:
        for batch_idx, (images, label, _) in enumerate(test_loader):
            images = images.cuda(args.device)
            feats = forward(images, model, projection_head=None, predict_token=predict_token, mode='test')
            all_feats.append(feats.detach().cpu())
            targets = np.append(targets, label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                            else False for x in label]))
            
            pbar.update(1)

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = torch.cat(all_feats, dim=0)
    if use_fast_Kmeans is True:
        all_feats = all_feats.to(args.device)
        kmeans = K_Means(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-6, max_iterations=500, init='k-means++', 
                n_init=20, random_state=0, n_jobs=1, 
                pairwise_batch_size=None if all_feats.size(0)<args.fast_kmeans_batch_size else args.fast_kmeans_batch_size, 
                mode=None)
        kmeans.fit(all_feats)
        preds = kmeans.labels_.detach().cpu().numpy()
        all_feats = all_feats.detach().cpu().numpy()
    else:
        begin = time.time()
        all_feats = all_feats.numpy()
        kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0, verbose=0).fit(all_feats)
        end = time.time()
        preds = kmeans.labels_
        
    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    )
    
    return all_acc, old_acc, new_acc

@torch.no_grad()
def test_kmeans(model, test_loader,
                epoch, save_name, device,
                args, logger_class=None, 
                output_kmeans=False, num_clusters=None):
    model.eval()

    all_feats = []
    all_concat_feats = []
    targets = np.array([])
    mask = np.array([])
    logger_class('Collating features...')

    # First extract all features
    for batch_idx, _item in enumerate(test_loader):
        images = _item[0]
        label = _item[1]
        images = images.to(device)

        # Pass features through base model and then additional learnable transform (linear layer)
        # concat_feats = model(images, concat=True)
        feats = model(images, concat=False)
        concat_feats = feats.detach().clone()
        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_concat_feats.append(concat_feats.cpu().numpy())
        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        # for True or False, True for seen labels while False for unseen labels
        mask = np.append(
            mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label])
        )

    logger_class('Using no_l2 feature Fitting K-Means...')
    all_concat_feats = np.concatenate(all_concat_feats)

    # clustering all features
    num_clusters = args.num_labeled_classes + args.num_unlabeled_classes if num_clusters is None else num_clusters
    kmeans = KMeans(
        n_clusters=num_clusters, random_state=0
    ).fit(all_concat_feats)
    preds = kmeans.labels_
    logger_class('Using no_l2 feature Done')
    
    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(
        y_true=targets, y_pred=preds, mask=mask, T=epoch, 
        eval_funcs=args.eval_funcs, save_name=save_name + 'add'
    )
    logger_class('no_l2 feature kmeans Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc,
                                                                                        new_acc))

    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_feats)
    preds = kmeans.labels_
    logger_class('Using contrastive feature Done')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(
        y_true=targets, y_pred=preds, mask=mask, T=epoch, 
        eval_funcs=args.eval_funcs, save_name=save_name
    )
    if output_kmeans:
        return all_acc, old_acc, new_acc, kmeans, all_feats
    else:
        return all_acc, old_acc, new_acc


def fake_label_kmeans(model, test_loader,
                epoch, save_name, device,
                args, logger_class=None, K=200):
    model.eval()

    all_feats = []
    all_concat_feats = []
    targets = np.array([])
    mask = np.array([])
    logger_class('Collating features...')

    # First extract all features
    for batch_idx, _item in enumerate(test_loader):
        images = _item[0]
        label = _item[1]
        images = images.to(device)

        # Pass features through base model and then additional learnable transform (linear layer)
        # concat_feats = model(images, concat=True)
        feats = model(images, concat=False)
        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        # all_feats.append(feats.cpu().detach().numpy())
        # targets = np.append(targets, label.cpu().detach().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    logger_class(f'Fitting K-Means... with K={K}, dataset K = {args.num_labeled_classes + args.num_unlabeled_classes}')


    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    logger_class('Using contrastive feature Done')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name)


    return all_acc, old_acc, new_acc, kmeans, all_feats


from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import numpy as np
from sklearn.cluster import KMeans
import torch
from project_utils.cluster_utils import cluster_acc
from tqdm import tqdm

from scipy.optimize import minimize_scalar
from functools import partial
# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def test_kmeans(K, merge_test_loader, model, args=None, verbose=False):
    if K is None:
        K = args.num_labeled_classes + args.num_unlabeled_classes
    all_feats = []
    targets = np.array([])
    mask_lab = np.array([])     # From all the data, which instances belong to the labelled set
    mask_cls = np.array([])     # From all the data, which instances belong to seen classes

    print('Collating features...')
    model.eval()
    # First extract all features
    for batch_idx, (images, label, _, mask_lab_, attr) in enumerate(tqdm(merge_test_loader)):
        images = images.cuda()
        with torch.no_grad():
            feats = model(images)
        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        # whether the class is seen or unseen
        mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes))
                                                 else False for x in label]))
        mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())

    # -----------------------
    # K-MEANS
    # -----------------------
    mask_lab = mask_lab.astype(bool)
    mask_cls = mask_cls.astype(bool)

    all_feats = np.concatenate(all_feats)

    print('Fitting K-Means...')
    kmeans = KMeans(n_clusters=K, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    # -----------------------
    # EVALUATE
    # -----------------------
    mask = mask_lab

    labelled_acc, labelled_nmi, labelled_ari = cluster_acc(targets.astype(int)[mask], preds.astype(int)[mask]), \
                                               nmi_score(targets[mask], preds[mask]), \
                                               ari_score(targets[mask], preds[mask])

    unlabelled_acc, unlabelled_nmi, unlabelled_ari = cluster_acc(targets.astype(int)[~mask], preds.astype(int)[~mask]), \
                                                     nmi_score(targets[~mask], preds[~mask]), \
                                                     ari_score(targets[~mask], preds[~mask])

    if verbose:
        print('K')
        print('Labelled Instances acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(labelled_acc, labelled_nmi,
                                                                             labelled_ari))
        print('Unlabelled Instances acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(unlabelled_acc, unlabelled_nmi,
                                                                               unlabelled_ari))
    return labelled_acc

def test_kmeans_for_scipy(K, merge_test_loader, model, args=None, verbose=False):
    """
    In this case, the test loader needs to have the labelled and unlabelled subsets of the training data
    """
    K = int(K)
    model.eval()
    all_feats = []
    targets = np.array([])
    mask_lab = np.array([])     # From all the data, which instances belong to the labelled set
    mask_cls = np.array([])     # From all the data, which instances belong to seen classes

    print('Collating features...')
    # First extract all features
    
    for batch_idx, (images, label, _, mask_lab_, attr) in enumerate(tqdm(merge_test_loader)):
        images = images.cuda()
        with torch.no_grad():
            feats = model(images)
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

    print(f'Fitting K-Means for K = {K}...')
    kmeans = KMeans(n_clusters=K, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    # -----------------------
    # EVALUATE
    # -----------------------
    mask = mask_lab


    labelled_acc, labelled_nmi, labelled_ari = cluster_acc(targets.astype(int)[mask], preds.astype(int)[mask]), \
                                               nmi_score(targets[mask], preds[mask]), \
                                               ari_score(targets[mask], preds[mask])

    unlabelled_acc, unlabelled_nmi, unlabelled_ari = cluster_acc(targets.astype(int)[~mask],
                                                                 preds.astype(int)[~mask]), \
                                                     nmi_score(targets[~mask], preds[~mask]), \
                                                     ari_score(targets[~mask], preds[~mask])

    print(f'K = {K}')
    print('Labelled Instances acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(labelled_acc, labelled_nmi,
                                                                         labelled_ari))
    print('Unlabelled Instances acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(unlabelled_acc, unlabelled_nmi,
                                                                           unlabelled_ari))

    return -labelled_acc


def binary_search(merge_test_loader, args, max_classes, model, min_classes=None):
    min_classes = args.num_labeled_classes if min_classes is None else min_classes
    model.eval()
    # Iter 0
    big_k = max_classes
    small_k = min_classes
    diff = big_k - small_k
    middle_k = int(0.5 * diff + small_k)

    labelled_acc_big = test_kmeans(K=big_k, merge_test_loader=merge_test_loader, model=model, args=args)
    labelled_acc_small = test_kmeans(K=small_k, merge_test_loader=merge_test_loader, model=model, args=args)
    labelled_acc_middle = test_kmeans(K=middle_k, merge_test_loader=merge_test_loader, model=model, args=args)

    print(f'Iter 0: BigK {big_k}, Acc {labelled_acc_big:.4f} | MiddleK {middle_k}, Acc {labelled_acc_middle:.4f} | SmallK {small_k}, Acc {labelled_acc_small:.4f} ')
    all_accs = [labelled_acc_small, labelled_acc_middle, labelled_acc_big]
    best_acc_so_far = np.max(all_accs)
    best_acc_at_k = np.array([small_k, middle_k, big_k])[np.argmax(all_accs)]
    print(f'Best Acc so far {best_acc_so_far:.4f} at K {best_acc_at_k}')

    for i in range(1, int(np.log2(diff)) + 1):

        if labelled_acc_big > labelled_acc_small:
            best_acc = max(labelled_acc_middle, labelled_acc_big)
            small_k = middle_k
            labelled_acc_small = labelled_acc_middle
            diff = big_k - small_k
            middle_k = int(0.5 * diff + small_k)
        else:
            best_acc = max(labelled_acc_middle, labelled_acc_small)
            big_k = middle_k

            diff = big_k - small_k
            middle_k = int(0.5 * diff + small_k)
            labelled_acc_big = labelled_acc_middle

        labelled_acc_middle = test_kmeans(middle_k, merge_test_loader, model, args)

        all_accs = [labelled_acc_small, labelled_acc_middle, labelled_acc_big]
        best_acc_so_far = np.max(all_accs)
        best_acc_at_k = np.array([small_k, middle_k, big_k])[np.argmax(all_accs)]
    return best_acc_at_k


def scipy_optimise(merge_test_loader, args, max_classes, model):
    small_k = args.num_labeled_classes
    big_k = max_classes
    test_k_means_partial = partial(test_kmeans_for_scipy, merge_test_loader=merge_test_loader, model=model, args=args, verbose=True)
    res = minimize_scalar(test_k_means_partial, bounds=(small_k, big_k), method='bounded', options={'disp': True})
    print(f'Optimal K is {res.x}')
    return res.x

import faiss
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def run_hkmeans(x, num_cluster, T):
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[], 'cluster2cluster':[], 'logits':[]}
    
    for seed, n_cluser in enumerate(num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(n_cluser)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed # layer number
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = 0
        index = faiss.GpuIndexFlatL2(res, d, cfg)  
        if seed==0: # the first hierarchy from instance directly
            clus.train(x, index)  
            D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        else:
            # use centers of last layer to cluster the following layers, get centers
            clus.train(results['centroids'][seed - 1].cpu().numpy(), index)
            # find new assigned centers and labels
            D, I = index.search(results['centroids'][seed - 1].cpu().numpy(), 1)
            
        # the pseudo labels
        im2cluster = [int(n[0]) for n in I]
        # sample-to-centroid distances for each cluster 
        ## centroid in lower level to higher level, collect sample 2 center distances in diff levels
        Dcluster = [[] for c in range(k)]          
        for im, i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])

       # get cluster centroids, k=num_clusters
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)

        if seed > 0: 
            # the clustering is based on previous centroids
            # kmeans is trained with last-layer centroids, so we need to reformulate it 
            # "seed"'s pseudo labels (200)
            im2cluster = np.array(im2cluster) 
            # the new pseudo labels for previous centrids, {old center ID -> new center ID}
            results['cluster2cluster'].append(torch.LongTensor(im2cluster).cuda())
            # reformulate the clustering results of current clustering to follow previous label set
            # get last layers' clustering labels
            # im2cluster, max=200, shape=250; index, max=250, shape=3680
            # new pseudo labels based after formulating label mapper
            im2cluster = im2cluster[results['im2cluster'][seed - 1].cpu().numpy()]
            im2cluster = list(im2cluster)
    
        if len(set(im2cluster))==1:
            print("Warning! All samples are assigned to one cluster")

        # concentration estimation (phi)
        density = np.zeros(k)
        # Dcluster -- k clusters, dist -- distances to centroid within cluster
        for i,dist in enumerate(Dcluster):
            if len(dist) > 1:
                # Eq. 7
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                density[i] = d     
                
        #if cluster only has one point, use the max to estimate its concentration        
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 
                
        # filter out too small or too big
        density = density.clip(np.percentile(density,10),np.percentile(density,90)) 
        density = T*density/density.mean() # (num_cluster, )
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)    
        if seed > 0: 
            # maintain a logits from lower prototypes to higher, e.g., 250 * 200
            proto_logits = torch.mm(results['centroids'][-1], centroids.t())
            results['logits'].append(proto_logits.cuda())

        density = torch.Tensor(density).cuda()
        im2cluster = torch.LongTensor(im2cluster).cuda()   
        
        results['centroids'].append(centroids) # cluster centroids
        results['density'].append(density) # density for each cluster
        results['im2cluster'].append(im2cluster) # pseudo labels of last layer
    
    return results 

def get_l2_dist(center, features, labels):
    dis = torch.zeros(features.shape[0], 1).cuda()
    for ii, (fea, lab) in enumerate(zip(features, labels)):
        cen = center[lab]
        dis[ii] = ((cen-fea)**2).sum()
    return dis

def finch_data_process(pseudo_labels, features, T=0.05):
    results = {'im2cluster':[],'centroids':[],'density':[], 'cluster2cluster':[], 'logits':[]}
    # 'im2cluster', 'centroids', 'density', 'cluster2cluster', 'logits'
    results['im2cluster'] = [torch.from_numpy(val).cuda().long() for val in pseudo_labels.T]
    for ii, labs in enumerate(results['im2cluster']):
        center = get_centroids(features, labs)
        instance2center = get_l2_dist(center, F.normalize(features.cuda(), dim=1, p=2), labs.cuda())
        if ii != 0:
            results['logits'].append(
                F.normalize(results['centroids'][-1], p=2, dim=1) @ F.normalize(
                    center, p=2, dim=1).t()
            )
            results['cluster2cluster'].append(labs.long().cuda())
        num_clusters = center.shape[0]
        density = np.zeros(num_clusters)
        Dcluster = [[] for c in range(num_clusters)]          
        for im, i in enumerate(labs):
            Dcluster[i].append(instance2center[im][0])
        
        # Dcluster -- k clusters, dist -- distances to centroid within cluster
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (torch.stack(dist)**0.5).mean()/np.log(len(dist)+10)            
                density[i] = d
                    
        #if cluster only has one point, use the max to estimate its concentration        
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax.cpu().numpy()
                    
        # filter out too small or too big
        density = density.clip(np.percentile(density,10),np.percentile(density,90)) 
        density = T*density/density.mean() # (num_cluster, )
        results['density'].append(torch.from_numpy(density).cuda())
        
        results['centroids'].append(center)
    return results

def get_centroids(features, labels):
    max_pids, feat_dim = 1 + labels.max(), features.shape[1]
    center = torch.zeros(max_pids, feat_dim).cuda()
    counter = torch.zeros(max_pids, 1).cuda()
    center.index_add_(0, labels, features.cuda())
    counter.index_add_(0, labels, torch.ones_like(labels).float())
    center /= counter
    return F.normalize(center, p=2, dim=1)

def get_ent_mask(score, thr=40):
    # cls_num * ins_num
    score = F.softmax(score, 0)
    # (ins, 1)
    ent = (-score*(score+1e-8).log()).sum(1)
    thres = np.percentile(ent.cpu().numpy(), thr)
    mask = torch.zeros(ent.shape[0]).cuda()
    mask[ent>thres] = 1 # assign unseen with 1
    mask = mask.bool()
    seen_pse = score.argmax(1)[~mask].cpu().numpy()
    return mask.cpu().numpy(), seen_pse
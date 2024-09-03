import networkx as nx
import numpy as np
import torch
import scipy
import pickle
import scipy.sparse as sp
import os
import pandas as pd
import dgl
from collections import Counter, defaultdict

Ntypes = {
    "acm": {"p": "paper", "a": "author", "s": "subject"},
    "aminer": {"p": "paper", "a": "author", "r": "reference"},
    "freebase": {"m": "movie", "a": "author", "w": "writer", "d": "director"},
    "PubMed": {"G": "Geng", "D": "Disease", "C": "Chemical", "S": "Species"},
    "Freebase": {"B": "Book", "F": "Film", "M": "Music", "S": "Sports", "P": "People", "L": "Location", "O": "Organization", "U": "Business"},
}
num_node = {
    "acm": {"p": 4019, "a": 7167, "s": 60},
    "aminer": {"p": 6564, "a": 13329, "r": 35890},
    "freebase": {"m": 3492, "a": 33401, "w": 4459, "d": 2502},
}
Relations = {
    "acm": ["pa", "ps", "ap", "sp"],
    "aminer": ["pa", "pr", "ap", "rp"],
    "freebase": ["ma", "mw", "md", "am", "wm", "dm"],
    "PubMed": ["GG", "GD", "DD", "CG", "CD", "CC", "CS", "SG", "SD", "SS", "DG", "GC", "DC", "SC", "GS", "DS"],
    "Freebase": [],
}
for n1 in Ntypes["Freebase"].keys():
    for n2 in Ntypes["Freebase"].keys():
        if f"{n1}{n2}" not in Relations["Freebase"]:
            Relations["Freebase"].append(f"{n1}{n2}")
P_ntype = {"acm": "p", "aminer": "p", "freebase": "m", "PubMed": "D", "Freebase": "B"}


def load_dataset(prefix="DBLP"):
    import torch.nn.functional as F

    dataset_ntypes = {
        "acm": {"p": "paper", "a": "author", "s": "subject"},
        "aminer": {"p": "paper", "a": "author", "r": "reference"},
        "freebase": {"m": "movie", "a": "author", "w": "writer", "d": "director"},
    }
    dataset_predict = {"acm": "p", "aminer": "p", "freebase": "m"}
    hg, feats = _read_edges(f"{args.root}/HeCo/{args.dataset}", dataset_ntypes[args.dataset])
    features_list = []

    for ntype in dataset_ntypes[args.dataset].keys():
        if ntype in feats:
            hg.nodes[ntype].data[ntype] = feats[ntype]
            features_list.append(feats[ntype])
        else:
            hg.nodes[ntype].data[ntype] = torch.eye(hg.num_nodes(ntype))
            features_list.append(torch.eye(hg.num_nodes(ntype)))
    max_feat_dim = max([x.shape[1] for x in features_list])
    for ntype in dataset_ntypes[args.dataset].keys():
        feat = hg.nodes[ntype].data[ntype]
        if feat.shape[1] < max_feat_dim:
            pad = F.pad(feat, (0, max_feat_dim - feat.shape[1]))
            hg.nodes[ntype].data[ntype] = pad

    adjs = {}
    for etype in hg.canonical_etypes:

        src, dst = hg.edges(etype=etype)
        num_src, num_dst = hg.num_nodes(etype[0]), hg.num_nodes(etype[2])
        adj = SparseTensor(row=torch.LongTensor(src), col=torch.LongTensor(dst), sparse_sizes=(num_src, num_dst))
        name = f"{etype[0]}{etype[2]}"
        adjs[name] = adj
    num_classes, ratio_init_labels, ratio_nid = load_labels(f"{args.root}/HeCo/{args.dataset}", hg, dataset_predict[args.dataset])

    return hg, adjs, ratio_init_labels, num_classes, ratio_nid


class hg_data_loader:
    def __init__(self, dataset):
        prefix = "../../../../../"
        if dataset == "PubMed":
            self.path = os.path.join(prefix, "data/PubMed")
        elif dataset == "Freebase":
            self.path = os.path.join(prefix, "data/HGB_Freebase_random_subg")
        else:
            self.path = os.path.join(prefix, f"data/HeCo/{dataset}")
        self.dataset = dataset
        self.ntypes = Ntypes[dataset]
        self.predict_ntype = P_ntype[dataset]
        self.relations = Relations[dataset]
        self.nodes = self.load_nodes()
        self.links = self.load_links()

    def list_to_sp_mat(self, li):
        data = [x[2] for x in li]
        i = [x[0] for x in li]
        j = [x[1] for x in li]
        return sp.coo_matrix((data, (i, j)), shape=(self.nodes["total"], self.nodes["total"])).tocsr()

    def load_nodes(self):
        larget_data = ["PubMed", "Freebase"]
        nodes = {"total": 0, "count": Counter(), "attr": {}, "shift": {}}
        node_type_name = list(self.ntypes.keys())
        for ntype_id in range(len(node_type_name)):
            nodes["attr"][ntype_id] = None
        for file in os.listdir(self.path):
            name, ext = os.path.splitext(file)
            if "_feat" in name:
                n_type_id = node_type_name.index(name.split("_")[0])
                nodes["attr"][n_type_id] = sp.load_npz(os.path.join(self.path, f"{name}.npz")).toarray()
        if self.dataset in larget_data:
            if self.dataset == "PubMed":
                read_names = ["node_id", "node_name", "node_type", "node_attr"]
            else:
                read_names = ["node_id", "node_type"]
            node_file = pd.read_csv(os.path.join(self.path, f"node.dat"), sep="\t", names=read_names)
            nodes["total"] = len(node_file)
            for idx, ntype in enumerate(node_type_name):
                nodes["count"][idx] = len(node_file[node_file["node_type"] == idx])
        else:
            for idx, ntype in enumerate(node_type_name):
                nodes["count"][idx] = num_node[self.dataset][ntype]
                nodes["total"] += num_node[self.dataset][ntype]
        return nodes

    def load_links(self):
        links = {"total": 0, "count": Counter(), "meta": {}, "data": defaultdict(list)}

        for file in os.listdir(self.path):
            name, ext = os.path.splitext(file)
            if ext == ".txt":
                u, v = name
                v_name = f"{v}_" if u == v else v
                e = pd.read_csv(os.path.join(self.path, f"{u}{v}.txt"), sep="\t", names=[u, v_name])
                src = e[u].to_list()
                dst = e[v].to_list()
                r_id = self.relations.index(f"{u}{v}")
                r_r_id = self.relations.index(f"{v}{u}")
                if r_id not in links["meta"]:
                    links["meta"][r_id] = (u, v)
                if r_r_id not in links["meta"]:
                    links["meta"][r_r_id] = (v, u)
                links["data"][r_id] = [(s, d, 1) for s, d in zip(src, dst)]
                links["data"][r_r_id] = [(d, s, 1) for s, d in zip(src, dst)]
                links["count"][r_id] = len(src)
                links["count"][r_r_id] = len(src)
                links["total"] += len(src)
                r_id += 1
        new_data = {}
        # print(links['data'])
        for r_id in links["data"]:
            new_data[r_id] = self.list_to_sp_mat(links["data"][r_id])
        links["data"] = new_data
        return links

    def load_labels_with_ratio(self, num_rand):
        from sklearn.model_selection import train_test_split

        label_define = "random"
        label_ratio = ["20", "40", "60"]
        labels = torch.from_numpy(np.load(os.path.join(self.path, "labels.npy"))).long()
        self.num_classes = labels.max().item() + 1

        p_ntype_id = list(self.ntypes.keys()).index(self.predict_ntype)
        n = self.nodes["count"][p_ntype_id]
        ratio_init_labels = {ratio: np.zeros(n, dtype=int) for ratio in label_ratio}
        ratio_nid = {ratio: {seed: {} for seed in range(num_rand)} for ratio in label_ratio}

        for ratio in label_ratio:
            for i in range(num_rand):
                num_train_val = int(ratio)
                num_test = 1000 // self.num_classes
                idx_train, idx_val, idx_test = [], [], []
                for class_label in range(self.num_classes):
                    class_indices = torch.where(labels == class_label)[0]

                    # shuffle the indices
                    class_indices = class_indices[torch.randperm(len(class_indices))]
                    # Select 20 for train/val and 333 for test per class
                    train_val_indices = class_indices[:num_train_val]
                    test_indices = class_indices[num_train_val : num_train_val + num_test]

                    # Further split the 20 samples into train and val (10 each)
                    train_indices, val_indices = train_test_split(train_val_indices.numpy(), test_size=0.5, random_state=42)
                    idx_train.extend(train_indices)
                    idx_val.extend(val_indices)
                    idx_test.extend(test_indices)
                idx_train = np.array(idx_train)
                idx_val = np.array(idx_val)
                idx_test = np.array(idx_test)

                # idx_train = np.load(os.path.join(self.path, f"train_{ratio}.npy"))
                # idx_val = np.load(os.path.join(self.path, f"val_{ratio}.npy"))
                # idx_test = np.load(os.path.join(self.path, f"test_{ratio}.npy"))

                ratio_init_labels[ratio][idx_train] = labels[idx_train]
                ratio_init_labels[ratio][idx_val] = labels[idx_val]
                ratio_init_labels[ratio][idx_test] = labels[idx_test]

                ratio_init_labels[ratio] = torch.LongTensor(ratio_init_labels[ratio])
                ratio_nid[ratio][i]["train_idx"] = idx_train
                ratio_nid[ratio][i]["val_idx"] = idx_val
                ratio_nid[ratio][i]["test_idx"] = idx_test

        return labels, ratio_nid

    def load_labels(self):

        ## 要 5個 train/test idx 去訓練
        from sklearn.model_selection import StratifiedKFold

        n_splits = 10
        labels = torch.from_numpy(np.load(os.path.join(self.path, "labels.npy"))).long()
        self.num_classes = labels.max().item() + 1
        p_ntype_id = list(self.ntypes.keys()).index(self.predict_ntype)
        n = self.nodes["count"][p_ntype_id]
        ratio_init_labels = {i: np.zeros(n, dtype=int) for i in range(n_splits)}
        ratio_nids = {i: [] for i in range(n_splits)}
        seed = np.random.seed(1)

        X = np.arange(len(labels)).reshape(-1, 1)  # dummy X
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for i, (trainval_idx, test_idx) in enumerate(skf.split(X=X, y=labels)):
            label_train_val = labels[trainval_idx]
            skf_train_val = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for train_idx, val_idx in skf_train_val.split(X=X[trainval_idx], y=label_train_val):
                train_idx = trainval_idx[train_idx]  # Adjust indices to original dataset
                val_idx = trainval_idx[val_idx]  # Adjust indices to original dataset

                train_idx = torch.from_numpy(train_idx)
                val_idx = torch.from_numpy(val_idx)
                test_idx = torch.from_numpy(test_idx)
                ratio_init_labels[i][train_idx] = labels[train_idx]
                ratio_init_labels[i][val_idx] = labels[val_idx]
                ratio_init_labels[i][test_idx] = labels[test_idx]
                ratio_init_labels[i] = torch.LongTensor(ratio_init_labels[i])
                # print(len(trainval_idx), len(train_idx), len(val_idx), len(test_idx))

                ratio_nids[i] = {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}
                break

        return labels, ratio_nids

    def gen_file_for_evaluate(self, test_idx, label, file_name, mode="bi"):
        from sklearn.metrics import f1_score, roc_auc_score

        if test_idx.shape[0] != label.shape[0]:
            return
        if mode == "multi":
            multi_label = []
            for i in range(label.shape[0]):
                label_list = [str(j) for j in range(label[i].shape[0]) if label[i][j] == 1]
                multi_label.append(",".join(label_list))
            label = multi_label
        elif mode == "bi":
            label = np.array(label)
        else:
            return
        with open(file_name, "w") as f:
            for nid, l in zip(test_idx, label):
                f.write(f"{nid}\t\t{self.get_node_type(nid)}\t{l}\n")

    def visualization(self, ratio_embs, labels, save_file, display=False):
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        label_rate = ["20", "40", "60"]
        ratio_embs = [embs.cpu().detach().numpy() for embs in ratio_embs]
        labels = labels.cpu().detach().numpy()
        perplexity = min(30, ratio_embs[0].shape[0] - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        ratio_embs_2d = [tsne.fit_transform(embs) for embs in ratio_embs]
        colors = ["red", "blue", "green", "yellow", "purple", "orange", "black", "pink", "brown", "gray"]
        if display:
            fig, axs = plt.subplots(1, len(ratio_embs), figsize=(36, 8))
            fig.suptitle("t-SNE visualization of node embeddings with class labels")
            for i, embs_2d in enumerate(ratio_embs_2d):

                for label in np.unique(labels):
                    indices = [i for i, lbl in enumerate(labels) if lbl == label]
                    axs[i].scatter(embs_2d[indices, 0], embs_2d[indices, 1], color=colors[label], label=f"Class {label}", alpha=0.6)

                axs[i].set_title(f"{label_rate[i]}Train label node per class")
                axs[i].set_xlabel("x t-SNE vector")
                axs[i].set_ylabel("y t-SNE vector")
                axs[i].legend()

            fig.savefig(save_file)
        ratio_embs_2d = [torch.tensor(embs_2d) for embs_2d in ratio_embs_2d]
        return ratio_embs_2d

    def node_clustering_evaluate(self, embeds, y, n_labels, iter=10):
        from sklearn.cluster import KMeans
        from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

        nmi_list, ari_list = [], []
        embeds = embeds.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        for kmeans_random_state in range(iter):
            Y_pred = KMeans(n_clusters=n_labels, random_state=kmeans_random_state, n_init=10).fit(embeds).predict(embeds)
            nmi = normalized_mutual_info_score(y, Y_pred)
            ari = adjusted_rand_score(y, Y_pred)
            nmi_list.append(nmi)
            ari_list.append(ari)
        mean = {
            "nmi": np.mean(nmi_list),
            "ari": np.mean(ari_list),
        }
        std = {
            "nmi": np.std(nmi_list),
            "ari": np.std(ari_list),
        }
        return mean["nmi"], mean["ari"]

    def evaluator(self, pred_score, gt, pred, multilabel=False):
        from sklearn.metrics import f1_score, roc_auc_score
        import torch.nn.functional as F

        gt = gt.cpu().numpy()
        softmax_score = F.softmax(pred_score, dim=1).cpu().numpy()
        # accuracy= (pred == gt).sum().item() / len(gt)
        auc_score = roc_auc_score(
            y_true=gt,
            y_score=softmax_score,
            multi_class="ovr",
            average=None if multilabel else "macro",
        )
        # print(gt)
        # print(pred)

        return f1_score(gt, pred, average="micro"), f1_score(gt, pred, average="macro"), auc_score


def load_data(args):
    from scripts.data_loader import data_loader

    dl = hg_data_loader(args.dataset)
    # dl = data_loader('../../data/'+prefix)
    features = []
    for i in range(len(dl.nodes["count"])):
        th = dl.nodes["attr"][i]
        if th is None:
            features.append(sp.eye(dl.nodes["count"][i]))
        else:
            features.append(th)
    adjM = sum(dl.links["data"].values())

    ## HGB dataset
    # labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    # val_ratio = 0.2
    # train_idx = np.nonzero(dl.labels_train['mask'])[0]
    # np.random.shuffle(train_idx)
    # split = int(train_idx.shape[0]*val_ratio)
    # val_idx = train_idx[:split]
    # train_idx = train_idx[split:]
    # train_idx = np.sort(train_idx)
    # val_idx = np.sort(val_idx)
    # test_idx = np.nonzero(dl.labels_test['mask'])[0]
    # labels[train_idx] = dl.labels_train['data'][train_idx]
    # labels[val_idx] = dl.labels_train['data'][val_idx]
    # if prefix != 'IMDB':
    #     labels = labels.argmax(axis=1)
    # train_val_test_idx = {}
    # train_val_test_idx['train_idx'] = train_idx
    # train_val_test_idx['val_idx'] = val_idx
    # train_val_test_idx['test_idx'] = test_idx
    if args.dataset == "PubMed" or args.dataset == "Freebase":
        labels, train_val_test_idx = dl.load_labels()
    else:
        labels, train_val_test_idx = dl.load_labels_with_ratio(args.repeat)
    return features, adjM, labels, train_val_test_idx, dl

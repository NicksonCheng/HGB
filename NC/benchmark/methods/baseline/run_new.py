import sys

sys.path.append("../../")
import time
from datetime import datetime
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np

from utils.pytorchtools import EarlyStopping
from utils.data import load_data

# from utils.tools import index_generator, evaluate_results_nc, parse_minibatch
from GNN import myGAT
import dgl
from collections import Counter


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)


def run_model_DBLP(args):
    feats_type = args.feats_type
    features_list, adjM, labels, train_val_test_idx, dl = load_data(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    features_list = [mat2tensor(features).to(device) for features in features_list]

    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []  # [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)

    formatted_now = datetime.now().strftime("[%Y-%m-%d_%H:%M:%S]")
    labels = torch.LongTensor(labels).to(device)

    if args.dataset == "PubMed":
        target_node_size = features_list[1].shape[0]
    else:
        target_node_size = features_list[0].shape[0]

    total_auc_list = []
    total_micro_list = []
    total_macro_list = []
    ratio_embs = []
    for ratio in train_val_test_idx.keys():
        if(args.dataset == "PubMed"):
            train_idx = train_val_test_idx[ratio]["train_idx"]
            train_idx = np.sort(train_idx)
            val_idx = train_val_test_idx[ratio]["val_idx"]
            val_idx = np.sort(val_idx)
            test_idx = train_val_test_idx[ratio]["test_idx"]
            test_idx = np.sort(test_idx)
        edge2type = {}
        for k in dl.links["data"]:
            for u, v in zip(*dl.links["data"][k].nonzero()):
                edge2type[(u, v)] = k
        for i in range(dl.nodes["total"]):
            if (i, i) not in edge2type:
                edge2type[(i, i)] = len(dl.links["count"])
        for k in dl.links["data"]:
            for u, v in zip(*dl.links["data"][k].nonzero()):
                if (v, u) not in edge2type:
                    edge2type[(v, u)] = k + 1 + len(dl.links["count"])

        g = dgl.DGLGraph(adjM + (adjM.T))
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        g = g.to(device)
        e_feat = []
        for u, v in zip(*g.edges()):
            u = u.cpu().item()
            v = v.cpu().item()
            e_feat.append(edge2type[(u, v)])
        e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)

        auc_list = []
        micro_list = []
        macro_list = []
        nmi_list = []
        ari_list = []
        for i in range(args.repeat):
            if(args.dataset != "PubMed"):
                train_idx = train_val_test_idx[ratio][i]["train_idx"]
                train_idx = np.sort(train_idx)
                val_idx = train_val_test_idx[ratio][i]["val_idx"]
                val_idx = np.sort(val_idx)
                test_idx = train_val_test_idx[ratio][i]["test_idx"]
                test_idx = np.sort(test_idx)
            # num_classes = dl.labels_train['num_classes']
            num_classes = dl.num_classes

            heads = [args.num_heads] * args.num_layers + [1]
            net = myGAT(
                g,
                args.edge_feats,
                len(dl.links["count"]) * 2 + 1,
                in_dims,
                args.hidden_dim,
                num_classes,
                args.num_layers,
                heads,
                F.elu,
                args.dropout,
                args.dropout,
                args.slope,
                True,
                0.05,
            )
            net.to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            # training loop
            net.train()
            early_stopping = EarlyStopping(
                patience=args.patience, verbose=True, save_path="checkpoint/checkpoint_{}_{}.pt".format(args.dataset, args.num_layers)
            )
            for epoch in range(args.epoch):
                t_start = time.time()
                # training
                net.train()
                logits = net(features_list, e_feat)
                logp = F.log_softmax(logits, 1)
                train_loss = F.nll_loss(logp[train_idx], labels[train_idx])

                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                t_end = time.time()

                # print training info
                print("Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}".format(epoch, train_loss.item(), t_end - t_start))

                t_start = time.time()
                # validation
                net.eval()
                with torch.no_grad():
                    logits = net(features_list, e_feat)
                    logp = F.log_softmax(logits, 1)
                    val_loss = F.nll_loss(logp[val_idx], labels[val_idx])
                t_end = time.time()
                # print validation info
                print("Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}".format(epoch, val_loss.item(), t_end - t_start))
                # early stopping
                early_stopping(val_loss, net)
                if early_stopping.early_stop:
                    print("Early stopping!")
                    break
                # break
            # testing with evaluate_results_nc
            net.load_state_dict(torch.load("checkpoint/checkpoint_{}_{}.pt".format(args.dataset, args.num_layers)))
            net.eval()
            test_logits = []
            with torch.no_grad():
                logits = net(features_list, e_feat)
                test_logits = logits[test_idx]
                test_pred = test_logits.cpu().numpy().argmax(axis=1)
                onehot = np.eye(num_classes, dtype=np.int32)
                # dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_name=f"{args.dataset}_{args.run}.txt")
                # test_pred = onehot[pred]

                # emb_2d=dl.visualization(logits[:target_node_size],labels,f"log/{args.dataset}/{formatted_now}_{ratio}.png",False)
                if args.dataset != "PubMed":
                    nmi, ari = dl.node_clustering_evaluate(logits[:target_node_size], labels, num_classes, 10)
                    nmi_list.append(nmi)
                    ari_list.append(ari)

                micro, macro, auc = dl.evaluator(test_logits, labels[test_idx], test_pred)
                auc_list.append(auc)
                micro_list.append(micro)
                macro_list.append(macro)
        ratio_embs.append(logits[:target_node_size])
        auc_list = np.array(auc_list)
        micro_list = np.array(micro_list)
        macro_list = np.array(macro_list)
        if args.dataset == "PubMed" or args.dataset == "Freebase":
            total_auc_list.append(np.mean(auc_list))
            total_micro_list.append(np.mean(micro_list))
            total_macro_list.append(np.mean(macro_list))
        else:
            with open(f"log/{args.dataset}/{formatted_now}", "a") as log_file:
                log_file.write(
                    "\t Label Rate:{}% Accuracy:[{:.4f},{:.4f}] Micro-F1:[{:.4f},{:.4f}] Macro-F1:[{:.4f},{:.4f}] \n".format(
                        ratio, np.mean(auc_list), np.std(auc_list), np.mean(micro_list), np.std(micro_list), np.mean(macro_list), np.std(macro_list)
                    )
                )
                log_file.write(
                    "\t[clustering] nmi: [{:.4f}, {:.4f}] ari: [{:.4f}, {:.4f}]\n".format(
                        np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)
                    )
                )
                log_file.close()
                # print(dl.evaluate(pred))
    if args.visual:
        with torch.no_grad():
            logits = net(features_list, e_feat)
            emb_2d = dl.visualization(ratio_embs, labels, f"log/{args.dataset}/{formatted_now}.png", args.visual)
    if args.dataset == "PubMed" or args.dataset == "Freebase":
        with open(f"log/{args.dataset}/{formatted_now}", "a") as log_file:
            # print(total_auc_list)
            # print(total_micro_list)
            # print(total_macro_list)
            total_auc_list = np.array(total_auc_list)
            total_micro_list = np.array(total_micro_list)
            total_macro_list = np.array(total_macro_list)
            log_file.write(
                "\t Label Rate:{}% Accuracy:[{:.4f},{:.4f}] Micro-F1:[{:.4f},{:.4f}] Macro-F1:[{:.4f},{:.4f}] \n".format(
                    ratio,
                    np.mean(total_auc_list),
                    np.std(total_auc_list),
                    np.mean(total_micro_list),
                    np.std(total_micro_list),
                    np.mean(total_macro_list),
                    np.std(total_macro_list),
                )
            )
            log_file.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MRGNN testing for the DBLP dataset")
    ap.add_argument(
        "--feats-type",
        type=int,
        default=3,
        help="Type of the node features used. "
        + "0 - loaded features; "
        + "1 - only target node features (zero vec for others); "
        + "2 - only target node features (id vec for others); "
        + "3 - all id vec. Default is 2;"
        + "4 - only term features (id vec for others);"
        + "5 - only term features (zero vec for others).",
    )
    ap.add_argument("--hidden-dim", type=int, default=64, help="Dimension of the node hidden state. Default is 64.")
    ap.add_argument("--num-heads", type=int, default=8, help="Number of the attention heads. Default is 8.")
    ap.add_argument("--epoch", type=int, default=300, help="Number of epochs.")
    ap.add_argument("--patience", type=int, default=30, help="Patience.")
    ap.add_argument("--repeat", type=int, default=1, help="Repeat the training and testing for N times. Default is 1.")
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--slope", type=float, default=0.05)
    ap.add_argument("--dataset", type=str)
    ap.add_argument("--edge-feats", type=int, default=64)
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--visual", type=bool)

    args = ap.parse_args()
    os.makedirs("checkpoint", exist_ok=True)
    run_model_DBLP(args)

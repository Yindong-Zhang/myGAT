import json
import os
# import pdb
import torch
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
from networkx.readwrite import json_graph

sys.setrecursionlimit(99999)


def run_dfs(adj, msk, u, ind, nb_nodes):
    if msk[u] == -1:
        msk[u] = ind
        #for v in range(nb_nodes):
        for v in adj[u,:].nonzero()[1]:
            #if adj[u,v]== 1:
            run_dfs(adj, msk, v, ind, nb_nodes)

# Use depth-first search to split a graph into subgraphs
def dfs_split(adj):
    # Assume adj is of shape [nb_nodes, nb_nodes]
    nb_nodes = adj.shape[0]
    ret = np.full(nb_nodes, -1, dtype=np.int32)

    graph_id = 0

    for i in range(nb_nodes):
        if ret[i] == -1:
            run_dfs(adj, ret, i, graph_id, nb_nodes)
            graph_id += 1

    return ret

def test(adj, mapping):
    nb_nodes = adj.shape[0]
    for i in range(nb_nodes):
        #for j in range(nb_nodes):
        for j in adj[i, :].nonzero()[1]:
            if mapping[i] != mapping[j]:
              #  if adj[i,j] == 1:
                 return False
    return True



def find_split(adj, mapping, ds_label):
    nb_nodes = adj.shape[0]
    dict_splits={}
    for i in range(nb_nodes):
        #for j in range(nb_nodes):
        for j in adj[i, :].nonzero()[1]:
            if mapping[i]==0 or mapping[j]==0:
                dict_splits[0]=None
            elif mapping[i] == mapping[j]:
                if ds_label[i]['val'] == ds_label[j]['val'] and ds_label[i]['test'] == ds_label[j]['test']:

                    if mapping[i] not in dict_splits.keys():
                        if ds_label[i]['val']:
                            dict_splits[mapping[i]] = 'val'

                        elif ds_label[i]['test']:
                            dict_splits[mapping[i]]='test'

                        else:
                            dict_splits[mapping[i]] = 'train'

                    else:
                        if ds_label[i]['test']:
                            ind_label='test'
                        elif ds_label[i]['val']:
                            ind_label='val'
                        else:
                            ind_label='train'
                        if dict_splits[mapping[i]]!= ind_label:
                            print ('inconsistent labels within a graph exiting!!!')
                            return None
                else:
                    print ('label of both nodes different, exiting!!')
                    return None
    return dict_splits




def process_p2p(datapath):


    print ('Loading G...')
    with open(os.path.join(datapath, 'ppi-G.json')) as jsonfile:
        g_data = json.load(jsonfile)
    print (len(g_data))
    G = json_graph.node_link_graph(g_data)

    #Extracting adjacency matrix
    adj=nx.adjacency_matrix(G)

    prev_key=''
    for key, value in g_data.items():
        if prev_key!=key:
            print (key)
            prev_key=key

    print ('Loading id_map...')
    with open(os.path.join(datapath, 'ppi-id_map.json')) as jsonfile:
        id_map = json.load(jsonfile)
    print (len(id_map))

    id_map = {int(k):int(v) for k,v in id_map.items()}
    for key, value in id_map.items():
        id_map[key]=[value]
    print (len(id_map))

    print ('Loading features...')
    features_=np.load(os.path.join(datapath, 'ppi-feats.npy'))
    print (features_.shape)

    #standarizing features
    from sklearn.preprocessing import StandardScaler

    train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
    train_feats = features_[train_ids[:,0]]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    features_ = scaler.transform(features_)

    features = sp.csr_matrix(features_).tolil()


    print ('Loading class_map...')
    class_map = {}
    with open(os.path.join(datapath, 'ppi-class_map.json')) as jsonfile:
        class_map = json.load(jsonfile)
    print (len(class_map))
    
    #pdb.set_trace()
    #Split graph into sub-graphs
    print ('Splitting graph...')
    splits=dfs_split(adj)

    #Rearrange sub-graph index and append sub-graphs with 1 or 2 nodes to bigger sub-graphs
    print ('Re-arranging sub-graph IDs...')
    list_splits=splits.tolist()
    group_inc=1

    for i in range(np.max(list_splits)+1):
        if list_splits.count(i)>=3:
            splits[np.array(list_splits) == i] =group_inc
            group_inc+=1
        else:
            #splits[np.array(list_splits) == i] = 0
            ind_nodes=np.argwhere(np.array(list_splits) == i)
            ind_nodes=ind_nodes[:,0].tolist()
            split=None
            
            for ind_node in ind_nodes:
                if g_data['nodes'][ind_node]['val']:
                    if split is None or split=='val':
                        splits[np.array(list_splits) == i] = 21
                        split='val'
                    else:
                        raise ValueError('new node is VAL but previously was {}'.format(split))
                elif g_data['nodes'][ind_node]['test']:
                    if split is None or split=='test':
                        splits[np.array(list_splits) == i] = 23
                        split='test'
                    else:
                        raise ValueError('new node is TEST but previously was {}'.format(split))
                else:
                    if split is None or split == 'train':
                        splits[np.array(list_splits) == i] = 1
                        split='train'
                    else:
                        # pdb.set_trace()
                        raise ValueError('new node is TRAIN but previously was {}'.format(split))

    #counting number of nodes per sub-graph
    list_splits=splits.tolist()
    nodes_per_graph=[]
    for i in range(1,np.max(list_splits) + 1):
        nodes_per_graph.append(list_splits.count(i))

    #Splitting adj matrix into sub-graphs
    subgraph_nodes=np.max(nodes_per_graph)
    adj_sub=np.empty((len(nodes_per_graph), subgraph_nodes, subgraph_nodes))
    feat_sub = np.empty((len(nodes_per_graph), subgraph_nodes, features.shape[1]))
    labels_sub = np.empty((len(nodes_per_graph), subgraph_nodes, 121))

    for i in range(1, np.max(list_splits) + 1):
        #Creating same size sub-graphs
        indexes = np.where(splits == i)[0]
        subgraph_=adj[indexes,:][:,indexes]

        if subgraph_.shape[0]<subgraph_nodes or subgraph_.shape[1]<subgraph_nodes:
            subgraph=np.identity(subgraph_nodes)
            feats=np.zeros([subgraph_nodes, features.shape[1]])
            labels=np.zeros([subgraph_nodes,121])
            #adj
            subgraph = sp.csr_matrix(subgraph).tolil()
            subgraph[0:subgraph_.shape[0],0:subgraph_.shape[1]]=subgraph_
            adj_sub[i-1,:,:]=subgraph.todense()
            #feats
            feats[0:len(indexes)]=features[indexes,:].todense()
            feat_sub[i-1,:,:]=feats
            #labels
            for j,node in enumerate(indexes):
                labels[j,:]=np.array(class_map[str(node)])
            labels[indexes.shape[0]:subgraph_nodes,:]=np.zeros([121])
            labels_sub[i - 1, :, :] = labels

        else:
            adj_sub[i - 1, :, :] = subgraph_.todense()
            feat_sub[i - 1, :, :]=features[indexes,:].todense()
            for j,node in enumerate(indexes):
                labels[j,:]=np.array(class_map[str(node)])
            labels_sub[i-1, :, :] = labels

    # Get relation between id sub-graph and tran,val or test set
    dict_splits = find_split(adj, splits, g_data['nodes'])

    # Testing if sub graphs are isolated
    print ('Are sub-graphs isolated?')
    print (test(adj, splits))

    #Splitting tensors into train,val and test
    train_split=[]
    val_split=[]
    test_split=[]
    for key, value in dict_splits.items():
        if dict_splits[key]=='train':
            train_split.append(int(key)-1)
        elif dict_splits[key] == 'val':
            val_split.append(int(key)-1)
        elif dict_splits[key] == 'test':
            test_split.append(int(key)-1)

    train_adj=adj_sub[train_split,:,:]
    val_adj=adj_sub[val_split,:,:]
    test_adj=adj_sub[test_split,:,:]

    train_feat=feat_sub[train_split,:,:]
    val_feat = feat_sub[val_split, :, :]
    test_feat = feat_sub[test_split, :, :]

    train_labels = labels_sub[train_split, :, :]
    val_labels = labels_sub[val_split, :, :]
    test_labels = labels_sub[test_split, :, :]

    train_nodes=np.array(nodes_per_graph[train_split[0]:train_split[-1]+1])
    val_nodes = np.array(nodes_per_graph[val_split[0]:val_split[-1]+1])
    test_nodes = np.array(nodes_per_graph[test_split[0]:test_split[-1]+1])


    #Masks with ones

    tr_msk = np.zeros((len(nodes_per_graph[train_split[0]:train_split[-1]+1]), subgraph_nodes))
    vl_msk = np.zeros((len(nodes_per_graph[val_split[0]:val_split[-1] + 1]), subgraph_nodes))
    ts_msk = np.zeros((len(nodes_per_graph[test_split[0]:test_split[-1]+1]), subgraph_nodes))

    for i in range(len(train_nodes)):
        for j in range(train_nodes[i]):
            tr_msk[i][j] = 1

    for i in range(len(val_nodes)):
        for j in range(val_nodes[i]):
            vl_msk[i][j] = 1

    for i in range(len(test_nodes)):
        for j in range(test_nodes[i]):
            ts_msk[i][j] = 1

    return train_adj,val_adj,test_adj,train_feat,val_feat,test_feat,train_labels,val_labels, test_labels, train_nodes, val_nodes, test_nodes, tr_msk, vl_msk, ts_msk

def save_p2p(datapath):
    train_adj, val_adj, test_adj, \
    train_feat, val_feat, test_feat, \
    train_labels, val_labels, test_labels, \
    train_nodes, val_nodes, test_nodes, \
    tr_msk, vl_msk, ts_msk = process_p2p(datapath)

    np.savez(os.path.join(datapath, 'train_ppi.npz'), **{'train_adj': train_adj, 'train_feat': train_feat, 'train_labels': train_labels, 'train_nodes': train_nodes, 'train_masks': tr_msk})
    np.savez(os.path.join(datapath, 'val_ppi.npz'), **{'val_adj': val_adj, 'val_feat': val_feat, 'val_labels': val_labels, 'val_nodes': val_nodes, 'val_masks': vl_msk})
    np.savez(os.path.join(datapath, 'test_ppi.npz'), **{'test_adj': test_adj, 'test_feat': test_feat, 'test_labels': test_labels, 'test_nodes': test_nodes, 'test_masks': ts_msk})



def load_p2p(datapath):
    print("Loading dataset p2p...")
    train_dict  = np.load(os.path.join(datapath, 'train_ppi.npz'))
    val_dict    = np.load(os.path.join(datapath, 'val_ppi.npz'))
    test_dict   = np.load(os.path.join(datapath, 'test_ppi.npz'))

    train_feat, val_feat, test_feat         = map(torch.FloatTensor, [train_dict['train_feat'], val_dict['val_feat'], test_dict['test_feat']])
    train_adj, val_adj, test_adj            = map(torch.FloatTensor, [train_dict['train_adj'], val_dict['val_adj'], test_dict['test_adj']])
    train_labels, val_labels, test_labels   = map(torch.FloatTensor, [train_dict['train_labels'], val_dict['val_labels'], test_dict['test_labels']])
    train_nodes, val_nodes, test_nodes      = map(torch.LongTensor, [train_dict['train_nodes'], val_dict['val_nodes'], test_dict['test_nodes']])
    tr_msk, vl_msk, ts_msk                  = map(torch.FloatTensor, [train_dict['train_masks'], val_dict['val_masks'], test_dict['test_masks']])
    return train_adj, val_adj, test_adj, \
    train_feat, val_feat, test_feat, \
    train_labels, val_labels, test_labels, \
    train_nodes, val_nodes, test_nodes, \
    tr_msk, vl_msk, ts_msk

def create_data():
    print("Create test data...")
    train_feat = 100 * torch.randn([20, 100, 50])
    train_adj  = torch.rand([20, 100, 100])
    train_labels= torch.randint(0, 2, [20, 100, 10], dtype= torch.float)
    train_nodes= torch.randint(10, 100, [20, ])
    tr_msk     = torch.randint(0, 2, [20, 100, 1], dtype= torch.float)

    val_feat   = 100 * torch.randn([4, 100, 50])
    val_adj    = torch.rand([4, 100, 100])
    val_labels  = torch.randint(0, 2, [4, 100, 10], dtype= torch.float)
    val_nodes  = torch.randint(10, 100, [4, ])
    vl_msk     = torch.randint(0, 2, [4, 100, 1], dtype= torch.float)

    test_feat = 100 * torch.randn([10, 100, 50])
    test_adj = torch.rand([10, 100, 100])
    # , dtype = torch.long
    test_labels = torch.randint(0, 2, [10, 100, 10], dtype= torch.float)
    test_nodes = torch.randint(10, 100, [10, ])
    ts_msk = torch.randint(0, 2, [10, 100, 1], dtype= torch.float)

    return train_adj, val_adj, test_adj, \
    train_feat, val_feat, test_feat, \
    train_labels, val_labels, test_labels, \
    train_nodes, val_nodes, test_nodes, \
    tr_msk, vl_msk, ts_msk

if __name__ == "__main__":
    # process_p2p('/home/yindong/Data/homogeneous_attributed_graph/ppi')
    load_p2p('./data/ppi')
    # save_p2p('./data/ppi')
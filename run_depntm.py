# This code is modified based on the source code of paper "CausalVAE: Disentangled Representation Learning via Neural Structural Causal Models", url: https://github.com/huawei-noah/trustworthyAI/tree/master/research/CausalVAE; 
#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
import time
import torch
from codebase import utils as ut
import argparse
from pprint import pprint
cuda = torch.cuda.is_available()
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
import numpy as np
import math
import time
from torch.utils import data
from utils import _h_A
# import matplotlib.pyplot as plt
import random
import torch.utils.data as Data
# from PIL import Image
import os
import numpy as np
# from torchvision import transforms
from codebase import utils as ut
from codebase.models.depntm import DepNTM
import argparse
from pprint import pprint
cuda = torch.cuda.is_available()
# from torchvision.utils import save_image

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--epoch_max',   type=int, default=200,    help="Number of training epochs")
parser.add_argument('--iter_save',   type=int, default=5, help="Save model every n epochs")
parser.add_argument('--run',         type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',       type=int, default=1,     help="Flag for training")
parser.add_argument('--color',       type=int, default=False,     help="Flag for color")
parser.add_argument('--toy',       type=str, default="pendulum_mask",     help="Flag for toy")
parser.add_argument('--flag', type=int, default=0, metavar='N', 
    help="flag")
parser.add_argument('--hidden', type=int, default=512, metavar='N', 
    help="The size of hidden units in MLP inference network (default 256)")
parser.add_argument('--dropout', type=float, default=0.8, metavar='N', 
    help="The drop-out probability of MLP (default 0.8)")
parser.add_argument('--lr', type=float, default=1e-3, metavar='N', 
    help="The learning rate of model (default 1e-3)")
parser.add_argument('--topics', type=int, default=50, metavar='N',
    help="The amount of topics to be discover (default 50)")
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
    help="Training batch size.")
parser.add_argument('--vocab-size', type=int, default=2000, metavar='N',
    help="Vocabulary size of topic modelling")
parser.add_argument('--topicembedsize', type=int, default=256, metavar='N',
    help="Topic embedding size of topic modelling")
parser.add_argument('--alternative-epoch', type=int, default=10, metavar='N',
    help="Alternative epoch size for wake sleep algorithm")
parser.add_argument('--training-epoch', type=int, default=200, metavar='N',
    help="Alternative epoch size for wake sleep algorithm")
parser.add_argument('--cuda', action='store_true', default=True,
    help="Flag for disable CUDA training.")
parser.add_argument('--inf-nonlinearity', default='tanh', metavar='N',
    help="Options for non-linear function.(default tanh)")
parser.add_argument('--data-path', default='data/20news/', metavar='N',
    help="Directory for corpus")
parser.add_argument('--model-save-path', default='new_sav_drop8_topic20vec512/', metavar='N',
    help="Directory for corpus")

parser.add_argument('--label-path', default='data/20news/', metavar='N',
    help="Directory for corpus")
parser.add_argument('--label-voc-path', default='data/20news/', metavar='N',
    help="Directory for corpus")
parser.add_argument('--gama', type=float, default=1.0, metavar='N',
    help="gama")
parser.add_argument('--z2-dim', type=int, default=16, metavar='N',
    help="z2_dim")
parser.add_argument('--background-topics', type=int, default=1, metavar='N',
    help="background_topics")
parser.add_argument('--fname', default='q_a_20', metavar='N',
    help="Directory for corpus")
parser.add_argument('--embedding-path', default='data/20news/', metavar='N',
    help="Directory for corpus")

MAX_TO_KEEP=5
TOPN=20
torch.manual_seed(0)

label_voc={}
args = parser.parse_args()
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
MODEL_SAV_PATH=args.model_save_path
print(MODEL_SAV_PATH)
align_list=[]

def _sigmoid(x):
    I = torch.eye(x.size()[0]).to(device)
    x = torch.inverse(I + torch.exp(-x))
    return x
    
class DeterministicWarmup(object):
    """
    Linear deterministic warm-up as described in
    [S?nderby 2016].
    """
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1/n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t
        return self.t

#Corpus object
class BOW_TopicModel_Corpus(torch.utils.data.Dataset):
    def __init__(self, vocabulary_size, data_path, label_path='', label_voc_path='', label_num=20, loader=None):
        #data_path: path route for train.feat and test.feat
        self.vocabulary_size=vocabulary_size
        self.doc_set = []
        self.doc_count=0.0
        doc_index=0
        self.label_voc = {}
        self.labels = []
        self.label_num=label_num

        with open(data_path, 'r') as f:
            self.doc_set=f.readlines()
            doc_index=len(self.doc_set)
        with open(label_voc_path, 'r') as f:
            docs=f.readlines()
            for label_ind in range(label_num):
                self.label_voc[docs[label_ind].strip().split('\t')[0]]=label_ind
        label_voc=self.label_voc
        with open(label_path, 'r') as f:
            self.labels=f.readlines()
        self.loader=loader
        self.doc_count=doc_index

    def __len__(self):
        return len(self.doc_set)
    
    def tokens2vec(self, token_list, vocabulary_size):
        vec=np.zeros(vocabulary_size)
        word_count=0
        for token in token_list:
            token_index=int(token.split(':')[0])
            token_tf=int(token.split(':')[1])
            word_count+=token_tf
            vec[token_index-1]=token_tf  #pay attention to the input word ID, whether to -1
        return vec, word_count

    def __getitem__(self, index):
        item_list=self.doc_set[index].strip().split(' ')
        class_label=np.zeros(self.label_num)
        ls=self.labels[index].strip().split('\t')[1].strip().split(' ')
        for l in ls:
            class_label[self.label_voc[l.strip()]]=1
        vec, word_count = self.tokens2vec(item_list[1:], self.vocabulary_size)
        vec=torch.from_numpy(np.array(vec, dtype='float32'))
        class_label=torch.from_numpy(np.array(class_label, dtype='float32'))
        word_count = torch.from_numpy(np.array(word_count,dtype=float))
        return vec, class_label, word_count

def collate_fn(batch):
    batch = list(zip(*batch))
    labels = batch[1]
    texts = batch[0]
    word_count=batch[2]

    del batch
    return texts,labels,word_count
#==============================================================================================================
def ReadDoc(name):
    fp=open(name,'r')
    doc=fp.readlines()
    fp.close()
    return doc

def ReadDictionary(vocabpath):
    word2id=dict()
    id2word=dict()
    vocabulary=dict()
    txt=ReadDoc(vocabpath)
    for i in range(0,len(txt)):
        if len(txt)>2:
            tmp_list=txt[i].strip().split(' ')
            word2id[tmp_list[0]]=i
            id2word[i]=tmp_list[0]
            vocabulary[tmp_list[0]]=tmp_list[1]
    return word2id, id2word, vocabulary

word2id, id2word, vocabulary = ReadDictionary(args.data_path + args.fname + '_voc.txt')
print('voc size=',len(word2id))

def topic_coherence_file_export(topicmat, id2word, topn, mat_theta, mat_topic, mat_word,mat_dagA):
    matrix_path = MODEL_SAV_PATH[0:-1] +"matrix/"
    if not os.path.exists(matrix_path):
        os.makedirs(matrix_path) 
    f=open(matrix_path +"topics.txt", 'w')

    np.savetxt(matrix_path+"beta.txt", topicmat)
    np.savetxt(matrix_path+"theta.txt", mat_theta)
    np.savetxt(matrix_path+"topic_embedding.txt", mat_topic)
    np.savetxt(matrix_path+"word_embedding.txt", mat_word)
    np.savetxt(matrix_path+"dagA.txt", mat_dagA)
    for topic in topicmat:
        topics_word_list=[]
        word_cnt=1
        tmp_list=[]
        # Build word, probability tuple for each topic
        for index, value in enumerate(topic):
            tmp_list.append((index, value))
        #Decently sort the word according to its probability
        sorted_list = sorted(tmp_list, key = lambda s:s[:][1], reverse=True)
        for pair in sorted_list:
            if word_cnt > topn: 
                break
            if pair[0] not in id2word.keys():
                continue
                # print(pair[0])
                # sys.exit(0)
            topics_word_list.append(id2word[pair[0]])
            word_cnt+=1
        f.write(' '.join(topics_word_list)+'\n')
    f.close()

def save_model_by_name(model, global_step):
    save_dir = os.path.join('checkpoints', model.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
    state = model.state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))

class CausalNTM(nn.Module):
    def __init__(self, name=None, vocabulary_size=2000, n_hidden=256, dropout_prob=0.8, n_topics=50, topic_embeddings_size=128, inf_nonlinearity='tanh', alternative_epoch=10,flag=0,gama=1.0,z2_dim=16,word_embedding=torch.zeros([1, 1], dtype=torch.float64),topic_embedding=torch.zeros([1, 1], dtype=torch.float64)):
        super(CausalNTM, self).__init__()
        self.lvae = DepNTM(name=name, vocabulary_size=args.vocab_size, n_hidden=args.hidden, n_topics=args.topics,n_background_topics=args.background_topics,flag=args.flag,z2_dim=args.z2_dim)
        self.gama = gama
#------------------------------------------------------------------------------------------------------------------------
    def train_and_eval(self, 
                       train_dataloader, 
                       test_dataloader, 
                       learning_rate=1e-4, 
                       batch_size=65, 
                       training_epoch=1000, 
                       alternative_epoch=10):
        optimizer = torch.optim.Adam(self.lvae.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        beta = DeterministicWarmup(n=100, t_max=1) # Linear warm-up from 0 to 1 over 50 epoch
        min_ppx=999999.0
        test_ppx_trend=[]
        test_kld_trend=[]
        no_decent_cnt = 0
        # If previous checkpoint files exist, load the pretrain paramter dictionary from them.
        ckpt = 0
        if os.path.exists(MODEL_SAV_PATH):
            if os.path.exists(MODEL_SAV_PATH+'current.pkl'):
                self.load_state_dict(torch.load(MODEL_SAV_PATH + 'current.pkl'))
                ckpt_list = os.listdir(MODEL_SAV_PATH)
                if len(ckpt_list)>0:
                    for ckpt_f in ckpt_list:
                        tmp=ckpt_f.split('-')
                        if len(tmp)==1:
                            continue
                        cc=tmp[1].split('.')[0]
                        if cc.isdigit():
                            current_ckpt = int(cc)
                            if current_ckpt > ckpt:
                                ckpt = current_ckpt+1
            else:
                ckpt_list = os.listdir(MODEL_SAV_PATH)
                if len(ckpt_list)>0:
                    for ckpt_f in ckpt_list:
                        cc=ckpt_f.split('-')[1].split('.')[0]
                        if cc.isdigit():
                            current_ckpt = int(cc)
                            if current_ckpt > ckpt:
                                ckpt = current_ckpt
                    self.load_state_dict(torch.load(MODEL_SAV_PATH + "model_parameters_epoch-"+str(ckpt)+".pkl"))
        else:
            os.makedirs(MODEL_SAV_PATH)

        #------------------------------------------------------------------------------------
        #Main training epoch control
        for epoch in range(ckpt, args.epoch_max):
            self.lvae.train()
            # for sub_epoch in range(alternative_epoch):
            loss_sum = 0.0
            total_rec = 0.0
            kld_sum = 0.0
            ppl_sum=0.0
            cal_ppl_sum = 0.0
            training_word_count=0
            doc_count=0
            flag=0

            for data, label, word_count in train_dataloader:
                optimizer.zero_grad()
                data = data.to(device)
                label = label.to(device)
                word_count = word_count.to(device)
                data_size=len(data)
                loss, KL, rec = self.lvae.negative_elbo_bound(data,label,dataset_flag=0,sample = False,gama=self.gama)
                dag_param = self.lvae.dag.A
                h_a = _h_A(dag_param, dag_param.size()[0])
                embedding_loss=torch.sum(nn.PairwiseDistance(p=2)(self.lvae.dec.topic_embeddings_mat,torch.mm(self.lvae.dec.beta,self.lvae.dec.word_embeddings_mat.T)))
                loss += (3*h_a + 0.5*h_a*h_a)+embedding_loss 
                loss.sum().backward()
                optimizer.step()
                loss_sum += loss.sum().item()
                kld_sum += (KL.sum()/data_size).item() 
                total_rec += rec.sum().item() 
                training_word_count += torch.sum(word_count)
                per_ppl=torch.div(rec, word_count+1e-10)
                ppl_sum += torch.sum(per_ppl).item()
                doc_count += len(data)
                m = len(train_dataloader)
            if epoch % 1 == 0:
                print(str(epoch)+' loss:'+str(loss_sum/m)+' KL:'+str(kld_sum/m)+' rec:'+str(total_rec/m)+'m:' + str(m))
            corpus_ppl = torch.exp(total_rec / training_word_count)
            per_doc_ppl= np.exp(ppl_sum /doc_count)
            kldc=torch.div(kld_sum, len(train_dataloader))
            print('| Training epoch %2d | Corpus PPX: %.5f | Per doc PPX: %.5f | KLD: %.5f' % (epoch+1, 
                corpus_ppl, 
                per_doc_ppl, 
                kldc))
            
            #Evaluating model on testset
            t_theta=[]
            with torch.no_grad():
                self.eval()
                loss_sum=0.0
                total_rec = 0.0
                ppl_sum=0.0
                kld_sum=0.0
                training_word_count=0
                doc_count=0
                for data, label, word_count in test_dataloader:
                    data = data.to(device)
                    label = torch.zeros(len(label),len(label[0])).to(device)
                    word_count = word_count.to(device)
                    data_size=len(data)
                    loss, KL, rec = self.lvae.negative_elbo_bound(data,label,dataset_flag=1,sample = False)
                    dag_param = self.lvae.dag.A
                    h_a = _h_A(dag_param, dag_param.size()[0])
                    loss += (3*h_a + 0.5*h_a*h_a)  #*data_size #- torch.norm(dag_param) 
                    loss_sum += loss.sum().item()
                    kld_sum += KL.sum().item() / len(data)
                    total_rec += rec.sum().item()
                    training_word_count += torch.sum(word_count)
                    per_ppl=torch.div(rec, word_count+1e-10)
                    ppl_sum += torch.sum(per_ppl).item()
                    doc_count += len(data)
                    mat_theta = self.lvae.dec.theta#.detach().cpu().numpy()
                    if len(t_theta)==0:
                        t_theta=mat_theta
                    else:
                        t_theta=torch.cat((t_theta,mat_theta),0)
                test_ppl = torch.exp(total_rec / training_word_count)
                per_doc_ppl= np.exp(ppl_sum / doc_count)
                kldc = torch.div(kld_sum, len(test_dataloader))
                print('===Test epoch %2d ===| Testset PPX: %.5f | Per doc PPX: %.5f | KLD: %.5f' % (epoch+1, 
                    test_ppl, 
                    per_doc_ppl, 
                    kldc))
                #Recording training statics
                curr_test_ppl = float(test_ppl.detach().cpu().numpy())
            torch.save(self.state_dict(), MODEL_SAV_PATH + "current.pkl")
            if epoch+1 > 0:
                #TODO: Serialize better model and Early stop when not consecutively decent for 30 epoch
                if min_ppx > curr_test_ppl:
                    mat_beta = self.lvae.dec.beta.detach().cpu().numpy()
                    mat_topic = self.lvae.dec.topic_embeddings_mat.detach().cpu().numpy()
                    mat_word = self.lvae.dec.word_embeddings_mat.detach().cpu().numpy()
                    mat_dagA = self.lvae.dag.A.detach().cpu().numpy()
                    t_theta=t_theta.detach().cpu().numpy()
                    topic_coherence_file_export(mat_beta, id2word, TOPN, t_theta, mat_topic, mat_word,mat_dagA)
                    no_decent_cnt = 0
                    min_ppx = curr_test_ppl
                    #Save all model parameters to designated path.
                    torch.save(self.state_dict(), MODEL_SAV_PATH + "model_parameters_epoch-"+str(epoch+1)+".pkl")
                    #Max_to_keep mechanism implementation
                    ckpt_tmp_list = os.listdir(MODEL_SAV_PATH)
                    if len(ckpt_tmp_list) > MAX_TO_KEEP :
                        os.remove(MODEL_SAV_PATH + ckpt_tmp_list[0])
                else:
                    #Early-stop
                    no_decent_cnt+=1
                    if no_decent_cnt > 30:
                        break
#==============================================================================================================
#==============================================================================================================
def main(args):
    layout = [
        ('model={:s}',  'causalvae'),
        ('run={:04d}', args.run),
        ('color=True', args.color),
        ('toy={:s}', str(args.toy))
    ]
    model_name = '_'.join([t.format(v) for (t, v) in layout])
    pprint(vars(args))
    print('Model name:', model_name)
    train_dataset = BOW_TopicModel_Corpus(vocabulary_size=args.vocab_size, data_path=args.data_path+args.fname+'.text.train.feat', label_path=args.data_path+args.label_path, label_voc_path=args.data_path+args.label_voc_path, label_num=args.topics)
    test_dataset = BOW_TopicModel_Corpus(vocabulary_size=args.vocab_size, data_path=args.data_path +args.fname+'.text.test.feat',label_path=args.data_path+args.label_path.replace('train','test'), label_voc_path=args.data_path+args.label_voc_path.replace('train','test'), label_num=args.topics)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True, 
        drop_last=False)#,
        # collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True,      
        drop_last=False)#,
        # collate_fn=collate_fn)

    #Build model
    model = CausalNTM(name=model_name, vocabulary_size=args.vocab_size, 
        n_hidden=args.hidden, 
        dropout_prob=args.dropout, 
        n_topics=args.topics, 
        topic_embeddings_size=args.topicembedsize, 
        inf_nonlinearity=args.inf_nonlinearity, 
        alternative_epoch=args.alternative_epoch,flag=args.flag,gama=args.gama,z2_dim=args.z2_dim).to(device)
    model.train_and_eval(train_loader, test_loader, batch_size=args.batch_size, learning_rate=args.lr,alternative_epoch=args.alternative_epoch)
    
if __name__ == '__main__':
    main(args)
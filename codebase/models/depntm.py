#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
import time
import torch
import numpy as np
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")



class DepNTM(nn.Module):
    def __init__(self, nn='mask', name='vae', n_topics=4,n_background_topics=4, z2_dim=16, inference = False, alpha=0.3, beta=1, n_hidden=256, vocabulary_size=2000, topic_embeddings_size=128,flag=0):
        super().__init__()
        self.flag = flag
        self.name = name
        self.n_topics = n_topics
        self.n_background_topics = n_background_topics
        self.n_total_topics = n_topics+n_background_topics
        self.n_hidden = n_hidden
        self.vocabulary_size = vocabulary_size
        self.topic_embeddings_size = topic_embeddings_size
        self.z2_dim = z2_dim
        self.channel = 4
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.n_total_topics, self.n_hidden, self.vocabulary_size,self.z2_dim,self.n_topics)
        self.dec = nn.Decoder_DAG(self.n_total_topics,self.n_topics, 0, self.vocabulary_size, topic_embeddings_size,self.flag,self.z2_dim)
        self.dag = nn.DagLayer(self.n_topics, self.n_topics, i = inference)
        self.attn = nn.Attention(self.n_topics)
        self.mask_z = nn.MaskLayer(self.n_topics,self.flag,self.z2_dim)
        self.mask_u = nn.MaskLayer(self.n_topics,self.flag,z2_dim=1)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x, label, mask = None, dataset_flag=0, sample = False, adj = None, alpha=0.3, beta=1, lambdav=0.001, gama=1):
        assert label.size()[1] == self.n_topics
        q_m_t, q_v_t = self.enc.encode(x.to(device),label.to(device))
        q_m_t, q_v_t = q_m_t.reshape([q_m_t.size()[0], self.n_total_topics,self.z2_dim]),torch.ones(q_m_t.size()[0], self.n_total_topics,self.z2_dim).to(device)

        q_m, q_v = q_m_t[:,:self.n_topics,:].to(device),q_v_t[:,:self.n_topics,:].to(device)
        decode_m, decode_v = self.dag.calculate_dag(q_m.to(device), torch.ones(q_m.size()[0], self.n_topics,self.z2_dim).to(device))
        decode_m, decode_v = decode_m.reshape([q_m.size()[0], self.n_topics,self.z2_dim]).to(device),decode_v.to(device)
        if sample == False:
          if mask != None and mask < 2:
              z_mask = torch.ones(q_m.size()[0], self.n_topics,self.z2_dim).to(device)*adj
              decode_m[:, mask, :] = z_mask[:, mask, :]
              decode_v[:, mask, :] = z_mask[:, mask, :]


          # m_zm, m_zv = self.dag.mask_z(decode_m.to(device)),decode_v
          # print(label.size())
          m_zm, m_zv = self.dag.mask_z(decode_m.to(device)).reshape([q_m.size()[0], self.n_topics,self.z2_dim]).to(device),decode_v.reshape([q_m.size()[0], self.n_topics,self.z2_dim])


          if dataset_flag==1:
            m_u = self.dag.mask_u(torch.zeros(label.size()[0],label.size()[1]).to(device))
          else:
            m_u = self.dag.mask_u(label.to(device))
          f_z = self.mask_z.mix(m_zm).reshape([q_m.size()[0], self.n_topics,self.z2_dim]).to(device)
          e_tilde = self.attn.attention(decode_m.reshape([q_m.size()[0], self.n_topics,self.z2_dim]).to(device),q_m.reshape([q_m.size()[0], self.n_topics,self.z2_dim]).to(device))[0]
          if mask != None and mask < 2:
              z_mask = torch.ones(q_m.size()[0],self.n_topics,self.z2_dim).to(device)*adj
              e_tilde[:, mask, :] = z_mask[:, mask, :]
              
          f_z1 = f_z+e_tilde
          if mask!= None and mask == 2 :
              z_mask = torch.ones(q_m.size()[0],self.n_topics,self.z2_dim).to(device)*adj
              f_z1[:, mask, :] = z_mask[:, mask, :]
              m_zv[:, mask, :] = z_mask[:, mask, :]
          if mask!= None and mask == 3 :
              z_mask = torch.ones(q_m.size()[0],self.n_topics,self.z2_dim).to(device)*adj
              f_z1[:, mask, :] = z_mask[:, mask, :]
              m_zv[:, mask, :] = z_mask[:, mask, :]
          g_u = self.mask_u.mix(m_u).to(device)
          z_given_dag = ut.conditional_sample_gaussian(f_z1, m_zv*lambdav).to(device)
        decoded_bernoulli_logits = self.dec.decode_sep(torch.cat( (z_given_dag,q_m_t[:,self.n_topics:,:]),1 ).to(device), label.to(device))
        rec = -ut.log_bernoulli_with_logits(x, decoded_bernoulli_logits.reshape(x.size()))
        p_m, p_v = torch.zeros(q_m.size()), torch.ones(q_m.size()).to(device)
        if dataset_flag==1:
            cp_m = torch.zeros(label.size()[0],label.size()[1],self.z2_dim).to(device)
        else:
            cp_m = label.view(label.size()[0],label.size()[1],1).expand(label.size()[0],label.size()[1],self.z2_dim).to(device)
        cp_v = torch.ones([q_m.size()[0],self.n_topics,self.z2_dim]).to(device)
        cp_z = ut.conditional_sample_gaussian(cp_m.to(device), cp_v.to(device))
        kl = torch.zeros(1).to(device)
        kl = alpha*ut.kl_normal(q_m.view(-1,self.n_topics*self.z2_dim).to(device), q_v.view(-1,self.n_topics*self.z2_dim).to(device), p_m.view(-1,self.n_topics*self.z2_dim).to(device), p_v.view(-1,self.n_topics*self.z2_dim).to(device))
        for i in range(self.n_topics):
            kl = kl + beta*ut.kl_normal(decode_m[:,i,:].to(device), cp_v[:,i,:].to(device),cp_m[:,i,:].to(device), cp_v[:,i,:].to(device))
        mask_kl = torch.zeros(1).to(device)
        mask_kl2 = torch.zeros(1).to(device)
        for i in range(self.n_topics):
            mask_kl = mask_kl + 1*ut.kl_normal(f_z1[:,i,:].to(device), cp_v[:,i,:].to(device),cp_m[:,i,:].to(device), cp_v[:,i,:].to(device))
        u_loss = torch.nn.MSELoss()
        mask_l = mask_kl + u_loss(g_u, label.float().to(device))
        if dataset_flag==1:
            label_cmp=0
        else:
            label_cmp=-gama*ut.log_bernoulli_with_logits(label.float().to(device),self.dec.theta[:,:self.n_topics])
        nelbo = rec + kl + mask_l + label_cmp
        return nelbo, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.n_topics),
            self.z_prior[1].expand(batch, self.n_topics))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))

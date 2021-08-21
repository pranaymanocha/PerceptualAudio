import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class base_encoder(nn.Module):
    def __init__(self,dev=torch.device('cpu'),n_layers=20,nefilters=16):
        super(base_encoder, self).__init__()
        self.dev = dev
        nlayers = n_layers
        
        self.num_layers = nlayers
        self.nefilters = nefilters
        filter_size = 15
        merge_filter_size = 5
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.ebatch = nn.ModuleList()
        self.dbatch = nn.ModuleList()
        echannelin = [1] + [(i + 1) * nefilters for i in range(nlayers-1)]
        echannelout = [(i + 1) * nefilters for i in range(nlayers)]
        
        nchan = nefilters
        for i in range(self.num_layers):
            if i==0:
                chin = 1
            else:
                chin = nchan
            if (i+1)%4==0:
                nchan = nchan*2
            self.encoder.append(nn.Conv1d(chin,nchan,filter_size,padding=filter_size//2))
            self.ebatch.append(nn.BatchNorm1d(nchan))
        
    def forward(self,x):
        input = x
        
        for i in range(self.num_layers):
            x = self.encoder[i](x)
            x = self.ebatch[i](x)
            x = F.leaky_relu(x,0.1)
            if (i+1)%4==0:
                x = x[:,:,::2]
        
        x = torch.sum(x,dim=(2))/x.shape[2] # average by channel dimension
        
        dim = 1
        acoustics, content = torch.split(x, x.size(dim) // 2, dim=dim)
        
        return x,acoustics,content

class projection_head(nn.Module):
    def __init__(self,ndim=[256,128],dp=0.1,BN=1,input_size=512):
        super(projection_head, self).__init__()
        n_layers = 2
        MLP = []
        for ilayer in range(n_layers):
            if ilayer==0:
                fin = input_size
            else:
                fin = ndim[ilayer-1]
            MLP.append(nn.Linear(fin,ndim[ilayer]))
            if BN==1 and ilayer==0: # only 1st hidden layer
                MLP.append(nn.BatchNorm1d(ndim[ilayer]))
            elif BN==2: # the two hidden layers
                MLP.append(nn.BatchNorm1d(ndim[ilayer]))
            if ilayer!=1:
                MLP.append(nn.LeakyReLU())
            if dp!=0:
                MLP.append(nn.Dropout(p=dp))
        # last linear maps to binary class probabilities ; loss includes LogSoftmax
        self.MLP = nn.Sequential(*MLP)
        
    def forward(self,dist):
        return self.MLP(dist)


class contrastive_disentanglement(nn.Module):
    def __init__(self,dev=torch.device('cpu'),encoder_layers=12,encoder_filters=24,input_size=512,proj_ndim=[512,256],proj_BN=1,proj_dp=0.25):
        super(contrastive_disentanglement, self).__init__()
        self.dev = dev
        self.base_encoder = base_encoder(n_layers=encoder_layers,nefilters=encoder_filters)
        
        self.projection_head = projection_head(ndim=proj_ndim,dp=proj_dp,BN=proj_BN,input_size=input_size)
        
    
    def forward(self,x1,x2,sim=1,normalise = 1):
                
        x1_proj,x1_acoustics,x1_content = self.base_encoder.forward(x1.unsqueeze(1))
        
        x2_proj,x2_acoustics,x2_content = self.base_encoder.forward(x2.unsqueeze(1))
        
        z1_gpu = []
        z2_gpu = []
        
        for i,condition in enumerate(sim):
            if condition.cpu().numpy()==1: #content invariance
                if normalise==1:
                    z1 = F.normalize(self.projection_head.forward(x1_acoustics[i,:].reshape([1,-1])).reshape([-1]), dim=0)
                    z2 = F.normalize(self.projection_head.forward(x2_acoustics[i,:].reshape([1,-1])).reshape([-1]), dim=0)
                else:
                    z1 = self.projection_head.forward(x1_acoustics[i,:].reshape([1,-1])).reshape([-1])
                    z2 = self.projection_head.forward(x2_acoustics[i,:].reshape([1,-1])).reshape([-1])
                    
            elif condition.cpu().numpy()==2: # acoustic invariance
                if normalise==1:
                    z1 = F.normalize(self.projection_head.forward(x1_content[i,:].reshape([1,-1])).reshape([-1]), dim=0)
                    z2 = F.normalize(self.projection_head.forward(x2_content[i,:].reshape([1,-1])).reshape([-1]), dim=0)
                else:
                    z1 = self.projection_head.forward(x1_content[i,:].reshape([1,-1])).reshape([-1])
                    z2 = self.projection_head.forward(x2_content[i,:].reshape([1,-1])).reshape([-1])
            z1_gpu.append(z1.unsqueeze(0))
            z2_gpu.append(z2.unsqueeze(0))
        
        z1 = torch.cat(z1_gpu, dim=0)
        z2 = torch.cat(z2_gpu, dim=0)
        
        return z1,z2
    
class contrastive_disentanglement_loss(nn.Module):
    
    def __init__(self,dev=torch.device('cpu'),batch_size = 16):
        super(contrastive_disentanglement_loss, self).__init__()
        
        self.dev = dev
        self.nt_xent_criterion = NTXentLoss(device = self.dev, batch_size = batch_size, use_cosine_similarity = 1)
        
    def forward(self,x1,x2,normalise = 1):
        
        loss = self.nt_xent_criterion.forward(x1, x2)
        
        return loss

    
## Taken from SimCLR https://github.com/sthalles/SimCLR/blob/master/loss/nt_xent.py
class NTXentLoss(torch.nn.Module):
    
    def __init__(self, device, batch_size, use_cosine_similarity,temperature=1):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    
    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity
    
    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs.squeeze(1), zis.squeeze(1)], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)
        
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)

def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)

class classifnet(nn.Module):
    def __init__(self,ndim=[16,6],dp=0.1,BN=1,classif_act='no'):
        # lossnet is pair of [batch,L] -> dist [batch]
        # classifnet goes dist [batch] -> pred [batch,2] == evaluate BCE with low-capacity
        super(classifnet, self).__init__()
        n_layers = 2
        MLP = []
        for ilayer in range(n_layers):
            if ilayer==0:
                fin = 1
            else:
                fin = ndim[ilayer-1]
            MLP.append(nn.Linear(fin,ndim[ilayer]))
            if BN==1 and ilayer==0: # only 1st hidden layer
                MLP.append(nn.BatchNorm1d(ndim[ilayer]))
            elif BN==2: # the two hidden layers
                MLP.append(nn.BatchNorm1d(ndim[ilayer]))
            MLP.append(nn.LeakyReLU())
            if dp!=0:
                MLP.append(nn.Dropout(p=dp))
        # last linear maps to binary class probabilities ; loss includes LogSoftmax
        MLP.append(nn.Linear(ndim[ilayer],2))
        if classif_act=='sig':
            MLP.append(nn.Sigmoid())
        if classif_act=='tanh':
            MLP.append(nn.Tanh())
        self.MLP = nn.Sequential(*MLP)
        
    def forward(self,dist):
        return self.MLP(dist.unsqueeze(1))

###############################################################################
### full model

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1 or classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight)
        # default Linear init is kaiming_uniform_ / default Conv1d init is a scaled uniform /  default BN init is constant gamma=1 and bias=0
        try:
            torch.nn.init.constant_(m.bias, 0.01)
        except:
            pass

class JNDnet(nn.Module):
    def __init__(self,dev=torch.device('cpu'),encoder_layers=12,encoder_filters=24,ndim=[16,6],classif_dp=0.1,classif_BN=0,classif_act='no',input_size=1024):
        super(JNDnet, self).__init__()
        
        self.dev = dev
        self.base_encoder = base_encoder(n_layers=encoder_layers,nefilters=encoder_filters)
        
        self.model_dist = lossnet_dfl(input_size)
        
        self.model_classif = classifnet(ndim=ndim,dp=classif_dp,BN=classif_BN,classif_act=classif_act)
        self.CE = nn.CrossEntropyLoss(reduction='mean')
        
    def forward(self,x1,x2,label,normalise = 1):
                
        x1_proj,x1_acoustics,x1_content = self.base_encoder.forward(x1.unsqueeze(1))
        
        x2_proj,x2_acoustics,x2_content = self.base_encoder.forward(x2.unsqueeze(1))
        
        if normalise==1:
            z1 = F.normalize(x1_acoustics, dim=1)
            z2 = F.normalize(x2_acoustics, dim=1)
        else:
            z1 = x1_acoustics
            z2 = x2_acoustics
        
        dist = self.model_dist.forward(z1,z2)
        pred = self.model_classif.forward(dist)
        pred = pred.squeeze(1)
        loss = self.CE(pred,label.squeeze(-1))
        class_prob = F.softmax(pred,dim=-1)
        class_pred = torch.argmax(class_prob,dim=-1)
        
        return loss,dist,class_pred,class_prob
        
    
    def grad_check(self,minibatch,optimizer,avg_channel=1):
        xref = minibatch[0].to(self.dev)
        xsample1 = minibatch[1].to(self.dev)
        labels  = minibatch[2].to(self.dev)
        
        loss,class_pred,_,_ = self.forward(xref,xsample1,labels)
        print('\nbackward on classification loss')
        optimizer.zero_grad()
        loss.backward()
        tot_grad = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                sum_abs_paramgrad = torch.sum(torch.abs(param.grad)).item()
                if sum_abs_paramgrad==0:
                    print(name,'sum_abs_paramgrad==0')
                else:
                    tot_grad += sum_abs_paramgrad
            else:
                print(name,'param.grad is None')
        print('tot_grad = ',tot_grad)
        
        norm_type = 2
        loss,class_pred,_,_ = self.forward(xref,xsample1,labels)
        optimizer.zero_grad()
        loss.backward()
        total_norm = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
            else:
                print(name,'param.grad is None')
        total_norm = total_norm ** (1. / norm_type)
        print('total_norm over all layers ==',total_norm)
        
        
class FINnet(nn.Module):
    def __init__(self,dev=torch.device('cpu'),encoder_layers=12,encoder_filters=24,ndim=[16,6],classif_dp=0.1,classif_BN=0,classif_act='no',input_size=1024,margin=0.1):
        super(FINnet, self).__init__()
        
        self.dev = dev
        self.base_encoder = base_encoder(n_layers=encoder_layers,nefilters=encoder_filters)
        
        self.model_dist = lossnet_dfl(input_size)
        
        self.model_classif = classifnet(ndim=ndim,dp=classif_dp,BN=classif_BN,classif_act=classif_act)
        self.CE = nn.CrossEntropyLoss(reduction='mean')
        self.margin_loss = torch.nn.MarginRankingLoss(margin = margin,reduction='mean')
        
    def forward(self,x1,x2,x3,labels,normalise = 1):
        
        x1_proj,x1_acoustics,x1_content = self.base_encoder.forward(x1.unsqueeze(1))
        
        x2_proj,x2_acoustics,x2_content = self.base_encoder.forward(x2.unsqueeze(1))
        
        x3_proj,x3_acoustics,x3_content = self.base_encoder.forward(x3.unsqueeze(1))
        
        if normalise==1:
            z1 = F.normalize(x1_acoustics, dim=1)
            z2 = F.normalize(x2_acoustics, dim=1)
            z3 = F.normalize(x3_acoustics, dim=1)
        else:
            z1 = x1_acoustics
            z2 = x2_acoustics
            z3 = x3_acoustics
        
        dist_sample1 = self.model_dist.forward(z1,z2)
        dist_sample2 = self.model_dist.forward(z1,z3)
        
        loss = self.margin_loss(dist_sample1,dist_sample2,labels)
        distance = torch.cat((dist_sample1.unsqueeze(-1),dist_sample2.unsqueeze(-1)), 1)
        class_pred = torch.argmin(distance,dim=-1)
        
        return loss,class_pred
        
    
    def grad_check(self,minibatch,optimizer,avg_channel=1):
        xref = minibatch[0].to(self.dev)
        xsample1 = minibatch[1].to(self.dev)
        xsample2 = minibatch[2].to(self.dev)
        labels  = minibatch[3].to(self.dev)
        
        loss,class_pred = self.forward(xref,xsample1,xsample2,labels)
        print('\nbackward on classification loss')
        optimizer.zero_grad()
        loss.backward()
        tot_grad = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                sum_abs_paramgrad = torch.sum(torch.abs(param.grad)).item()
                if sum_abs_paramgrad==0:
                    print(name,'sum_abs_paramgrad==0')
                else:
                    tot_grad += sum_abs_paramgrad
            else:
                print(name,'param.grad is None')
        print('tot_grad = ',tot_grad)
        
        norm_type = 2
        loss,class_pred = self.forward(xref,xsample1,xsample2,labels)
        optimizer.zero_grad()
        loss.backward()
        total_norm = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
            else:
                print(name,'param.grad is None')
        total_norm = total_norm ** (1. / norm_type)
        print('total_norm over all layers ==',total_norm)
        

        
class lossnet_dfl(nn.Module):
    def __init__(self,input_size):
        super(lossnet_dfl, self).__init__()
        self.convs = nn.ModuleList()
        self.chan_w = nn.ParameterList()
        for iconv in range(4):
            if iconv==0:
                conv = [nn.Linear(input_size,256),nn.LeakyReLU()]
                self.chan_w.append(nn.Parameter(torch.randn(256),requires_grad=True))
            elif iconv==1:
                conv = [nn.Linear(256,64),nn.LeakyReLU()]
                self.chan_w.append(nn.Parameter(torch.randn(64),requires_grad=True))
            elif iconv==2:
                conv = [nn.Linear(64,16),nn.LeakyReLU()]
                self.chan_w.append(nn.Parameter(torch.randn(16),requires_grad=True))
            elif iconv==3:
                conv = [nn.Linear(16,4)]
                self.chan_w.append(nn.Parameter(torch.randn(4),requires_grad=True))
            self.convs.append(nn.Sequential(*conv))
    
    def forward(self,xref,xper,avg_channel=0):
        # xref and xper are [batch,L]
        dist = 0
        for iconv in range(4):
            xref = self.convs[iconv](xref)
            xper = self.convs[iconv](xper)
            diff = (xper-xref)
            wdiff = diff*self.chan_w[iconv]
            if avg_channel==1:
                wdiff = torch.sum(torch.abs(wdiff),dim=(1))/diff.shape[1] # average by time and channel dimensions
            elif avg_channel==0:
                wdiff = torch.sum(torch.abs(wdiff),dim=(1)) # average by time
            dist = dist+wdiff
        
        return dist
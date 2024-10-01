import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import l2_reg_loss

class TwinCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(TwinCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['TwinCL'])
        self.mcl_rate = float(args['-lambda'])

        self.temp = float(args['-tau'])
        self.n_layers = int(args['-n_layer'])
        self.model = TwinCL_Encoder(self.data, self.emb_size, self.n_layers)
        self.gamma = float(args['-gamma'])
        
        self.m = float(args['-m'])
        self.key_model = TwinCL_Encoder(self.data, self.emb_size, self.n_layers)
        self.key_model.load_state_dict(self.model.state_dict())
        self.key_model.eval()
        
    def train(self):
        model = self.model.cuda()

        key_model = self.key_model.cuda()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, _ = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx]
                rec_loss, l_align, l_uniform = self.calculate_loss(user_emb, pos_item_emb)
                cl_loss = self.mcl_rate * self.cal_momentum_loss([user_idx, pos_idx], key_model)
                batch_loss = rec_loss + cl_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb)
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                # Momentum update of key encoder
                for theta_q, theta_k in zip(model.parameters(), key_model.parameters()):
                    with torch.no_grad():
                        theta_k.copy_(theta_k * self.m + theta_q * (1. - self.m))
                
                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:',
                        rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            # evaluation
            if epoch >= 0:
                self.fast_evaluation(epoch)
                
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_momentum_loss(self, idx, key_model):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.model()
        with torch.no_grad():
            key_model.eval()
            user_view_2, item_view_2 = key_model()
        
        user_cl_loss = self.InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        item_cl_loss = self.InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        
        return user_cl_loss + item_cl_loss

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()
    
    def calculate_loss(self, user_emb, item_emb):
        align = self.alignment(user_emb, item_emb)
        uniform = (self.uniformity(user_emb) + self.uniformity(item_emb)) / 2
        return align + self.gamma * uniform, align, uniform
    
    def alignment(self, x, y, alpha=2):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    def uniformity(self, x, t=2):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
    
    def InfoNCE(self, query, key, temperature, b_cos=True):
        if b_cos:
            query = F.normalize(query, dim=1)
            key = F.normalize(key, dim=1)

        pos_score = torch.sum(query * key, dim=-1)
        pos_score = torch.exp(pos_score / temperature)

        ttl_score = torch.matmul(query, key.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / (ttl_score + 1e-6))

        return torch.mean(cl_loss)
    
class TwinCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(TwinCL_Encoder, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(data.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat(
            [self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings, [self.data.user_num, self.data.item_num])
        
        return user_all_embeddings, item_all_embeddings
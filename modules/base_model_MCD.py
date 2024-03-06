import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import math
import utils.config as config
from modules.fc import FCNet
from modules.classifier import SimpleClassifier
from modules.attention import Attention, NewAttention
from modules.language_model import WordEmbedding, QuestionEmbedding
import numpy as np
from collections import Counter
from torch.autograd import Variable
import torch.nn.init as init

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, fusion, num_hid, num_class):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.weight = SimpleClassifier(num_hid, num_hid * 2, num_class, 0.5)
        #self.qweight = SimpleClassifier(num_hid, num_hid * 2, 65, 0.5)
        # self.weight = nn.Parameter(torch.FloatTensor(num_class, num_hid))
        # nn.init.xavier_normal_(self.weight)

    def forward(self, v, q):
        """
        Forward=
        v: [batch, num_objs, obj_dim]
        q: [batch_size, seq_length]
        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb, _ = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)

        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr #Final multimodal features, denoted as x in the main paper. This is the UpDn model.

        #This is the bias injecting component, as shown in subsection 3.4 of the main paper
        ce_logits = self.weight(joint_repr)
        #q_logits = self.qweight(joint_repr)
        
        return joint_repr, ce_logits, w_emb


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=config.scale, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.std = 0.1
        self.temp = config.temp

    def forward(self, input, learned_mg, m, epoch, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if self.training is False:
            return None, cosine

        #Set beta (Subsecion 3.3 in main paper
        beta_factor = epoch // 15
        beta = 1.0 - (beta_factor * 0.1)

        #Calculate the learnable instance-level margins, Subsection 3.3 in main paper
        learned_mg = torch.where(m > 1e-12, learned_mg.double(), -1000.0).float()
        margin = F.softmax(learned_mg / self.temp, dim=1)

        # Perform randomization as mentioned in Section 3 of main paper
        if config.randomization:
            m = torch.normal(mean=m, std=self.std)

        #Combine the margins, as in Subsection 3.3 of main paper.
        if config.learnable_margins:
            m[label != 0] = beta * m[label != 0] + (1 - beta) * margin[label != 0]
        m = 1 - m

        #Compute the AdaArc angular margins and the corresponding logits
        self.cos_m = torch.cos(m)
        self.sin_m = torch.sin(m)
        self.th = torch.cos(math.pi - m)
        self.mm = torch.sin(math.pi - m) * m
        # --------------------------- cos(theta) & phi(theta) ---------------------------

        # cosine = input
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = phi * self.s
        return output, cosine

class GenB(nn.Module):
    def __init__(self, num_hid, dataset):
        super(GenB, self).__init__()
        self.num_hid = num_hid
        self.w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
        self.q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
        self.v_att = NewAttention(dataset.v_dim, self.q_emb.num_hid, num_hid)
        self.q_net = FCNet([self.q_emb.num_hid, num_hid])
        self.v_net = FCNet([dataset.v_dim, num_hid])
        self.classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.generate = nn.Sequential(
            *block(num_hid//8, num_hid//4),
            *block(num_hid//4, num_hid//2),
            *block(num_hid//2, num_hid),
            nn.Linear(num_hid, num_hid*2),
            nn.ReLU(inplace=True)
            )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, v, q, v_mask, name, gen=True):
        w_emb = self.w_emb(q)
        q_emb, _ = self.q_emb(w_emb)

        if gen==True:
            if config.css:
                # if name=='vcss':
                batch_size, num_boxes = v_mask.shape
                zero_list = []
                for i in range(batch_size):
                    # 获取当前张量中为0的索引
                    current_indices = torch.nonzero(v_mask[i] == 0).squeeze()
                    # 将当前张量的索引添加到列表中
                    zero_list.append(current_indices)
                b, c, f = v.shape
                b1 = 3
                v_mask1 = torch.ones((b, c, f)).cuda()
                random_noise = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (b, b1, 128))))
                z = self.generate(random_noise.view(-1, 128)).view(b, b1, f)

                # generate from noise
                for i in range(len(zero_list)):
                    try:
                        v_mask1[i, zero_list[i], :] = z[i]
                    except:
                        print(z[i].size())

                v = v * (v_mask1)
            else:
                b, c, f = v.shape
                v_z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (b, c, 128))))
                v = self.generate(v_z.view(-1, 128)).view(b, c, f)

        att = self.v_att(v, q_emb)

        att = nn.functional.softmax(att, 1)
        v_emb = (att * v).sum(1)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)

        joint_repr = v_repr * q_repr

        logits = self.classifier(joint_repr)

        return logits


def build_baseline(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    fusion = FCNet([num_hid, num_hid*2], dropout=0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net,
                     fusion, num_hid, dataset.num_ans_candidates)


def build_baseline_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    fusion = FCNet([num_hid, num_hid*2], dropout=0.5)
    basemodel = BaseModel(w_emb, q_emb, v_att, q_net, v_net,
                     fusion, num_hid, dataset.num_ans_candidates)
    margin_model = ArcMarginProduct(num_hid, dataset.num_ans_candidates)
    return basemodel, margin_model

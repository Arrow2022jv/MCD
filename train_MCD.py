import pickle
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import utils.config as config
from torch.nn import functional as F
import numpy as np
import copy
import random
from tensorboardX import SummaryWriter
from torch.autograd import Variable

writer_tsne = SummaryWriter('runs/tsne')

def compute_supcon_loss(feats, qtype):
    tau = 1.0
    if isinstance(qtype, tuple):
      i = 0
      dic = {}
      for item in qtype:
          if item not in dic:
              dic[item] = i
              i = i + 1
      tau = 1.0
      qtype = torch.tensor([dic[item] for item in qtype]).cuda()
    feats_filt = F.normalize(feats, dim=1)
    targets_r = qtype.reshape(-1, 1)
    targets_c = qtype.reshape(1, -1)
    mask = targets_r == targets_c
    mask = mask.int().cuda()
    feats_sim = torch.exp(torch.matmul(feats_filt, feats_filt.T) / tau)
    negatives = feats_sim*(1.0 - mask)
    negative_sum = torch.sum(negatives)
    positives = torch.log(feats_sim/negative_sum)*mask
    positive_sum = torch.sum(positives)
    positive_sum = positive_sum/torch.sum(mask)

    sup_con_loss = -1*torch.mean(positive_sum)
    return sup_con_loss

def compute_acc(logits, labels):
    pred = torch.argmax(logits, dim = 1)
    pred = pred.detach().cpu().numpy()
    score = (pred == np.array(labels))
    tot_correct = score.sum()
    return tot_correct


def compute_score_with_logits(logits, labels):
    _, log_index = logits.max(dim=1, keepdim=True)
    scores = labels.gather(dim=1, index=log_index)
    return scores
    
def compute_loss(output, labels):

    #Function for calculating loss
    
    ce_loss = nn.CrossEntropyLoss(reduction='mean')(output, labels.squeeze(-1).long())
    
    return ce_loss


def saved_for_eval(dataloader, results, question_ids, answer_preds):
    """ Save as a format accepted by the evaluation server. """
    _, answer_ids = answer_preds.max(dim=1)
    answers = [dataloader.dataset.label2ans[i] for i in answer_ids]
    for q, a in zip(question_ids, answers):
        entry = {
            'question_id': q.item(),
            'answer': a,
        }
        results.append(entry)
    return results


def train(model, m_model, genb, discriminator, optim, optim_G, optim_D, train_loader, loss_fn, tracker, writer, tb_count, epoch, args):

    loader = tqdm(train_loader, ncols=0)
    loss_trk = tracker.track('loss', tracker.MovingMeanMonitor(momentum=0.99))
    acc_trk = tracker.track('acc', tracker.MovingMeanMonitor(momentum=0.99))

    genb.train(True)
    discriminator.train(True)

    kld = nn.KLDivLoss(reduction='batchmean')
    bce = nn.BCELoss()

    for v, q, a, mg, bias, q_id, f1, type, train_hint, type_mask, notype_mask, ques_mask in loader:
        # print('begin batch')
        v = Variable(v).cuda().requires_grad_()
        q = Variable(q).cuda()
        a = Variable(a).cuda()
        mg = mg.cuda()
        bias = bias.cuda()

        #--------------genb---------------#
        valid = Variable(torch.ones(v.size(0), 1)).cuda()
        fake = Variable(torch.ones(v.size(0), 1)).cuda()
        #--------------CSS----------------#
        hintscore = Variable(train_hint).cuda()
        type_mask = Variable(type_mask).float().cuda()

        optim_G.zero_grad()
        optim_D.zero_grad()
        hidden_, ce_logits, w_emb = model(v, q)
        hidden, pred = m_model(hidden_, ce_logits, mg, epoch, a)
        f1 = f1.cuda()
        dict_args = {'margin': mg, 'bias': bias, 'hidden': hidden, 'epoch': epoch, 'per': f1}
        if config.css:
            # train genb
            visual_grad = torch.autograd.grad((pred * (a > 0).float()).sum(), v, create_graph=True)[0]
            word_grad = torch.autograd.grad((pred * (a > 0).float()).sum(), w_emb, create_graph=True)[0]
            # calculate v_css
            v_mask = torch.ones(v.shape[0], 36).cuda()
            visual_grad_cam = visual_grad.sum(2)
            hint_sort, hint_ind = hintscore.sort(1, descending=True)
            v_ind = hint_ind[:, :18]
            v_grad = visual_grad_cam.gather(1, v_ind)
            v_grad_ind = v_grad.sort(1, descending=True)[1][:, :3]
            v_star = v_ind.gather(1, v_grad_ind)
            v_mask.scatter_(1, v_star, 0)
            # done

            # calculate q_css
            word_grad_cam = word_grad.sum(2)
            # word_grad_cam_sigmoid = torch.sigmoid(word_grad_cam * 1000)
            word_grad_cam_sigmoid = torch.exp(word_grad_cam * type_mask)
            word_grad_cam_sigmoid = word_grad_cam_sigmoid * type_mask
            w_ind = word_grad_cam_sigmoid.sort(1, descending=True)[1][:, :5]
            q_bias = copy.deepcopy(q)
            if config.version=='v1':
                q_bias.scatter_(1, w_ind, 18329)
            else:
                q_bias.scatter_(1, w_ind, 18455)
            # done
            pred_g1 = genb(v, q, v_mask, 'vcss', gen=True)
            pred_g2 = genb(v, q_bias, None, 'qcss',gen=False)
            alpha = 0.5  # 可以根据需要调整权重
            pred_g = alpha * pred_g1 + (1 - alpha) * pred_g2
        else:
            pred_g = genb(v, q, None, 'vcss', gen=True)

        g_loss = F.binary_cross_entropy_with_logits(pred_g, a, reduction='none').mean()
        g_loss = g_loss * a.size(1)

        vae_preds = discriminator(pred_g)
        main_preds = discriminator(hidden)
        g_distill = kld(pred_g, hidden.detach())
        dsc_loss = bce(vae_preds, valid) + bce(main_preds, valid)
        g_loss = g_loss + dsc_loss + g_distill * 5
        # g_loss = g_loss + dsc_loss
        # g_loss = g_loss + g_distill * 5
        # torch.autograd.set_detect_anomaly = True
        g_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(genb.parameters(), 0.25)
        optim_G.step()
        # done training genb

        # use genb to train the robust model
        genb.train(False)
        pred_g = genb(v, q, None, None,gen=False)
        a = torch.clamp(2 * a * torch.sigmoid(-2 * a * pred_g.detach()), 0, 1)
        gt = torch.argmax(a, 1)


        #If bias-injection or learnable margins is enabled.
        if config.learnable_margins or config.bias_inject:
            #Use cross entropy loss to train the bias-injecting module
            ce_loss = - F.log_softmax(ce_logits, dim=-1) * a
            ce_loss = ce_loss * f1
            loss = ce_loss.sum(dim=-1).mean() + loss_fn(hidden, a, **dict_args)
        else:
            loss = loss_fn(hidden, a, **dict_args)
        
        #Add the supcon loss, as mentioned in Section 3 of main paper.
        if config.supcon:
            loss = loss + compute_supcon_loss(hidden_, gt)
        writer.add_scalars('data/losses', {
        }, tb_count)
        tb_count += 1

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optim.step()
        optim.zero_grad()
        genb.train(True)
        
        #Ensemble the logit heads, as mentioned in Section 3 of the main paper, if bias-injection is enabled
        if config.bias_inject or config.learnable_margins:
          ce_logits = F.normalize(ce_logits)
          pred_l = F.normalize(pred)
          pred = (ce_logits + pred_l) / 2
        batch_score = compute_score_with_logits(pred, a.data)

        fmt = '{:.4f}'.format
        loss_trk.append(loss.item())
        acc_trk.append(batch_score.mean())
        loader.set_postfix(loss=fmt(loss_trk.mean.value),
                           acc=fmt(acc_trk.mean.value))

    return tb_count


#Evaluation code
def evaluate(model, m_model, dataloader, qid2type, epoch=0, write=False):
    score = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0
    results = []  # saving for evaluation
    for v, q, a, mg, _, q_id, _, qtype in tqdm(dataloader, ncols=0, leave=True):
        v = v.cuda()
        q = q.cuda()
        mg = mg.cuda()
        a = a.cuda()
        hidden, ce_logits, _ = model(v, q)
        hidden, pred = m_model(hidden, ce_logits, mg, epoch, a)
        
        #Ensemble the logit heads
        if config.learnable_margins or config.bias_inject:
          ce_logits = F.softmax(F.normalize(ce_logits) / config.temp, 1)
          pred_l = F.softmax(F.normalize(pred), 1)
          pred = config.alpha * pred_l + (1-config.alpha) * ce_logits
        if write:
            results = saved_for_eval(dataloader, results, q_id, pred)
        each_score = compute_score_with_logits(pred, a.cuda())
        batch_score = each_score.sum()
        score += batch_score
        qids = q_id.detach().cpu().int().numpy()
        for j in range(len(qids)):
            qid = qids[j]
            typ = qid2type[str(qid)]
            if typ == 'yes/no':
                score_yesno += each_score[j]
                total_yesno += 1
            elif typ == 'other':
                score_other += each_score[j]
                total_other += 1
            elif typ == 'number':
                score_number += each_score[j]
                total_number += 1
            else:
                print('Hahahahahahahahahahaha')

    print(score, len(dataloader.dataset))
    score = score / len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number
    
    if write:
        print("saving prediction results to disk...")
        result_file = 'vqa_{}_{}_{}_{}_results.json'.format(
            config.task, config.test_split, config.version, epoch)
        with open(result_file, 'w') as fd:
            json.dump(results, fd)
    print(score)
    return score, score_yesno, score_other, score_number


# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc" style="margin-top: 1em;"><ul class="toc-item"><li><span><a href="#imports" data-toc-modified-id="imports-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>imports</a></span></li><li><span><a href="#load-original-subsets:-train,-test-and-split-or..." data-toc-modified-id="load-original-subsets:-train,-test-and-split-or...-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>load original subsets: train, test and split or...</a></span></li><li><span><a href="#...load-my-subsets:-train,-dev,-test" data-toc-modified-id="...load-my-subsets:-train,-dev,-test-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>...load my subsets: train, dev, test</a></span></li><li><span><a href="#Neural-net-to-make-predictions-or..." data-toc-modified-id="Neural-net-to-make-predictions-or...-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Neural net to make predictions or...</a></span><ul class="toc-item"><li><span><a href="#train-th-model-or..." data-toc-modified-id="train-th-model-or...-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>train th model or...</a></span></li><li><span><a href="#...load-a-pretrained-one" data-toc-modified-id="...load-a-pretrained-one-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>...load a pretrained one</a></span></li><li><span><a href="#make-predictions" data-toc-modified-id="make-predictions-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>make predictions</a></span></li></ul></li><li><span><a href="#...load-mlp-probs-or..." data-toc-modified-id="...load-mlp-probs-or...-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>...load mlp probs or...</a></span></li><li><span><a href="#...load-LabelPowerset-probs" data-toc-modified-id="...load-LabelPowerset-probs-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>...load LabelPowerset probs</a></span></li><li><span><a href="#Static-thresholds" data-toc-modified-id="Static-thresholds-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Static thresholds</a></span></li><li><span><a href="#SGLThresh" data-toc-modified-id="SGLThresh-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>SGLThresh</a></span><ul class="toc-item"><li><span><a href="#SurrogateHeaviside-definition" data-toc-modified-id="SurrogateHeaviside-definition-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>SurrogateHeaviside definition</a></span></li><li><span><a href="#SurrogateHeaviside-definition-thresh-and-sigma-learnable" data-toc-modified-id="SurrogateHeaviside-definition-thresh-and-sigma-learnable-8.2"><span class="toc-item-num">8.2&nbsp;&nbsp;</span>SurrogateHeaviside definition thresh and sigma learnable</a></span></li><li><span><a href="#numerical-application" data-toc-modified-id="numerical-application-8.3"><span class="toc-item-num">8.3&nbsp;&nbsp;</span>numerical application</a></span></li><li><span><a href="#numerical-application,-for-loop" data-toc-modified-id="numerical-application,-for-loop-8.4"><span class="toc-item-num">8.4&nbsp;&nbsp;</span>numerical application, for loop</a></span></li></ul></li><li><span><a href="#NumThresh" data-toc-modified-id="NumThresh-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>NumThresh</a></span><ul class="toc-item"><li><span><a href="#utility-function" data-toc-modified-id="utility-function-9.1"><span class="toc-item-num">9.1&nbsp;&nbsp;</span>utility function</a></span></li><li><span><a href="#numerical-application" data-toc-modified-id="numerical-application-9.2"><span class="toc-item-num">9.2&nbsp;&nbsp;</span>numerical application</a></span></li></ul></li></ul></div>

# # imports

# In[ ]:


# paper http://lpis.csd.auth.gr/publications/tsoumakas-ismir08.pdf

# !pip install scikit-multilearn
# !pip install skorch
# !pip install liac-arff


# In[1]:
import sys
sys.path.append("sgl_utils")

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
# from skorch import NeuralNetClassifier

print(torch.__version__)

# In[2]:


import numpy as np
import sklearn.metrics as metrics
# from skmultilearn.dataset import load_dataset
import matplotlib.pyplot as plt
import time
import os

# from sklearn.model_selection import train_test_split

from sgl_utils.numThresh import *
from sgl_utils.sglThresh import *
from sgl_utils.misc import *

# In[3]:


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# # load original subsets: train, test and split or...

# In[5]:


nb_classes=6
dataset = 'emotions'
nb_runs = 10
log_dir='exp/emotions/21_03_2020/'

# ...load my subsets: train, dev, test

arr = np.load("datasets/emotions/%s_train_dev_test.npz"%dataset)
# print(arr["train_proba"])

X_train_numpy = arr["X_train_numpy"]
X_dev_numpy = arr['X_dev_numpy']
X_test_numpy = arr['X_test_numpy']
y_train_numpy = arr['y_train_numpy']
y_dev_numpy = arr['y_dev_numpy']
y_test_numpy = arr['y_test_numpy']
                
print(X_train_numpy.shape, X_dev_numpy.shape, X_test_numpy.shape, y_train_numpy.shape, y_dev_numpy.shape, y_test_numpy.shape)


# In[12]:

X_train_pth = torch.tensor(X_train_numpy, dtype=torch.float)
X_dev_pth = torch.tensor(X_dev_numpy, dtype=torch.float)
X_test_pth = torch.tensor(X_test_numpy, dtype=torch.float)

y_train_pth = torch.tensor(y_train_numpy, dtype=torch.float)
y_dev_pth = torch.tensor(y_dev_numpy, dtype=torch.float)
y_test_pth = torch.tensor(y_test_numpy, dtype=torch.float)


# Neural net to make predictions or...


input_dim = X_train_pth.size()[1]
hidden_dim = 200
# output_dim = len(np.unique(y_train.rows))
output_dim = nb_classes


train_dataset = data.TensorDataset(X_train_pth, y_train_pth) # create your datset
train_dataloader = data.DataLoader(train_dataset, batch_size=6, shuffle=True)
train_dataloader_noShuffle = data.DataLoader(train_dataset, batch_size=32, shuffle=False)

dev_dataset = data.TensorDataset(X_dev_pth, y_dev_pth) # create your datset
dev_dataloader = data.DataLoader(dev_dataset, batch_size=32, shuffle=False)

test_dataset = data.TensorDataset(X_test_pth, y_test_pth) # create your datset
test_dataloader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)


# In[16]:


class MultiClassClassifierModule(nn.Module):
    def __init__(
            self,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=0.5,
    ):
        super(MultiClassClassifierModule, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.hidden(X))
        X = self.dropout(X)
        X = self.output(X)
#         X = F.softmax(X, dim=-1)
        X = torch.sigmoid(X)

        return X



for i in range(nb_runs):

    log_dir += '%d'%i
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_fh = open("%s/log.txt"%(log_dir),"wt")
    log_fh.write("Train MLP\n")

    model = MultiClassClassifierModule(input_dim, hidden_dim, output_dim)

    criterion = torch.nn.BCELoss(reduction="mean")
    # criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # print(model)


    # train the model
    num_epochs = 10

    losses_MLP = []
    for epoch in range(num_epochs):

        debut = time.time()
        model.train()

        for i_batch, sample_batched in enumerate(train_dataloader):

            X_batch, y_batch = sample_batched

    #         print(i_batch, X_batch.size(), y_batch.size())

            # Forward pass
            # inputs:  predictions_tensor
            outputs = model(X_batch)
    #         print(outputs.size())
        #     if epoch % 10 == 0:
        #         print(outputs[-1])
            loss = criterion(outputs, y_batch)

            # Backward and optimize
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            losses_MLP.append(loss)

        duree_epoch = time.time() - debut

        print ('Epoch [{}/{}], Loss: {:.4f}, Duration: {:.1f} s'
               .format(epoch+1, num_epochs, loss, duree_epoch))
        log_fh.write(' Epoch [{}/{}], Loss: {:.4f}, Duration: {:.1f} s\n'
               .format(epoch+1, num_epochs, loss, duree_epoch))


    plt.plot(losses_MLP)
    plt.savefig("%s/losses_MLP.eps"%(log_dir))


    # torch.save(model.state_dict(), "exp/emotions/21_03_2020/mlp_%s.pth"%dataset)


    # ## ...load a pretrained one

    # In[51]:

    #
    # model.load_state_dict(torch.load("exp/emotions/21_03_2020/mlp_%s.pth"%dataset))
    # model.eval()


    # ## make predictions

    # In[20]:


    train_outputs = predict(train_dataloader_noShuffle, y_train_pth, model)
    dev_outputs = predict(dev_dataloader, y_dev_pth, model)
    test_outputs = predict(test_dataloader, y_test_pth, model)

    train_outputs_numpy = train_outputs.clone().detach().cpu().numpy()
    dev_outputs_numpy = dev_outputs.clone().detach().cpu().numpy()
    test_outputs_numpy = test_outputs.clone().detach().cpu().numpy()
    # y_test[:5], test_outputs[:5]


    # In[37]:


    # # !mkdir exp/emotions/21_03_2020
    # np.savez("exp/emotions/21_03_2020/mlp_probs_%s_train_dev_test.npz"%dataset,
    #          train_outputs_numpy=train_outputs_numpy,
    #          dev_outputs_numpy = dev_outputs_numpy,
    #          test_outputs_numpy=test_outputs_numpy)


    # arr = np.load("exp/emotions/mlp/%s_MLP_train_test_proba.npz"%dataset)
    # # print(arr["train_proba"])

    # train_outputs_numpy = arr["train_proba"]
    # test_outputs_numpy = arr["test_proba"]
    # train_outputs_numpy.shape, test_outputs_numpy.shape

    # Static thresholds

    static_thresh = 0.3
    train_pred = train_outputs_numpy>static_thresh
    dev_pred = dev_outputs_numpy>static_thresh
    test_pred = test_outputs_numpy>static_thresh

    log_fh.write("1 - Static thresholds\n")
    print("train"); log_fh.write("train\n")
    print_scores(y_train_numpy, train_pred)
    print_scores_fh(y_train_numpy, train_pred, log_fh)
    print("dev"); log_fh.write("dev\n")
    print_scores(y_dev_numpy, dev_pred)
    print_scores_fh(y_dev_numpy, dev_pred, log_fh)
    print("test"); log_fh.write("test\n")
    print_scores(y_test_numpy, test_pred)
    print_scores_fh(y_test_numpy, test_pred, log_fh)
    # compute_accuracy_from_numpy_tensors(y_test_numpy, test_pred)

    # numThresh
    log_fh.write("\n\n2 - numThresh\n\n")

    thresh = [static_thresh]*nb_classes

    average = 'micro'

    manual_thres_f1 = calculate_f1(y_dev_numpy, dev_outputs_numpy, thresholds=thresh, average=average)
    # print_scores(y_test_numpy, test_pred)

    # Optimize thresholds
    # (auto_thres_f1, auto_thresholds, metric_asfo_epoch) = optimize_at_with_gd(y_train_numpy, train_outputs_numpy,
    #                                                                           thresh, average=average)
    (auto_thres_f1, auto_thresholds, metric_asfo_epoch) = optimize_at_with_gd(y_dev_numpy, dev_outputs_numpy,
                                                                              thresh, average=average)

    print_thresholds(auto_thresholds, nb_classes)
    # print("%.3f %.3f"%(manual_thres_f1*100, fscore2*100))

    print('dev manual_thres f1: {:.3f}'.format(manual_thres_f1))
    print('dev auto_thres f1: {:.3f}'.format(auto_thres_f1))
    log_fh.write('dev manual_thres f1: {:.3f}\n'.format(manual_thres_f1))
    log_fh.write('dev auto_thres f1: {:.3f}\n'.format(auto_thres_f1))
    # In[47]:


    # fontsize=14
    # plt.plot(metric_asfo_epoch)
    # plt.xlabel("epochs", fontsize=fontsize)
    # plt.ylabel("F1", fontsize=fontsize)
    # plt.xticks(fontsize=fontsize)
    # plt.yticks(fontsize=fontsize)
    # plt.savefig("datasets/emotions/emotions_F1_asof_epochs_myNetwork.png")


    # In[48]:
    print_thresholds(auto_thresholds, nb_classes)
    print_thresholds_fh(auto_thresholds, nb_classes, log_fh)

    train_pred = train_outputs_numpy>auto_thresholds
    dev_pred = dev_outputs_numpy>auto_thresholds
    test_pred = test_outputs_numpy>auto_thresholds

    # compute_accuracy_from_numpy_tensors(y_test_numpy, test_pred)
    print("train"); log_fh.write("train\n")
    print_scores(y_train_numpy, train_pred)
    print_scores_fh(y_train_numpy, train_pred, log_fh)
    print("dev"); log_fh.write("dev\n")
    print_scores(y_dev_numpy, dev_pred)
    print_scores_fh(y_dev_numpy, dev_pred, log_fh)
    print("test"); log_fh.write("test\n")
    print_scores(y_test_numpy, test_pred)
    print_scores_fh(y_test_numpy, test_pred, log_fh)


    # SGLThresh

    pth_train_probs = torch.tensor(train_outputs_numpy, dtype=torch.float).to(device)
    pth_dev_probs = torch.tensor(dev_outputs_numpy, dtype=torch.float).to(device)
    pth_test_probs = torch.tensor(test_outputs_numpy, dtype=torch.float).to(device)


    pth_train_gt = y_train_pth.to(device, dtype=torch.float)
    pth_dev_gt = y_dev_pth.to(device, dtype=torch.float)
    pth_test_gt = y_test_pth.to(device, dtype=torch.float)

    # from torch.optim.lr_scheduler import MultiStepLR


    # In[66]:

    sigma_init=40.
    # criterion = torch.nn.BCELoss(reduction="mean")
    criterion = F1_loss_objective

    learning_rate = 1e-3
    num_epochs = 200
    # num_epochs = 12

    for ind_optim in range(2):

        THRESHmodel = ThresholdModel(threshold_fn=threshold_fn, t=static_thresh, sigma=sigma_init, nb_classes=nb_classes)
        THRESHmodel = THRESHmodel.to(device, dtype=torch.float)

        # THRESHoptimizer = torch.optim.Adam(THRESHmodel.parameters(), lr=learning_rate)
        # scheduler = MultiStepLR(THRESHoptimizer, milestones=[180], gamma=0.1)

        if ind_optim == 0:
            log_fh.write("\n\n3 - SGL with thresh learnable\n\n")
            THRESHoptimizer = torch.optim.Adam([
                        {'params': THRESHmodel.thresh}
                    ], lr=learning_rate)
        else:
            log_fh.write("\n\n4 - SGL with thresh and sigma learnable\n\n")
            THRESHoptimizer = torch.optim.Adam([
                        {'params': THRESHmodel.thresh},
                        {'params': THRESHmodel.sigma, 'lr': 1.}
                    ], lr=learning_rate)

        # Train the model, in batch mode

        cumul_delta_thresh = torch.zeros(nb_classes,)
        delta_thresh = torch.zeros(nb_classes,)

        for el in THRESHmodel.parameters():
            PREC_learned_AT_thresholds = el.clone().detach().cpu()

        losses = []
        for epoch in range(num_epochs):

            debut = time.time()

            THRESHmodel.train()

            # Forward pass
            # inputs:  predictions_tensor
        #     outputs = THRESHmodel(pth_train_probs)
            outputs = THRESHmodel(pth_dev_probs)

        #     if epoch % 10 == 0:
        #         print(outputs[-1])
        #     loss = criterion(outputs, pth_train_gt)
            loss = criterion(outputs, pth_dev_gt)

            # Backward and optimize
            THRESHoptimizer.zero_grad()

            loss.backward()
        #     loss.mean().backward()
        #         loss.backward(at_batch_y)
            # loss.backward(torch.ones_like(loss))

        #     scheduler.step()

            THRESHoptimizer.step()
            # THRESHmodel.clamp()
            losses.append(loss)

            duree_epoch = time.time() - debut

        #     print ('Epoch [{}/{}], Loss: {:.4f}, Duration: {:.1f} s'
        #            .format(epoch+1, num_epochs, loss.mean(), duree_epoch))
            print ('Epoch [{}/{}], Loss: {:.4f}, Duration: {:.1f} s'
                   .format(epoch+1, num_epochs, loss, duree_epoch))

            for el in THRESHmodel.parameters():
                learned_AT_thresholds = el.clone().detach().cpu()

            delta_thresh = learned_AT_thresholds - PREC_learned_AT_thresholds
            cumul_delta_thresh += delta_thresh
            PREC_learned_AT_thresholds = learned_AT_thresholds
            if epoch % 50 == 0: print('threshs:', learned_AT_thresholds)
            # if torch.sum(delta_thresh) < 0.01: break

        print('delta:', cumul_delta_thresh)
        # plt.figure(figsize=(8,6))
        # plt.plot(losses)

        learned_AT_thresholds=THRESHmodel.thresh.clone().detach().cpu().numpy()
        sigma = THRESHmodel.sigma.clone().detach().cpu().numpy()
        print(learned_AT_thresholds, sigma)


        print_thresholds(learned_AT_thresholds, nb_classes)
        print_thresholds_fh(learned_AT_thresholds, nb_classes, log_fh)
        train_pred = train_outputs_numpy>learned_AT_thresholds
        dev_pred = dev_outputs_numpy>learned_AT_thresholds
        test_pred = test_outputs_numpy>learned_AT_thresholds

        log_fh.write("1 - Static thresholds\n")
        print("train"); log_fh.write("train\n")
        print_scores(y_train_numpy, train_pred)
        print_scores_fh(y_train_numpy, train_pred, log_fh)
        print("dev"); log_fh.write("dev\n")
        print_scores(y_dev_numpy, dev_pred)
        print_scores_fh(y_dev_numpy, dev_pred, log_fh)
        print("test"); log_fh.write("test\n")
        print_scores(y_test_numpy, test_pred)
        print_scores_fh(y_test_numpy, test_pred, log_fh)

        # In[86]:
        if ind_optim == 0:
            sgl_loss_dev_thresh_list = [-1*el.clone().detach().cpu().numpy() for el in losses]
        else:
            sgl_loss_dev_threshANDsigma_list = [-1*el.clone().detach().cpu().numpy() for el in losses]


    fontsize=14
    linewidth=2
    plt.figure(figsize=(8,6))
    plt.plot(metric_asfo_epoch, ':k', linewidth=linewidth, label="numThresh")
    plt.plot(sgl_loss_dev_thresh_list, '--b', linewidth=linewidth, label="SGLThresh t")
    plt.plot(sgl_loss_dev_threshANDsigma_list, 'g', linewidth=linewidth, label="SGLThresh t+a")
    plt.xlabel("epochs", fontsize=fontsize)
    plt.ylabel("F1", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='center right')
    plt.savefig("%s/emotions_F1_numThresh_SGL_DEV_asof_epochs.png"%(log_dir))
    plt.savefig("%s/emotions_F1_numThresh_SGL_DEV_asof_epochs.eps"%(log_dir))


    # In[57]:
    np.savez("%s/emotions_F1_numThresh_SGL_DEV_asof_epochs.npz"%(log_dir),
             sgl_loss_dev_threshANDsigma_list=np.array(sgl_loss_dev_threshANDsigma_list),
             metric_asfo_epoch = np.array(metric_asfo_epoch)
            )


    # train_pred = train_outputs_numpy>learned_AT_thresholds
    # test_pred = test_outputs_numpy>learned_AT_thresholds
    # print("train")
    # print_scores(y_train, train_pred)
    # # compute_accuracy_from_numpy_tensors(y_train_numpy, train_pred)
    # print("test")
    # print_scores(y_test, test_pred)
    # # compute_accuracy_from_numpy_tensors(y_test_numpy, test_pred)


    # # ## numerical application, for loop
    #
    # # In[202]:
    #
    #
    # criterion = F1_loss_objective
    # # fh = open("datasets/emotions/sglthresh_emotions_F1obj_micro_sigma_LabelPowerset.txt","wt")
    # fh = open("datasets/emotions/sglthresh_emotions_F1obj_micro_sigma_myNetwork.txt","wt")
    #
    # # criterion = torch.nn.BCELoss(reduction="mean")
    # # fh = open("datasets/emotions/sglthresh_emotions_BCEobj_micro_sigma.txt","wt")
    #
    # learning_rate = 1e-3
    # num_epochs = 200
    # fh.write("a,f1train,ptrain,rtrain,f1test,ptest,rtest\n")
    # metrics_list = []
    # scale_param_values_list = []
    #
    # scale_list = range(1, 150, 10) # F1
    # # scale_list = range(1, 80, 5) # BCE
    #
    # for sigma_value in scale_list:
    #
    #     sigma_value = float(sigma_value)
    #     if sigma_value>1: sigma_value-=1.
    #     scale_param_values_list.append(sigma_value)
    #
    #     print("sigma:", sigma_value)
    #
    #     THRESHmodel = ThresholdModel(threshold_fn=threshold_fn, t=0.3, nb_classes=nb_classes)
    #     THRESHmodel = THRESHmodel.to(device, dtype=torch.float)
    #     # criterion = torch.nn.BCELoss(reduction="mean")
    #     THRESHoptimizer = torch.optim.Adam(THRESHmodel.parameters(), lr=learning_rate)
    #
    #
    #     sigma = torch.nn.Parameter(torch.tensor(sigma_value), requires_grad=True)
    #
    #     cumul_delta_thresh = torch.zeros(nb_classes,)
    #     delta_thresh = torch.zeros(nb_classes,)
    #
    #     for el in THRESHmodel.parameters():
    #         PREC_learned_AT_thresholds = el.clone().detach().cpu()
    #
    #     losses = []
    #     for epoch in range(num_epochs):
    #
    #         debut = time.time()
    #
    #         THRESHmodel.train()
    #
    #         # Forward pass
    #         # inputs:  predictions_tensor
    #         outputs = THRESHmodel(pth_train_probs, sigma)
    #
    #     #     if epoch % 10 == 0:
    #     #         print(outputs[-1])
    #         loss = criterion(outputs, pth_train_gt)
    #
    #         # Backward and optimize
    #         THRESHoptimizer.zero_grad()
    #
    #         loss.backward()
    #     #     loss.mean().backward()
    #     #         loss.backward(at_batch_y)
    #         # loss.backward(torch.ones_like(loss))
    #
    #         THRESHoptimizer.step()
    #         # THRESHmodel.clamp()
    #         losses.append(loss)
    #
    #         duree_epoch = time.time() - debut
    #
    # #         print ('Epoch [{}/{}], Loss: {:.4f}, Duration: {:.1f} s'
    # #                .format(epoch+1, num_epochs, loss, duree_epoch))
    #
    #         for el in THRESHmodel.parameters():
    #             learned_AT_thresholds = el.clone().detach().cpu()
    #
    #         delta_thresh = learned_AT_thresholds - PREC_learned_AT_thresholds
    #         cumul_delta_thresh += delta_thresh
    #         PREC_learned_AT_thresholds = learned_AT_thresholds
    # #         if epoch % 10 == 0: print('threshs:', learned_AT_thresholds)
    #         # if torch.sum(delta_thresh) < 0.01: break
    #
    # #     print('delta:', cumul_delta_thresh)
    #     plt.figure(figsize=(8,6))
    #     plt.plot(losses)
    #
    #     for el in THRESHmodel.parameters():
    #         learned_AT_thresholds = el.detach().cpu().numpy()
    #
    #     train_pred = train_outputs_numpy>learned_AT_thresholds
    #     test_pred = test_outputs_numpy>learned_AT_thresholds
    #     print("train")
    #     print_scores(y_train, train_pred)
    #     # compute_accuracy_from_numpy_tensors(y_train_numpy, train_pred)
    #     print("test")
    #     print_scores(y_test, test_pred)
    #
    #     p2, r2, fscore2, support = precision_recall_fscore_support(y_train, train_pred, pos_label=1, average='micro')
    # #     p3, r3, fscore3, support = precision_recall_fscore_support(gt_y, preds, pos_label=1, average='macro')
    #     p2te, r2te, fscore2te, _ = precision_recall_fscore_support(y_test, test_pred, pos_label=1, average='micro')
    #
    #     fh.write("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n"%(sigma_value, 100.*fscore2, 100.*p2, 100.*r2,
    #                                                    100.*fscore2te, 100.*p2te, 100.*r2te))
    #     metrics_list.append([p2, r2, fscore2, p2te, r2te, fscore2te])
    #
    # fh.close()


    # In[203]:


    # def plot_influence_de_a(scale_param_values, ptest, rtest, f1test, peval, reval, f1eval):
    #
    #     fontsize=12
    #     fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12,8))
    #
    #     xlimsup=140
    # #     xlimsup=30
    #
    #     ax1.plot(scale_param_values, ptest, '-+', label='precision - SGLThresh')
    #     ax1.plot(scale_param_values, rtest, label='recall - SGLThresh')
    #     ax1.plot(scale_param_values, f1test, '-o', label='F1 - SGLThresh')
    # #     ax1.plot([1, xlimsup], [0.5659, 0.5659], '-.', label='F1 - static')
    # #     ax1.plot([1, xlimsup], [0.6118, 0.6118], '--', label='F1 - numThresh')
    #     ax1.plot([1, xlimsup], [0.5678, 0.5678], '-.', label='F1 - static')
    #     ax1.plot([1, xlimsup], [0.6128, 0.6128], '--', label='F1 - numThresh')
    #     ax1.legend(fontsize=fontsize)
    #     ax1.tick_params(axis='x', labelsize=fontsize)
    #     ax1.tick_params(axis='y', labelsize=fontsize)
    #     ax1.set_xlabel('Sigmoid scale parameter',fontsize=fontsize)
    # #     ax1.set_ylim([0.52, 0.66])
    #     ax1.set_title("Train subset",fontsize=14)
    #
    #     ax2.plot(scale_param_values, peval, '-+', label='precision - SGLThresh')
    #     ax2.plot(scale_param_values, reval, '-', label='recall - SGLThresh')
    #     ax2.plot(scale_param_values, f1eval, '-o', label='F1 - SGLThresh')
    # #     ax2.plot([1, xlimsup], [0.5698, 0.5698], '-.', label='F1 - static')
    # #     ax2.plot([1, xlimsup], [0.6024, 0.6024], '--', label='F1 - numThresh')
    #     ax2.plot([1, xlimsup], [0.5509, 0.5509], '-.', label='F1 - static')
    #     ax2.plot([1, xlimsup], [0.5828, 0.5828], '--', label='F1 - numThresh')
    # #     ax2.legend(fontsize=fontsize)
    #     ax2.tick_params(axis='x', labelsize=fontsize)
    #     ax2.set_xlabel("Sigmoid scale parameter",fontsize=fontsize)
    #     ax2.set_title("Test subset",fontsize=14)
    # #     plt.suptitle("BCE objective",fontsize=16)
    # #     plt.savefig("influence_du_sigmoid_scale_param_objective_BCE.png")
    # #     plt.suptitle("F1 objective",fontsize=16)
    # #     plt.savefig("datasets/emotions/influence_du_sigmoid_scale_param_objective_F1_LabelPowerset.png")
    # #     plt.savefig("datasets/emotions/influence_du_sigmoid_scale_param_objective_F1_myNetwork.png")
    #
    # metrics_array = np.array(metrics_list)
    # scale_param_values = np.array(scale_param_values_list)
    # # scale_param_values = np.array(range(1, 80, 5))
    # plot_influence_de_a(scale_param_values, metrics_array[:,0], metrics_array[:,1], metrics_array[:,2], metrics_array[:,3], metrics_array[:,4], metrics_array[:,5])


    # In[153]:


    # metrics_list = np.array(metrics_list)


    # # NumThresh

    # In[31]:



    # ## numerical application

    # In[46]:



    #
    #
    # # In[83]:
    #
    #

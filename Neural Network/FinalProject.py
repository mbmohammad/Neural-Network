# Neural Network : [The underlying mechanisms of alignment in error backpropagation through arbitrary weights]

## Advance Neuroscience - Final Project

### Kimia Hajisadeghian
### Arya Koureshi
### Mohammad Mohammad Beigi

#### Department EE, Sharif University of Technology

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import gc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import torch
from torchvision import transforms as trans
import torchvision.transforms.functional as Func
import pandas as pd

def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  # device = "cpu"
  if device != "cuda":
    print("GPU is not enabled in this notebook. \n"
          "If you want to enable it, in the menu under `Runtime` -> \n"
          "`Hardware accelerator.` and select `GPU` from the dropdown menu")
  else:
    print("GPU is enabled in this notebook. \n"
          "If you want to disable it, in the menu under `Runtime` -> \n"
          "`Hardware accelerator.` and select `None` from the dropdown menu")

  return device
DEVICE = set_device()
torch.cuda.get_device_name(0)

def ReLU(x):
    return torch.fmax(x, torch.zeros(x.shape, device=DEVICE))

def ReLUdr(x):
    output = torch.zeros(x.shape, device=DEVICE)
    output[torch.where(x >= 0)] = 1
    return output

def angleFunc(A, B):
    num = torch.trace(torch.matmul(torch.conj(A.T), B))
    norm_A = math.sqrt(torch.trace(torch.matmul(A,torch.conj(A.T))))
    norm_B = math.sqrt(torch.trace(torch.matmul(B,torch.conj(B.T))))
    denum = norm_A*norm_B
    angle = torch.arccos(num/denum)*180/math.pi
    return angle

def nhot_coder(x,n,hot_labels,output_neuron_num):
    labels = x.cpu().numpy()
    category_num = 10
    if(category_num * n <= output_neuron_num):
      output = torch.zeros((len(x),output_neuron_num), device=DEVICE)
      output[np.where(labels == 0),:] = hot_labels[0,:]
      output[np.where(labels == 1),:] = hot_labels[1,:]
      output[np.where(labels == 2),:] = hot_labels[2,:]
      output[np.where(labels == 3),:] = hot_labels[3,:]
      output[np.where(labels == 4),:] = hot_labels[4,:]
      output[np.where(labels == 5),:] = hot_labels[5,:]
      output[np.where(labels == 6),:] = hot_labels[6,:]
      output[np.where(labels == 7),:] = hot_labels[7,:]
      output[np.where(labels == 8),:] = hot_labels[8,:]
      output[np.where(labels == 9),:] = hot_labels[9,:]
      return output

def lossFunc(outL, labels):
    return (torch.sum(labels - outL**2))/2

def tanhReLUdr(x):
    non_negs = torch.where(x > 0)
    output = torch.zeros(x.shape, device=DEVICE)
    output[non_negs] = 1-torch.tanh(x[non_negs])**2
    return output

def binarizing(output, n, categories_nhot):
    binarized = torch.zeros((output.shape[0], output.shape[1]), device=DEVICE)
    _, closest_label = torch.max(torch.matmul(output, categories_nhot.T), dim=1)
    binarized = categories_nhot[closest_label, :]
    return binarized

def accFunc(output,label):
    acc = torch.sum(torch.eq(torch.sum(torch.eq(output,label),dim=1), output.shape[1]*torch.ones((output.shape[0]), device=DEVICE)))/len(output)
    return acc * 100

def movingAverage(arr, window_size):
    numbers_series = pd.Series(arr)
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    final_list = moving_averages_list[window_size - 1:]
    return final_list

# Autocorrelated
mu = 0
sigma = 1
lr = 0.001
batch_size = 64
iterations = 1500
epochs = 100
angles1 = np.zeros((epochs, iterations))

for ep in range(epochs):
    inputL = np.random.normal(loc=mu, scale=sigma, size=(batch_size, 20))
    fw0 = np.random.normal(loc=mu, scale=sigma, size=(20, 100))
    fw1 = np.random.normal(loc=mu, scale=sigma, size=(100, 20))
    bw1 = np.random.normal(loc=mu, scale=sigma, size=(20, 100))
    ed2 = np.random.normal(loc=mu, scale=sigma, size=(batch_size, 20))
    ed1 = np.matmul(ed2, bw1)

    for iter in range(iterations):
        num = np.trace(np.matmul(np.conj(fw1.T), bw1.T))
        norm_A = math.sqrt(np.trace(np.matmul(fw1, np.conj(fw1.T))))
        norm_B = math.sqrt(np.trace(np.matmul(bw1.T, np.conj(bw1.T.T))))
        denum = norm_A * norm_B
        angles1[ep, iter] = math.acos(num / denum) * 180 / math.pi
        hl1 = np.maximum(np.matmul(inputL, fw0), np.zeros_like(np.matmul(inputL, fw0)))
        temp = np.zeros_like(np.matmul(inputL, fw0))
        temp[np.where(np.matmul(inputL, fw0) >= 0)] = 1
        fw0 += lr * np.matmul(inputL.T, ed1 * temp)
        fw1 += lr * np.matmul(hl1.T, ed2)
    print("Epoch: {} / {}".format(ep+1, epochs), end='\r')

plt.figure()
plt.plot(np.arange(1, iterations+1), np.mean(angles1, axis=0), 'b', lw=1.5)
plt.fill_between(np.arange(1, iterations+1),
                 np.mean(angles1, axis=0) - np.std(angles1, axis=0),
                 np.mean(angles1, axis=0) + np.std(angles1, axis=0),
                 color='b', alpha=0.1)
plt.ylim([0, 100])
plt.title('Autocorrelated')
plt.ylabel('Angle')
plt.xlabel('Iteration')
plt.savefig("Autocorrelated.png", format="png")

# Cross-correlated
mu = 0
sigma = 1
lr = 0.001
batch_size = 64
iterations = 1500
epochs = 100
angles2 = np.zeros((epochs, iterations))

for ep in range (epochs):
    inputL = np.random.normal(loc=mu, scale=sigma, size=(batch_size, 20))
    fw0 = np.random.normal(loc=mu, scale=sigma, size=(20, 100))
    fw1 = np.random.normal(loc=mu, scale=sigma, size=(100, 20))
    bw1 = np.random.normal(loc=mu, scale=sigma, size=(20, 100))
    ed2 = np.copy(inputL)
    ed1 = np.matmul(ed2, bw1)

    for iter in range (iterations):
        num = np.trace(np.matmul(np.conj(fw1.T), bw1.T))
        norm_A = math.sqrt(np.trace(np.matmul(fw1, np.conj(fw1.T))))
        norm_B = math.sqrt(np.trace(np.matmul(bw1.T, np.conj(bw1.T.T))))
        denum = norm_A * norm_B
        angles2[ep, iter] = math.acos(num / denum) * 180 / math.pi
        hl1 = np.maximum(np.matmul(inputL, fw0), np.zeros_like(np.matmul(inputL, fw0)))
        temp = np.zeros_like(np.matmul(hl1, fw1))
        temp[np.where(np.matmul(hl1, fw1) >= 0)] = 1
        delta2 = np.copy(ed2 * temp)
        temp = np.zeros_like(np.matmul(inputL, fw0))
        temp[np.where(np.matmul(inputL, fw0) >= 0)] = 1
        fw0 += lr * np.matmul(inputL.T, ed1 * temp)
        fw1 += lr * np.matmul(hl1.T, delta2)

        inputL = np.random.normal(loc=mu, scale=sigma, size=(batch_size, 20))
        ed2 = np.copy(inputL)
        temp = np.zeros_like(np.matmul(hl1, fw1))
        temp[np.where(np.matmul(hl1, fw1) >= 0)] = 1
        ed1 = np.matmul(ed2 * temp, bw1)
    print("Epoch: {} / {}".format(ep+1, epochs), end='\r')

plt.figure()
plt.plot(np.arange(1, iterations+1), np.mean(angles2, axis=0), 'g', lw=1.5)
plt.fill_between(np.arange(1, iterations+1),
                 np.mean(angles2, axis=0) - np.std(angles2, axis=0),
                 np.mean(angles2, axis=0) + np.std(angles2, axis=0),
                 color='g', alpha=0.1)
plt.ylim([0, 100])
plt.title('Cross-correlated')
plt.ylabel('Angle')
plt.xlabel('Iteration')
plt.savefig("Cross-correlated.png", format="png")

# Uncorrelated
mu = 0
sigma = 1
lr = 0.001
batch_size = 64
iterations = 1500
epochs = 100
angles3 = np.zeros((epochs, iterations))

for ep in range (epochs):
    inputL = np.random.normal(loc=mu, scale=sigma, size=(batch_size, 20))
    fw0 = np.random.normal(loc=mu, scale=sigma, size=(20, 100))
    fw1 = np.random.normal(loc=mu, scale=sigma, size=(100, 20))
    bw1 = np.random.normal(loc=mu, scale=sigma, size=(20, 100))
    ed2 = np.random.normal(loc=mu, scale=sigma, size=(batch_size, 20))
    ed1 = np.matmul(ed2, bw1)

    for iter in range (iterations):
        num = np.trace(np.matmul(np.conj(fw1.T), bw1.T))
        norm_A = math.sqrt(np.trace(np.matmul(fw1, np.conj(fw1.T))))
        norm_B = math.sqrt(np.trace(np.matmul(bw1.T, np.conj(bw1.T.T))))
        denum = norm_A * norm_B
        angles3[ep, iter] = math.acos(num / denum) * 180 / math.pi
        hl1 = np.maximum(np.matmul(fw0.T, inputL.T).T, np.zeros_like(np.matmul(fw0.T, inputL.T).T))
        temp = np.zeros_like(np.matmul(fw0.T, inputL.T).T)
        temp[np.where(np.matmul(fw0.T, inputL.T).T >= 0)] = 1
        fw0 += lr * np.matmul(inputL.T, ed1 * temp)
        fw1 += lr * np.matmul(hl1.T, ed2)

        inputL = np.random.normal(loc=mu, scale=sigma, size=(batch_size, 20))
        ed2 = np.random.normal(loc=mu, scale=sigma, size=(batch_size, 20))
        ed1 = np.matmul(ed2, bw1)
    print("Epoch: {} / {}".format(ep+1, epochs), end='\r')

plt.figure()
plt.plot(np.arange(1, iterations+1), np.mean(angles3, axis=0), 'orange', lw=1.5)
plt.fill_between(np.arange(1, iterations+1),
                 np.mean(angles3, axis=0) - np.std(angles3, axis=0),
                 np.mean(angles3, axis=0) + np.std(angles3, axis=0),
                 color='orange', alpha=0.1)
plt.ylim([0, 100])
plt.title('Uncorrelated')
plt.ylabel('Angle')
plt.xlabel('Iteration')
plt.savefig("Uncorrelated.png", format="png")

plt.figure()
plt.plot(np.arange(1, iterations+1), np.mean(angles1, axis=0), 'b', lw=1.5, label="Autocorrelated")
plt.fill_between(np.arange(1, iterations+1),
                 np.mean(angles1, axis=0) - np.std(angles1, axis=0),
                 np.mean(angles1, axis=0) + np.std(angles1, axis=0),
                 color='b', alpha=0.1)

plt.plot(np.arange(1, iterations+1), np.mean(angles2, axis=0), 'g', lw=1.5, label="Cross-correlated")
plt.fill_between(np.arange(1, iterations+1),
                 np.mean(angles2, axis=0) - np.std(angles2, axis=0),
                 np.mean(angles2, axis=0) + np.std(angles2, axis=0),
                 color='g', alpha=0.1)

plt.plot(np.arange(1, iterations+1), np.mean(angles3, axis=0), 'orange', lw=1.5, label="Uncorrelated")
plt.fill_between(np.arange(1, iterations+1),
                 np.mean(angles3, axis=0) - np.std(angles3, axis=0),
                 np.mean(angles3, axis=0) + np.std(angles3, axis=0),
                 color='orange', alpha=0.1)

plt.ylim([0, 100])
plt.ylabel('Angle')
plt.xlabel('Iteration')
plt.legend()
plt.savefig("All.png", format="png")

(xTrain, yTrain), (_, _) = mnist.load_data()
xTrain = torch.from_numpy(xTrain)
yTrain = torch.from_numpy(yTrain)
xTrainR = torch.zeros((len(xTrain), 15, 15), device=DEVICE)
resize_trans = trans.Resize(15)

for i in range (len(xTrain)):
    xTrainR[i, :, :] = resize_trans(xTrain[i, :, :].unsqueeze(0))

xTrainN = xTrainR.reshape(xTrainR.shape[0],-1)/255

categoriesNhot = torch.zeros((10, 50), device=DEVICE)
ind = torch.randperm(50, device=DEVICE)

for i in range(10):
    categoriesNhot[i, ind[i*5:(i+1)*5]] = 1

xTrainNhot = nhot_coder(yTrain, 5, categoriesNhot, 50)

mu = 0
sigma = 0.1
lr = 0.0005
batch_size = 1000
batch_num = 60
layer_num = 4
inputL = xTrainN.reshape((batch_num, batch_size, 15*15))
epoch_num = 100

b1 =  torch.normal(mu, sigma, size=(1,50), device=DEVICE)
b2 =  torch.normal(mu, sigma, size=(1,50), device=DEVICE)
b3 =  torch.normal(mu, sigma, size=(1,50), device=DEVICE)
b4 =  torch.normal(mu, sigma, size=(1,50), device=DEVICE)
b5 =  torch.normal(mu, sigma, size=(1,50), device=DEVICE)
B1 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
B2 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
B3 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
B4 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
w0 = torch.normal(mu, sigma, size=(225,50), device=DEVICE)
w1 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
w2 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
w3 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
w4 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
w0_0 = torch.clone(w0)
w1_0 = torch.clone(w1)
w2_0 = torch.clone(w2)
w3_0 = torch.clone(w3)
w4_0 = torch.clone(w4)
hlayer1_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
hlayer2_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
hlayer3_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
hlayer4_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
delta5_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
delta4_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
delta3_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
delta2_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)

alignment = torch.zeros((1, 4, batch_num*epoch_num), device=DEVICE)
w0_wk = torch.zeros((1, 5, batch_num*epoch_num), device=DEVICE)
alignmentDWFAB = torch.zeros((1, 4, batch_num*epoch_num), device=DEVICE)
accTrain = torch.zeros((1, batch_num, epoch_num), device=DEVICE)
lossTrain = torch.zeros((1, batch_num, epoch_num), device=DEVICE)

for j in range(epoch_num):
    order = torch.arange(0,batch_num+1, device=DEVICE)
    for i in range(batch_num):
        w0_wk[0,0,j*batch_num+i] = torch.sqrt(torch.sum(torch.pow(w0,2)))/torch.sqrt(torch.sum(torch.pow(w0_0,2)))
        w0_wk[0,1,j*batch_num+i] = torch.sqrt(torch.sum(torch.pow(w1,2)))/torch.sqrt(torch.sum(torch.pow(w1_0,2)))
        w0_wk[0,2,j*batch_num+i] = torch.sqrt(torch.sum(torch.pow(w2,2)))/torch.sqrt(torch.sum(torch.pow(w2_0,2)))
        w0_wk[0,3,j*batch_num+i] = torch.sqrt(torch.sum(torch.pow(w3,2)))/torch.sqrt(torch.sum(torch.pow(w3_0,2)))
        w0_wk[0,4,j*batch_num+i] = torch.sqrt(torch.sum(torch.pow(w4,2)))/torch.sqrt(torch.sum(torch.pow(w4_0,2)))

        hl1 = torch.tanh(ReLU(torch.matmul(inputL[order[i],:,:], w0) + b1))
        hl2 = torch.tanh(ReLU(torch.matmul(hl1, w1) + b2))
        hl3 = torch.tanh(ReLU(torch.matmul(hl2, w2) + b3))
        hl4 = torch.tanh(ReLU(torch.matmul(hl3, w3) + b4))
        outL = torch.tanh(ReLU(torch.matmul(hl4, w4) + b5))

        E = xTrainNhot[order[i]*batch_size:(order[i]+1)*batch_size,:]-outL

        delta5 = torch.multiply(E,tanhReLUdr(torch.matmul(hl4,w4) + b5))
        delta4 = torch.multiply(torch.matmul(delta5,B4),tanhReLUdr(torch.matmul(hl3,w3) + b4))
        delta3 = torch.multiply(torch.matmul(delta4,B3),tanhReLUdr(torch.matmul(hl2,w2) + b3))
        delta2 = torch.multiply(torch.matmul(delta3,B2),tanhReLUdr(torch.matmul(hl1,w1) + b2))
        delta1 = torch.multiply(torch.matmul(delta2,B1),tanhReLUdr(torch.matmul(inputL[order[i],:,:],w0) + b1))

        if(j*batch_num+i < 1260):
            hlayer1_temp[j*batch_num+i,:,:] = hl1
            hlayer2_temp[j*batch_num+i,:,:] = hl2
            hlayer3_temp[j*batch_num+i,:,:] = hl3
            hlayer4_temp[j*batch_num+i,:,:] = hl4
            delta5_temp[j*batch_num+i,:,:] = delta5
            delta4_temp[j*batch_num+i,:,:] = delta4
            delta3_temp[j*batch_num+i,:,:] = delta3
            delta2_temp[j*batch_num+i,:,:] = delta2

        w4 += lr*torch.matmul(hl4.T,delta5)
        w3 += lr*torch.matmul(hl3.T,delta4)
        w2 += lr*torch.matmul(hl2.T,delta3)
        w1 += lr*torch.matmul(hl1.T,delta2)
        w0 += lr*torch.matmul(inputL[order[i],:,:].T,delta1)

        J = torch.ones((1,batch_size), device=DEVICE)
        b5 += torch.squeeze(lr*torch.matmul(J,delta5))
        b4 += torch.squeeze(lr*torch.matmul(J,delta4))
        b3 += torch.squeeze(lr*torch.matmul(J,delta3))
        b2 += torch.squeeze(lr*torch.matmul(J,delta2))
        b1 += torch.squeeze(lr*torch.matmul(J,delta1))

        outL_B = binarizing(outL,5,categoriesNhot)

        lossTrain[0,i,j] = lossFunc(outL,xTrainNhot[order[i]*batch_size:(order[i]+1)*batch_size,:])
        accTrain[0,i,j] = accFunc(outL_B,xTrainNhot[order[i]*batch_size:(order[i]+1)*batch_size,:]);

        alignment[0,0,j*batch_num+i] = angleFunc(w1,B1.T)
        alignment[0,1,j*batch_num+i] = angleFunc(w2,B2.T)
        alignment[0,2,j*batch_num+i] = angleFunc(w3,B3.T)
        alignment[0,3,j*batch_num+i] = angleFunc(w4,B4.T)

        delta5FA_temp = lr*torch.matmul(hl4.T,delta5)
        delta4FA_temp = lr*torch.matmul(hl3.T,delta4)
        delta3FA_temp = lr*torch.matmul(hl2.T,delta3)
        delta2FA_temp = lr*torch.matmul(hl1.T,delta2)

        alignmentDWFAB[0,0,j*batch_num+i] = angleFunc(delta2FA_temp,B1.T)
        alignmentDWFAB[0,1,j*batch_num+i] = angleFunc(delta3FA_temp,B2.T)
        alignmentDWFAB[0,2,j*batch_num+i] = angleFunc(delta4FA_temp,B3.T)
        alignmentDWFAB[0,3,j*batch_num+i] = angleFunc(delta5FA_temp,B4.T)
    print("Acc: {}".format(torch.mean(accTrain[0,:,j])) + " , Loss: {}".format(torch.mean(lossTrain[0,:,j])))

alignmentTerms = []
anglesAlignmentTerms = np.zeros((1, 2, 65, 4))
for i in range(1, 1260):
  if(i == 65 or i == 1259):
      o = 0
      torch.cuda.empty_cache()
      alignmentTermsK = torch.zeros((i, 4, 50, 50), device=DEVICE)
      while (o < i):
          alignmentTermsK[o,0,:,:] = (lr**2)*torch.chain_matmul(B1.T,delta2_temp[i-o-1,:,:].T,\
                                                                  inputL[np.mod(i-o-1,batch_num),:,:],inputL[np.mod(i,batch_num),:,:].T,\
                                                                  delta2_temp[i,:,:])
          alignmentTermsK[o,1,:,:] = (lr**2)*torch.chain_matmul(B2.T,delta3_temp[i-o-1,:,:].T,\
                                                                  hlayer1_temp[i-o-1,:,:],hlayer1_temp[i,:,:].T,\
                                                                  delta3_temp[i,:,:])
          alignmentTermsK[o,2,:,:] = (lr**2)*torch.chain_matmul(B3.T,delta4_temp[i-o-1,:,:].T,\
                                                                  hlayer2_temp[i-o-1,:,:],hlayer2_temp[i,:,:].T,\
                                                                  delta4_temp[i,:,:])
          alignmentTermsK[o,3,:,:] = (lr**2)*torch.chain_matmul(B4.T,delta5_temp[i-o-1,:,:].T,\
                                                            hlayer3_temp[i-o-1,:,:],hlayer3_temp[i,:,:].T,\
                                                            delta5_temp[i,:,:])
          o += 1
      alignmentTerms.append(alignmentTermsK)

for kk in range(anglesAlignmentTerms.shape[1]):
  T_k = alignmentTerms[-1]
  for ii in range(anglesAlignmentTerms.shape[2]):
      T_o = np.squeeze(T_k[ii,:,:,:])
      anglesAlignmentTerms[0,kk,ii,0] = angleFunc(T_o[0,:,:],B1.T)
      anglesAlignmentTerms[0,kk,ii,1] = angleFunc(T_o[1,:,:],B2.T)
      anglesAlignmentTerms[0,kk,ii,2] = angleFunc(T_o[2,:,:],B3.T)
      anglesAlignmentTerms[0,kk,ii,3] = angleFunc(T_o[3,:,:],B4.T)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for i in range(1, epoch_num+1):
  ax[0].scatter(i, torch.mean(accTrain[0,:,i-1]).cpu().numpy(), c='k', s=3)
  ax[1].scatter(i, torch.mean(lossTrain[0,:,i-1]).cpu().numpy(), c='r', s=3)
ax[0].set_ylim(0, 100)
ax[0].set_title('Accuarcy')
ax[0].set_xlabel('Epoch')
ax[1].set_title('Loss')
ax[1].set_xlabel('Epoch')

# Normalized
def normalizing(W):
    normalized_w = torch.clone(W)
    for i in range(W.shape[1]):
      normalized_w[:,i] = W[:,i]/torch.sqrt(torch.sum(torch.pow(W[:,i],2)))
    return normalized_w
mu = 0
sigma = 0.1
lr = 0.0005
batch_size = 1000
batch_num = 60
layer_num = 4
inputL = xTrainN.reshape((batch_num, batch_size, 15*15))
epoch_num = 100

b1 =  torch.normal(mu, sigma, size=(1,50), device=DEVICE)
b2 =  torch.normal(mu, sigma, size=(1,50), device=DEVICE)
b3 =  torch.normal(mu, sigma, size=(1,50), device=DEVICE)
b4 =  torch.normal(mu, sigma, size=(1,50), device=DEVICE)
b5 =  torch.normal(mu, sigma, size=(1,50), device=DEVICE)
B1 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
B2 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
B3 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
B4 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
B1 = normalizing(B1)
B2 = normalizing(B2)
B3 = normalizing(B3)
B4 = normalizing(B4)
w0 = torch.normal(mu, sigma, size=(225,50), device=DEVICE)
w1 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
w2 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
w3 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
w4 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
w0_0 = torch.clone(w0)
w1_0 = torch.clone(w1)
w2_0 = torch.clone(w2)
w3_0 = torch.clone(w3)
w4_0 = torch.clone(w4)
hlayer1_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
hlayer2_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
hlayer3_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
hlayer4_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
delta5_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
delta4_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
delta3_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
delta2_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)

alignmentN = torch.zeros((1, 4, batch_num*epoch_num), device=DEVICE)
w0_wk = torch.zeros((1, 5, batch_num*epoch_num), device=DEVICE)
alignmentDWFAB_N = torch.zeros((1, 4, batch_num*epoch_num), device=DEVICE)
accTrain = torch.zeros((1, batch_num, epoch_num), device=DEVICE)
lossTrain = torch.zeros((1, batch_num, epoch_num), device=DEVICE)

for j in range(epoch_num):
    order = torch.arange(0,batch_num+1, device=DEVICE)
    for i in range(batch_num):
        w0_wk[0,0,j*batch_num+i] = torch.sqrt(torch.sum(torch.pow(w0,2)))/torch.sqrt(torch.sum(torch.pow(w0_0,2)))
        w0_wk[0,1,j*batch_num+i] = torch.sqrt(torch.sum(torch.pow(w1,2)))/torch.sqrt(torch.sum(torch.pow(w1_0,2)))
        w0_wk[0,2,j*batch_num+i] = torch.sqrt(torch.sum(torch.pow(w2,2)))/torch.sqrt(torch.sum(torch.pow(w2_0,2)))
        w0_wk[0,3,j*batch_num+i] = torch.sqrt(torch.sum(torch.pow(w3,2)))/torch.sqrt(torch.sum(torch.pow(w3_0,2)))
        w0_wk[0,4,j*batch_num+i] = torch.sqrt(torch.sum(torch.pow(w4,2)))/torch.sqrt(torch.sum(torch.pow(w4_0,2)))

        hl1 = torch.tanh(ReLU(torch.matmul(inputL[order[i],:,:], w0) + b1))
        hl2 = torch.tanh(ReLU(torch.matmul(hl1, w1) + b2))
        hl3 = torch.tanh(ReLU(torch.matmul(hl2, w2) + b3))
        hl4 = torch.tanh(ReLU(torch.matmul(hl3, w3) + b4))
        outL = torch.tanh(ReLU(torch.matmul(hl4, w4) + b5))

        E = xTrainNhot[order[i]*batch_size:(order[i]+1)*batch_size,:]-outL

        delta5 = torch.multiply(E,tanhReLUdr(torch.matmul(hl4,w4) + b5))
        delta4 = torch.multiply(torch.matmul(delta5,B4),tanhReLUdr(torch.matmul(hl3,w3) + b4))
        delta3 = torch.multiply(torch.matmul(delta4,B3),tanhReLUdr(torch.matmul(hl2,w2) + b3))
        delta2 = torch.multiply(torch.matmul(delta3,B2),tanhReLUdr(torch.matmul(hl1,w1) + b2))
        delta1 = torch.multiply(torch.matmul(delta2,B1),tanhReLUdr(torch.matmul(inputL[order[i],:,:],w0) + b1))

        if(j*batch_num+i < 1260):
            hlayer1_temp[j*batch_num+i,:,:] = hl1
            hlayer2_temp[j*batch_num+i,:,:] = hl2
            hlayer3_temp[j*batch_num+i,:,:] = hl3
            hlayer4_temp[j*batch_num+i,:,:] = hl4
            delta5_temp[j*batch_num+i,:,:] = delta5
            delta4_temp[j*batch_num+i,:,:] = delta4
            delta3_temp[j*batch_num+i,:,:] = delta3
            delta2_temp[j*batch_num+i,:,:] = delta2

        w4 += lr*torch.matmul(hl4.T,delta5)
        w3 += lr*torch.matmul(hl3.T,delta4)
        w2 += lr*torch.matmul(hl2.T,delta3)
        w1 += lr*torch.matmul(hl1.T,delta2)
        w0 += lr*torch.matmul(inputL[order[i],:,:].T,delta1)
        w4 = normalizing(w4)
        w3 = normalizing(w3)
        w2 = normalizing(w2)
        w1 = normalizing(w1)
        w0 = normalizing(w0)

        J = torch.ones((1,batch_size), device=DEVICE)
        b5 += torch.squeeze(lr*torch.matmul(J,delta5))
        b4 += torch.squeeze(lr*torch.matmul(J,delta4))
        b3 += torch.squeeze(lr*torch.matmul(J,delta3))
        b2 += torch.squeeze(lr*torch.matmul(J,delta2))
        b1 += torch.squeeze(lr*torch.matmul(J,delta1))

        outL_B = binarizing(outL,5,categoriesNhot)

        lossTrain[0,i,j] = lossFunc(outL,xTrainNhot[order[i]*batch_size:(order[i]+1)*batch_size,:])
        accTrain[0,i,j] = accFunc(outL_B,xTrainNhot[order[i]*batch_size:(order[i]+1)*batch_size,:]);

        alignmentN[0,0,j*batch_num+i] = angleFunc(w1,B1.T)
        alignmentN[0,1,j*batch_num+i] = angleFunc(w2,B2.T)
        alignmentN[0,2,j*batch_num+i] = angleFunc(w3,B3.T)
        alignmentN[0,3,j*batch_num+i] = angleFunc(w4,B4.T)

        delta5FA_temp = lr*torch.matmul(hl4.T,delta5)
        delta4FA_temp = lr*torch.matmul(hl3.T,delta4)
        delta3FA_temp = lr*torch.matmul(hl2.T,delta3)
        delta2FA_temp = lr*torch.matmul(hl1.T,delta2)

        alignmentDWFAB_N[0,0,j*batch_num+i] = angleFunc(delta2FA_temp,B1.T)
        alignmentDWFAB_N[0,1,j*batch_num+i] = angleFunc(delta3FA_temp,B2.T)
        alignmentDWFAB_N[0,2,j*batch_num+i] = angleFunc(delta4FA_temp,B3.T)
        alignmentDWFAB_N[0,3,j*batch_num+i] = angleFunc(delta5FA_temp,B4.T)
    print("Acc: {}".format(torch.mean(accTrain[0,:,j])) + " , Loss: {}".format(torch.mean(lossTrain[0,:,j])))

alignmentTerms = []
anglesAlignmentTerms = np.zeros((1, 2, 65, layer_num))
for ii in range(1, 1260):
  if(ii == 65 or ii == 1259):
      torch.cuda.empty_cache()
      alignmentTermsK = torch.zeros((ii, layer_num, 50, 50), device=DEVICE)
      for o in range(ii):
          alignmentTermsK[o,0,:,:] = (lr**2)*torch.chain_matmul(B1.T,delta2_temp[ii-o-1,:,:].T,\
                                                                  inputL[np.mod(ii-o-1,batch_num),:,:],inputL[np.mod(ii,batch_num),:,:].T,\
                                                                  delta2_temp[ii,:,:])
          alignmentTermsK[o,1,:,:] = (lr**2)*torch.chain_matmul(B2.T,delta3_temp[ii-o-1,:,:].T,\
                                                                  hlayer1_temp[ii-o-1,:,:],hlayer1_temp[ii,:,:].T,\
                                                                  delta3_temp[ii,:,:])
          alignmentTermsK[o,2,:,:] = (lr**2)*torch.chain_matmul(B3.T,delta4_temp[ii-o-1,:,:].T,\
                                                                  hlayer2_temp[ii-o-1,:,:],hlayer2_temp[ii,:,:].T,\
                                                                  delta4_temp[ii,:,:])
          alignmentTermsK[o,3,:,:] = (lr**2)*torch.chain_matmul(B4.T,delta5_temp[ii-o-1,:,:].T,\
                                                            hlayer3_temp[ii-o-1,:,:],hlayer3_temp[ii,:,:].T,\
                                                            delta5_temp[ii,:,:])
      alignmentTerms.append(alignmentTermsK)

for kk in range(anglesAlignmentTerms.shape[1]):
  T_k = alignmentTerms[kk]
  for ii in range(anglesAlignmentTerms.shape[2]):
      T_o = np.squeeze(T_k[ii,:,:,:])
      anglesAlignmentTerms[0,kk,ii,0] = angleFunc(T_o[0,:,:],B1.T)
      anglesAlignmentTerms[0,kk,ii,1] = angleFunc(T_o[1,:,:],B2.T)
      anglesAlignmentTerms[0,kk,ii,2] = angleFunc(T_o[2,:,:],B3.T)
      anglesAlignmentTerms[0,kk,ii,3] = angleFunc(T_o[3,:,:],B4.T)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for i in range(1, epoch_num+1):
  ax[0].scatter(i, torch.mean(accTrain[0,:,i-1]).cpu().numpy(), c='k', s=3)
  ax[1].scatter(i, torch.mean(lossTrain[0,:,i-1]).cpu().numpy(), c='r', s=3)
ax[0].set_ylim(0, 100)
ax[0].set_title('Accuarcy (Normalized)')
ax[0].set_xlabel('Epoch')
ax[1].set_title('Loss (Normalized)')
ax[1].set_xlabel('Epoch')

plt.figure()
plt.plot(np.arange(1,epoch_num*batch_num+1),np.mean(alignment[:,0,:].cpu().numpy(),axis=(0)),':',color='orange')
plt.plot(np.arange(1,epoch_num*batch_num+1),np.mean(alignment[:,1,:].cpu().numpy(),axis=(0)),':',color='green')
plt.plot(np.arange(1,epoch_num*batch_num+1),np.mean(alignment[:,2,:].cpu().numpy(),axis=(0)),':',color='red')
plt.plot(np.arange(1,epoch_num*batch_num+1),np.mean(alignment[:,3,:].cpu().numpy(),axis=(0)),':',color='gray')
plt.plot(np.arange(1,epoch_num*batch_num+1),np.mean(alignmentN[:,0,:].cpu().numpy(),axis=(0)),color='orange')
plt.plot(np.arange(1,epoch_num*batch_num+1),np.mean(alignmentN[:,1,:].cpu().numpy(),axis=(0)),color='green')
plt.plot(np.arange(1,epoch_num*batch_num+1),np.mean(alignmentN[:,2,:].cpu().numpy(),axis=(0)),color='red')
plt.plot(np.arange(1,epoch_num*batch_num+1),np.mean(alignmentN[:,3,:].cpu().numpy(),axis=(0)),color='gray')
plt.title(r'Angle between $W_i$ & $B_i$')
plt.ylabel('Angle')
plt.xlabel('Iteration')
plt.legend(['W1 & B1.T','W2 & B2.T','W3 & B3.T','W4 & B4.T','W1_WN & B1_WN.T','W2_WN & B2_WN.T','W3_WN & B3_WN.T','W4_WN & B4_WN.T'])
plt.savefig("AngleW_iB_i.png", format="png")

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1,epoch_num*batch_num+2-60),movingAverage(np.mean(alignmentDWFAB[:,0,:].cpu().numpy(),axis=(0)), 60),':',color='orange', alpha=0.5)
plt.plot(np.arange(1,epoch_num*batch_num+2-60),movingAverage(np.mean(alignmentDWFAB[:,1,:].cpu().numpy(),axis=(0)), 60),':',color='green', alpha=0.5)
plt.plot(np.arange(1,epoch_num*batch_num+2-60),movingAverage(np.mean(alignmentDWFAB[:,2,:].cpu().numpy(),axis=(0)), 60),':',color='red', alpha=0.5)
plt.plot(np.arange(1,epoch_num*batch_num+2-60),movingAverage(np.mean(alignmentDWFAB[:,3,:].cpu().numpy(),axis=(0)), 60),':',color='gray', alpha=0.5)
plt.plot(np.arange(1,epoch_num*batch_num+2-60),movingAverage(np.mean(alignmentDWFAB_N[:,0,:].cpu().numpy(),axis=(0)), 60),color='orange')
plt.plot(np.arange(1,epoch_num*batch_num+2-60),movingAverage(np.mean(alignmentDWFAB_N[:,1,:].cpu().numpy(),axis=(0)), 60),color='green')
plt.plot(np.arange(1,epoch_num*batch_num+2-60),movingAverage(np.mean(alignmentDWFAB_N[:,2,:].cpu().numpy(),axis=(0)), 60),color='red')
plt.plot(np.arange(1,epoch_num*batch_num+2-60),movingAverage(np.mean(alignmentDWFAB_N[:,3,:].cpu().numpy(),axis=(0)), 60),color='gray')
plt.title(r'Angle between $\Delta$ $W_{l,FA}$ & ${B_l}^T$')
plt.ylabel('Angle')
plt.xlabel('Iteration')
plt.legend(['l = 1','l = 2','l = 3','l = 4','l-WN = 1','l-WN = 2','l-WN = 3','l-WN = 4'])
plt.savefig("AngleW_lFA_B_l.png", format="png", dpi=312)

fig, ax = plt.subplots(1,4,figsize=(20,4))
ax[0].scatter(np.arange(1,66),np.mean(anglesAlignmentTerms[:,0,:,0],axis=0),color = "b", s=4)
ax[1].scatter(np.arange(1,66),np.mean(anglesAlignmentTerms[:,0,:,1],axis=0),color = "b", s=4)
ax[2].scatter(np.arange(1,66),np.mean(anglesAlignmentTerms[:,0,:,2],axis=0),color = "b", s=4)
ax[3].scatter(np.arange(1,66),np.mean(anglesAlignmentTerms[:,0,:,3],axis=0),color = "b", s=4)
ax[0].scatter(np.arange(1,66),np.mean(anglesAlignmentTerms[:,1,:,0],axis=0),color = "r", s=4)
ax[1].scatter(np.arange(1,66),np.mean(anglesAlignmentTerms[:,1,:,1],axis=0),color = "r", s=4)
ax[2].scatter(np.arange(1,66),np.mean(anglesAlignmentTerms[:,1,:,2],axis=0),color = "r", s=4)
ax[3].scatter(np.arange(1,66),np.mean(anglesAlignmentTerms[:,1,:,3],axis=0),color = "r", s=4)
ax[0].set_ylim([0, 70])
ax[1].set_ylim([0, 70])
ax[2].set_ylim([0, 70])
ax[3].set_ylim([0, 70])
ax[0].set_ylim([60, 100])
ax[1].set_ylim([60, 100])
ax[2].set_ylim([60, 100])
ax[3].set_ylim([60, 100])
ax[0].set_title(r'Angle between $T_{1}^{o}[k]$ & ${B_{1}}^{T}$')
ax[1].set_title(r'Angle between $T_{2}^{o}[k]$ & ${B_{2}}^{T}$')
ax[2].set_title(r'Angle between $T_{3}^{o}[k]$ & ${B_{3}}^{T}$')
ax[3].set_title(r'Angle between $T_{4}^{o}[k]$ & ${B_{4}}^{T}$')
ax[0].set_xlabel('Batch')
ax[1].set_xlabel('Batch')
ax[2].set_xlabel('Batch')
ax[3].set_xlabel('Batch')
ax[0].set_ylabel('Angle')
ax[1].set_ylabel('Angle')
ax[2].set_ylabel('Angle')
ax[3].set_ylabel('Angle')
ax[0].legend(['k=66', 'k=1260'])
ax[1].legend(['k=66', 'k=1260'])
ax[2].legend(['k=66', 'k=1260'])
ax[3].legend(['k=66', 'k=1260'])
plt.savefig("scatter.png", format="png")

# Shuffled
mu = 0
sigma = 0.1
lr = 0.0005
batch_size = 1000
batch_num = 60
layer_num = 4
inputL = xTrainN.reshape((batch_num, batch_size, 15*15))
epoch_num = 100

b1 =  torch.normal(mu, sigma, size=(1,50), device=DEVICE)
b2 =  torch.normal(mu, sigma, size=(1,50), device=DEVICE)
b3 =  torch.normal(mu, sigma, size=(1,50), device=DEVICE)
b4 =  torch.normal(mu, sigma, size=(1,50), device=DEVICE)
b5 =  torch.normal(mu, sigma, size=(1,50), device=DEVICE)
B1 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
B2 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
B3 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
B4 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
w0 = torch.normal(mu, sigma, size=(225,50), device=DEVICE)
w1 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
w2 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
w3 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
w4 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
w0_0 = torch.clone(w0)
w1_0 = torch.clone(w1)
w2_0 = torch.clone(w2)
w3_0 = torch.clone(w3)
w4_0 = torch.clone(w4)
hlayer1_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
hlayer2_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
hlayer3_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
hlayer4_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
delta5_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
delta4_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
delta3_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
delta2_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)

alignment = torch.zeros((1, 4, batch_num*epoch_num), device=DEVICE)
w0_wk = torch.zeros((1, 5, batch_num*epoch_num), device=DEVICE)
alignmentDWFAB = torch.zeros((1, 4, batch_num*epoch_num), device=DEVICE)
accTrain = torch.zeros((1, batch_num, epoch_num), device=DEVICE)
lossTrain = torch.zeros((1, batch_num, epoch_num), device=DEVICE)

for j in range(epoch_num):
    order = torch.randperm(batch_num, device=DEVICE)
    for i in range(batch_num):
        w0_wk[0,0,j*batch_num+i] = torch.sqrt(torch.sum(torch.pow(w0,2)))/torch.sqrt(torch.sum(torch.pow(w0_0,2)))
        w0_wk[0,1,j*batch_num+i] = torch.sqrt(torch.sum(torch.pow(w1,2)))/torch.sqrt(torch.sum(torch.pow(w1_0,2)))
        w0_wk[0,2,j*batch_num+i] = torch.sqrt(torch.sum(torch.pow(w2,2)))/torch.sqrt(torch.sum(torch.pow(w2_0,2)))
        w0_wk[0,3,j*batch_num+i] = torch.sqrt(torch.sum(torch.pow(w3,2)))/torch.sqrt(torch.sum(torch.pow(w3_0,2)))
        w0_wk[0,4,j*batch_num+i] = torch.sqrt(torch.sum(torch.pow(w4,2)))/torch.sqrt(torch.sum(torch.pow(w4_0,2)))

        hl1 = torch.tanh(ReLU(torch.matmul(inputL[order[i],:,:], w0) + b1))
        hl2 = torch.tanh(ReLU(torch.matmul(hl1, w1) + b2))
        hl3 = torch.tanh(ReLU(torch.matmul(hl2, w2) + b3))
        hl4 = torch.tanh(ReLU(torch.matmul(hl3, w3) + b4))
        outL = torch.tanh(ReLU(torch.matmul(hl4, w4) + b5))

        E = xTrainNhot[order[i]*batch_size:(order[i]+1)*batch_size,:]-outL

        delta5 = torch.multiply(E,tanhReLUdr(torch.matmul(hl4,w4) + b5))
        delta4 = torch.multiply(torch.matmul(delta5,B4),tanhReLUdr(torch.matmul(hl3,w3) + b4))
        delta3 = torch.multiply(torch.matmul(delta4,B3),tanhReLUdr(torch.matmul(hl2,w2) + b3))
        delta2 = torch.multiply(torch.matmul(delta3,B2),tanhReLUdr(torch.matmul(hl1,w1) + b2))
        delta1 = torch.multiply(torch.matmul(delta2,B1),tanhReLUdr(torch.matmul(inputL[order[i],:,:],w0) + b1))

        if(j*batch_num+i < 1260):
            hlayer1_temp[j*batch_num+i,:,:] = hl1
            hlayer2_temp[j*batch_num+i,:,:] = hl2
            hlayer3_temp[j*batch_num+i,:,:] = hl3
            hlayer4_temp[j*batch_num+i,:,:] = hl4
            delta5_temp[j*batch_num+i,:,:] = delta5
            delta4_temp[j*batch_num+i,:,:] = delta4
            delta3_temp[j*batch_num+i,:,:] = delta3
            delta2_temp[j*batch_num+i,:,:] = delta2

        w4 += lr*torch.matmul(hl4.T,delta5)
        w3 += lr*torch.matmul(hl3.T,delta4)
        w2 += lr*torch.matmul(hl2.T,delta3)
        w1 += lr*torch.matmul(hl1.T,delta2)
        w0 += lr*torch.matmul(inputL[order[i],:,:].T,delta1)

        J = torch.ones((1,batch_size), device=DEVICE)
        b5 += torch.squeeze(lr*torch.matmul(J,delta5))
        b4 += torch.squeeze(lr*torch.matmul(J,delta4))
        b3 += torch.squeeze(lr*torch.matmul(J,delta3))
        b2 += torch.squeeze(lr*torch.matmul(J,delta2))
        b1 += torch.squeeze(lr*torch.matmul(J,delta1))

        outL_B = binarizing(outL,5,categoriesNhot)

        lossTrain[0,i,j] = lossFunc(outL,xTrainNhot[order[i]*batch_size:(order[i]+1)*batch_size,:])
        accTrain[0,i,j] = accFunc(outL_B,xTrainNhot[order[i]*batch_size:(order[i]+1)*batch_size,:]);

        alignment[0,0,j*batch_num+i] = angleFunc(w1,B1.T)
        alignment[0,1,j*batch_num+i] = angleFunc(w2,B2.T)
        alignment[0,2,j*batch_num+i] = angleFunc(w3,B3.T)
        alignment[0,3,j*batch_num+i] = angleFunc(w4,B4.T)

        delta5FA_temp = lr*torch.matmul(hl4.T,delta5)
        delta4FA_temp = lr*torch.matmul(hl3.T,delta4)
        delta3FA_temp = lr*torch.matmul(hl2.T,delta3)
        delta2FA_temp = lr*torch.matmul(hl1.T,delta2)

        alignmentDWFAB[0,0,j*batch_num+i] = angleFunc(delta2FA_temp,B1.T)
        alignmentDWFAB[0,1,j*batch_num+i] = angleFunc(delta3FA_temp,B2.T)
        alignmentDWFAB[0,2,j*batch_num+i] = angleFunc(delta4FA_temp,B3.T)
        alignmentDWFAB[0,3,j*batch_num+i] = angleFunc(delta5FA_temp,B4.T)
    print("Acc: {}".format(torch.mean(accTrain[0,:,j])) + " , Loss: {}".format(torch.mean(lossTrain[0,:,j])))


alignmentTerms = []
anglesAlignmentTerms = np.zeros((1, 2, 65, layer_num))
for ii in range(1, 1260):
  if(ii == 65 or ii == 1259):
      torch.cuda.empty_cache()
      alignmentTermsK = torch.zeros((ii,layer_num,50,50), device=DEVICE)
      for o in range(ii):
          alignmentTermsK[o,0,:,:] = (lr**2)*torch.chain_matmul(B1.T,delta2_temp[ii-o-1,:,:].T,\
                                                                  inputL[np.mod(ii-o-1,batch_num),:,:],inputL[np.mod(ii,batch_num),:,:].T,\
                                                                  delta2_temp[ii,:,:])
          alignmentTermsK[o,1,:,:] = (lr**2)*torch.chain_matmul(B2.T,delta3_temp[ii-o-1,:,:].T,\
                                                                  hlayer1_temp[ii-o-1,:,:],hlayer1_temp[ii,:,:].T,\
                                                                  delta3_temp[ii,:,:])
          alignmentTermsK[o,2,:,:] = (lr**2)*torch.chain_matmul(B3.T,delta4_temp[ii-o-1,:,:].T,\
                                                                  hlayer2_temp[ii-o-1,:,:],hlayer2_temp[ii,:,:].T,\
                                                                  delta4_temp[ii,:,:])
          alignmentTermsK[o,3,:,:] = (lr**2)*torch.chain_matmul(B4.T,delta5_temp[ii-o-1,:,:].T,\
                                                            hlayer3_temp[ii-o-1,:,:],hlayer3_temp[ii,:,:].T,\
                                                            delta5_temp[ii,:,:])
      alignmentTerms.append(alignmentTermsK)

for kk in range(anglesAlignmentTerms.shape[1]):
  T_k = alignmentTerms[kk]
  for ii in range(anglesAlignmentTerms.shape[2]):
      T_o = np.squeeze(T_k[ii,:,:,:])
      anglesAlignmentTerms[0,kk,ii,0] = angleFunc(T_o[0,:,:],B1.T)
      anglesAlignmentTerms[0,kk,ii,1] = angleFunc(T_o[1,:,:],B2.T)
      anglesAlignmentTerms[0,kk,ii,2] = angleFunc(T_o[2,:,:],B3.T)
      anglesAlignmentTerms[0,kk,ii,3] = angleFunc(T_o[3,:,:],B4.T)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for i in range(1, epoch_num+1):
  ax[0].scatter(i, torch.mean(accTrain[0,:,i-1]).cpu().numpy(), c='k', s=3)
  ax[1].scatter(i, torch.mean(lossTrain[0,:,i-1]).cpu().numpy(), c='r', s=3)
ax[0].set_ylim(0, 100)
ax[0].set_title('Accuarcy + Shuffled')
ax[0].set_xlabel('Epoch')
ax[1].set_title('Loss + Shuffled')
ax[1].set_xlabel('Epoch')

# Normalized + Shuffled
def normalizing(W):
    normalized_w = torch.clone(W)
    for i in range(W.shape[1]):
      normalized_w[:,i] = W[:,i]/torch.sqrt(torch.sum(torch.pow(W[:,i],2)))
    return normalized_w
mu = 0
sigma = 0.1
lr = 0.0005
batch_size = 1000
batch_num = 60
layer_num = 4
inputL = xTrainN.reshape((batch_num, batch_size, 15*15))
epoch_num = 100

b1 =  torch.normal(mu, sigma, size=(1,50), device=DEVICE)
b2 =  torch.normal(mu, sigma, size=(1,50), device=DEVICE)
b3 =  torch.normal(mu, sigma, size=(1,50), device=DEVICE)
b4 =  torch.normal(mu, sigma, size=(1,50), device=DEVICE)
b5 =  torch.normal(mu, sigma, size=(1,50), device=DEVICE)
B1 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
B2 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
B3 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
B4 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
B1 = normalizing(B1)
B2 = normalizing(B2)
B3 = normalizing(B3)
B4 = normalizing(B4)
w0 = torch.normal(mu, sigma, size=(225,50), device=DEVICE)
w1 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
w2 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
w3 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
w4 = torch.normal(mu, sigma, size=(50,50), device=DEVICE)
w0_0 = torch.clone(w0)
w1_0 = torch.clone(w1)
w2_0 = torch.clone(w2)
w3_0 = torch.clone(w3)
w4_0 = torch.clone(w4)
hlayer1_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
hlayer2_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
hlayer3_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
hlayer4_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
delta5_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
delta4_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
delta3_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)
delta2_temp = torch.zeros((1260, batch_size, 50), device=DEVICE)

alignmentN = torch.zeros((1, 4, batch_num*epoch_num), device=DEVICE)
w0_wk = torch.zeros((1, 5, batch_num*epoch_num), device=DEVICE)
alignmentDWFAB_N = torch.zeros((1, 4, batch_num*epoch_num), device=DEVICE)
accTrain = torch.zeros((1, batch_num, epoch_num), device=DEVICE)
lossTrain = torch.zeros((1, batch_num, epoch_num), device=DEVICE)

for j in range(epoch_num):
    order = torch.arange(0,batch_num+1, device=DEVICE)
    for i in range(batch_num):
        w0_wk[0,0,j*batch_num+i] = torch.sqrt(torch.sum(torch.pow(w0,2)))/torch.sqrt(torch.sum(torch.pow(w0_0,2)))
        w0_wk[0,1,j*batch_num+i] = torch.sqrt(torch.sum(torch.pow(w1,2)))/torch.sqrt(torch.sum(torch.pow(w1_0,2)))
        w0_wk[0,2,j*batch_num+i] = torch.sqrt(torch.sum(torch.pow(w2,2)))/torch.sqrt(torch.sum(torch.pow(w2_0,2)))
        w0_wk[0,3,j*batch_num+i] = torch.sqrt(torch.sum(torch.pow(w3,2)))/torch.sqrt(torch.sum(torch.pow(w3_0,2)))
        w0_wk[0,4,j*batch_num+i] = torch.sqrt(torch.sum(torch.pow(w4,2)))/torch.sqrt(torch.sum(torch.pow(w4_0,2)))

        hl1 = torch.tanh(ReLU(torch.matmul(inputL[order[i],:,:], w0) + b1))
        hl2 = torch.tanh(ReLU(torch.matmul(hl1, w1) + b2))
        hl3 = torch.tanh(ReLU(torch.matmul(hl2, w2) + b3))
        hl4 = torch.tanh(ReLU(torch.matmul(hl3, w3) + b4))
        outL = torch.tanh(ReLU(torch.matmul(hl4, w4) + b5))

        E = xTrainNhot[order[i]*batch_size:(order[i]+1)*batch_size,:]-outL

        delta5 = torch.multiply(E,tanhReLUdr(torch.matmul(hl4,w4) + b5))
        delta4 = torch.multiply(torch.matmul(delta5,B4),tanhReLUdr(torch.matmul(hl3,w3) + b4))
        delta3 = torch.multiply(torch.matmul(delta4,B3),tanhReLUdr(torch.matmul(hl2,w2) + b3))
        delta2 = torch.multiply(torch.matmul(delta3,B2),tanhReLUdr(torch.matmul(hl1,w1) + b2))
        delta1 = torch.multiply(torch.matmul(delta2,B1),tanhReLUdr(torch.matmul(inputL[order[i],:,:],w0) + b1))

        if(j*batch_num+i < 1260):
            hlayer1_temp[j*batch_num+i,:,:] = hl1
            hlayer2_temp[j*batch_num+i,:,:] = hl2
            hlayer3_temp[j*batch_num+i,:,:] = hl3
            hlayer4_temp[j*batch_num+i,:,:] = hl4
            delta5_temp[j*batch_num+i,:,:] = delta5
            delta4_temp[j*batch_num+i,:,:] = delta4
            delta3_temp[j*batch_num+i,:,:] = delta3
            delta2_temp[j*batch_num+i,:,:] = delta2

        w4 += lr*torch.matmul(hl4.T,delta5)
        w3 += lr*torch.matmul(hl3.T,delta4)
        w2 += lr*torch.matmul(hl2.T,delta3)
        w1 += lr*torch.matmul(hl1.T,delta2)
        w0 += lr*torch.matmul(inputL[order[i],:,:].T,delta1)
        w4 = normalizing(w4)
        w3 = normalizing(w3)
        w2 = normalizing(w2)
        w1 = normalizing(w1)
        w0 = normalizing(w0)

        J = torch.ones((1,batch_size), device=DEVICE)
        b5 += torch.squeeze(lr*torch.matmul(J,delta5))
        b4 += torch.squeeze(lr*torch.matmul(J,delta4))
        b3 += torch.squeeze(lr*torch.matmul(J,delta3))
        b2 += torch.squeeze(lr*torch.matmul(J,delta2))
        b1 += torch.squeeze(lr*torch.matmul(J,delta1))

        outL_B = binarizing(outL,5,categoriesNhot)

        lossTrain[0,i,j] = lossFunc(outL,xTrainNhot[order[i]*batch_size:(order[i]+1)*batch_size,:])
        accTrain[0,i,j] = accFunc(outL_B,xTrainNhot[order[i]*batch_size:(order[i]+1)*batch_size,:]);

        alignmentN[0,0,j*batch_num+i] = angleFunc(w1,B1.T)
        alignmentN[0,1,j*batch_num+i] = angleFunc(w2,B2.T)
        alignmentN[0,2,j*batch_num+i] = angleFunc(w3,B3.T)
        alignmentN[0,3,j*batch_num+i] = angleFunc(w4,B4.T)

        delta5FA_temp = lr*torch.matmul(hl4.T,delta5)
        delta4FA_temp = lr*torch.matmul(hl3.T,delta4)
        delta3FA_temp = lr*torch.matmul(hl2.T,delta3)
        delta2FA_temp = lr*torch.matmul(hl1.T,delta2)

        alignmentDWFAB_N[0,0,j*batch_num+i] = angleFunc(delta2FA_temp,B1.T)
        alignmentDWFAB_N[0,1,j*batch_num+i] = angleFunc(delta3FA_temp,B2.T)
        alignmentDWFAB_N[0,2,j*batch_num+i] = angleFunc(delta4FA_temp,B3.T)
        alignmentDWFAB_N[0,3,j*batch_num+i] = angleFunc(delta5FA_temp,B4.T)
    print("Acc: {}".format(torch.mean(accTrain[0,:,j])) + " , Loss: {}".format(torch.mean(lossTrain[0,:,j])))

alignmentTerms = []
anglesAlignmentTerms = np.zeros((1, 2, 65, layer_num))
for ii in range(1, 1260):
  if(ii == 65 or ii == 1259):
      torch.cuda.empty_cache()
      alignmentTermsK = torch.zeros((ii,layer_num,50,50), device=DEVICE)
      for o in range(ii):
          alignmentTermsK[o,0,:,:] = (lr**2)*torch.chain_matmul(B1.T,delta2_temp[ii-o-1,:,:].T,\
                                                                  inputL[np.mod(ii-o-1,batch_num),:,:],inputL[np.mod(ii,batch_num),:,:].T,\
                                                                  delta2_temp[ii,:,:])
          alignmentTermsK[o,1,:,:] = (lr**2)*torch.chain_matmul(B2.T,delta3_temp[ii-o-1,:,:].T,\
                                                                  hlayer1_temp[ii-o-1,:,:],hlayer1_temp[ii,:,:].T,\
                                                                  delta3_temp[ii,:,:])
          alignmentTermsK[o,2,:,:] = (lr**2)*torch.chain_matmul(B3.T,delta4_temp[ii-o-1,:,:].T,\
                                                                  hlayer2_temp[ii-o-1,:,:],hlayer2_temp[ii,:,:].T,\
                                                                  delta4_temp[ii,:,:])
          alignmentTermsK[o,3,:,:] = (lr**2)*torch.chain_matmul(B4.T,delta5_temp[ii-o-1,:,:].T,\
                                                            hlayer3_temp[ii-o-1,:,:],hlayer3_temp[ii,:,:].T,\
                                                            delta5_temp[ii,:,:])
      alignmentTerms.append(alignmentTermsK)

for kk in range(anglesAlignmentTerms.shape[1]):
  T_k = alignmentTerms[kk]
  for ii in range(anglesAlignmentTerms.shape[2]):
      T_o = np.squeeze(T_k[ii,:,:,:])
      anglesAlignmentTerms[0,kk,ii,0] = angleFunc(T_o[0,:,:],B1.T)
      anglesAlignmentTerms[0,kk,ii,1] = angleFunc(T_o[1,:,:],B2.T)
      anglesAlignmentTerms[0,kk,ii,2] = angleFunc(T_o[2,:,:],B3.T)
      anglesAlignmentTerms[0,kk,ii,3] = angleFunc(T_o[3,:,:],B4.T)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for i in range(1, epoch_num+1):
  ax[0].scatter(i, torch.mean(accTrain[0,:,i-1]).cpu().numpy(), c='k', s=3)
  ax[1].scatter(i, torch.mean(lossTrain[0,:,i-1]).cpu().numpy(), c='r', s=3)
ax[0].set_ylim(0, 100)
ax[0].set_title('Accuarcy (Normalized) + Shuffled')
ax[0].set_xlabel('Epoch')
ax[1].set_title('Loss (Normalized) + Shuffled')
ax[1].set_xlabel('Epoch')

plt.figure()
plt.plot(np.arange(1,epoch_num*batch_num+1),np.mean(alignment[:,0,:].cpu().numpy(),axis=(0)),':',color='orange')
plt.plot(np.arange(1,epoch_num*batch_num+1),np.mean(alignment[:,1,:].cpu().numpy(),axis=(0)),':',color='green')
plt.plot(np.arange(1,epoch_num*batch_num+1),np.mean(alignment[:,2,:].cpu().numpy(),axis=(0)),':',color='red')
plt.plot(np.arange(1,epoch_num*batch_num+1),np.mean(alignment[:,3,:].cpu().numpy(),axis=(0)),':',color='gray')
plt.plot(np.arange(1,epoch_num*batch_num+1),np.mean(alignmentN[:,0,:].cpu().numpy(),axis=(0)),color='orange')
plt.plot(np.arange(1,epoch_num*batch_num+1),np.mean(alignmentN[:,1,:].cpu().numpy(),axis=(0)),color='green')
plt.plot(np.arange(1,epoch_num*batch_num+1),np.mean(alignmentN[:,2,:].cpu().numpy(),axis=(0)),color='red')
plt.plot(np.arange(1,epoch_num*batch_num+1),np.mean(alignmentN[:,3,:].cpu().numpy(),axis=(0)),color='gray')
plt.title(r'Angle between $W_i$ & $B_i$ (Shuffled)')
plt.ylabel('Angle')
plt.xlabel('Iteration')
plt.legend(['W1 & B1.T','W2 & B2.T','W3 & B3.T','W4 & B4.T','W1_WN & B1_WN.T','W2_WN & B2_WN.T','W3_WN & B3_WN.T','W4_WN & B4_WN.T'])
plt.savefig("AngleW_iB_i_Suffled.png", format="png")

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1,epoch_num*batch_num+2-60),movingAverage(np.mean(alignmentDWFAB[:,0,:].cpu().numpy(),axis=(0)), 60),':',color='orange', alpha=0.5)
plt.plot(np.arange(1,epoch_num*batch_num+2-60),movingAverage(np.mean(alignmentDWFAB[:,1,:].cpu().numpy(),axis=(0)), 60),':',color='green', alpha=0.5)
plt.plot(np.arange(1,epoch_num*batch_num+2-60),movingAverage(np.mean(alignmentDWFAB[:,2,:].cpu().numpy(),axis=(0)), 60),':',color='red', alpha=0.5)
plt.plot(np.arange(1,epoch_num*batch_num+2-60),movingAverage(np.mean(alignmentDWFAB[:,3,:].cpu().numpy(),axis=(0)), 60),':',color='gray', alpha=0.5)
plt.plot(np.arange(1,epoch_num*batch_num+2-60),movingAverage(np.mean(alignmentDWFAB_N[:,0,:].cpu().numpy(),axis=(0)), 60),color='orange')
plt.plot(np.arange(1,epoch_num*batch_num+2-60),movingAverage(np.mean(alignmentDWFAB_N[:,1,:].cpu().numpy(),axis=(0)), 60),color='green')
plt.plot(np.arange(1,epoch_num*batch_num+2-60),movingAverage(np.mean(alignmentDWFAB_N[:,2,:].cpu().numpy(),axis=(0)), 60),color='red')
plt.plot(np.arange(1,epoch_num*batch_num+2-60),movingAverage(np.mean(alignmentDWFAB_N[:,3,:].cpu().numpy(),axis=(0)), 60),color='gray')
plt.title(r'Angle between $\Delta$ $W_{l,FA}$ & ${B_l}^T$ (Shuffled)')
plt.ylabel('Angle')
plt.xlabel('Iteration')
plt.legend(['l = 1','l = 2','l = 3','l = 4','l-WN = 1','l-WN = 2','l-WN = 3','l-WN = 4'])
plt.savefig("AngleW_lFA_B_l_Shuffled.png", format="png", dpi=312)

lr = 0.0005
batch_size = 1000
batch_num = 60
layer_num = 4
inputL = xTrainN.reshape((batch_num,batch_size,15*15))
epoch_num = 100

alignmentDWFABP = torch.zeros((1,5,batch_num*epoch_num), device=DEVICE)

w0_fa = torch.normal(mu,sigma,size=(225,50), device=DEVICE)
w1_fa = torch.normal(mu,sigma,size=(50,50), device=DEVICE)
w2_fa = torch.normal(mu,sigma,size=(50,50), device=DEVICE)
w3_fa = torch.normal(mu,sigma,size=(50,50), device=DEVICE)
w4_fa = torch.normal(mu,sigma,size=(50,50), device=DEVICE)
B4_fa = torch.normal(mu,sigma,size=(50,50), device=DEVICE)
B3_fa = torch.normal(mu,sigma,size=(50,50), device=DEVICE)
B2_fa = torch.normal(mu,sigma,size=(50,50), device=DEVICE)
B1_fa = torch.normal(mu,sigma,size=(50,50), device=DEVICE)
b1_fa =  torch.normal(mu,sigma,size=(1,50), device=DEVICE)
b2_fa =  torch.normal(mu,sigma,size=(1,50), device=DEVICE)
b3_fa =  torch.normal(mu,sigma,size=(1,50), device=DEVICE)
b4_fa =  torch.normal(mu,sigma,size=(1,50), device=DEVICE)
b5_fa =  torch.normal(mu,sigma,size=(1,50), device=DEVICE)
B1_bp = torch.clone(w1_fa.T)
B2_bp = torch.clone(w2_fa.T)
B3_bp = torch.clone(w3_fa.T)
B4_bp = torch.clone(w4_fa.T)

for j in range(epoch_num):
  for i in range(batch_num):
      z1_fa = torch.matmul(inputL[i,:,:],w0_fa) + b1_fa
      hl1_fa = torch.tanh(ReLU(z1_fa))
      z2_fa = torch.matmul(hl1_fa,w1_fa) + b2_fa
      hl2_fa = torch.tanh(ReLU(z2_fa))
      z3_fa = torch.matmul(hl2_fa,w2_fa) + b3_fa
      hl3_fa = torch.tanh(ReLU(z3_fa))
      z4_fa = torch.matmul(hl3_fa,w3_fa) + b4_fa
      hl4_fa = torch.tanh(ReLU(z4_fa))
      z5_fa = torch.matmul(hl4_fa,w4_fa) + b5_fa
      outL_fa = torch.tanh(ReLU(z5_fa))

      E_fa = xTrainNhot[i*batch_size:(i+1)*batch_size,:]-outL_fa
      delta5_fa = torch.multiply(E_fa,tanhReLUdr(z5_fa))
      delta4_fa = torch.multiply(torch.matmul(delta5_fa,B4_fa),tanhReLUdr(z4_fa))
      delta3_fa = torch.multiply(torch.matmul(delta4_fa,B3_fa),tanhReLUdr(z3_fa))
      delta2_fa = torch.multiply(torch.matmul(delta3_fa,B2_fa),tanhReLUdr(z2_fa))
      delta1_fa = torch.multiply(torch.matmul(delta2_fa,B1_fa),tanhReLUdr(z1_fa))

      outL_B_fa = binarizing(outL_fa,5,categoriesNhot)

      E_fa = xTrainNhot[i*batch_size:(i+1)*batch_size,:]-outL_fa
      delta5_bp = torch.multiply(E_fa,tanhReLUdr(z5_fa))
      delta4_bp = torch.multiply(torch.matmul(delta5_bp,B4_bp),tanhReLUdr(z4_fa))
      delta3_bp = torch.multiply(torch.matmul(delta4_bp,B3_bp),tanhReLUdr(z3_fa))
      delta2_bp = torch.multiply(torch.matmul(delta3_bp,B2_bp),tanhReLUdr(z2_fa))
      delta1_bp = torch.multiply(torch.matmul(delta2_bp,B1_bp),tanhReLUdr(z1_fa))

      w4_fa += lr*torch.matmul(hl4_fa.T,delta5_fa)
      w3_fa += lr*torch.matmul(hl3_fa.T,delta4_fa)
      w2_fa += lr*torch.matmul(hl2_fa.T,delta3_fa)
      w1_fa += lr*torch.matmul(hl1_fa.T,delta2_fa)
      w0_fa += lr*torch.matmul(inputL[i,:,:].T,delta1_fa)

      J_fa = torch.ones((1,batch_size), device=DEVICE)
      b5_fa += torch.squeeze(lr*torch.matmul(J_fa,delta5_fa))
      b4_fa += torch.squeeze(lr*torch.matmul(J_fa,delta4_fa))
      b3_fa += torch.squeeze(lr*torch.matmul(J_fa,delta3_fa))
      b2_fa += torch.squeeze(lr*torch.matmul(J_fa,delta2_fa))
      b1_fa += torch.squeeze(lr*torch.matmul(J_fa,delta1_fa))

      delta5FA_temp = lr*torch.matmul(hl4_fa.T,delta5_fa)
      delta4FA_temp = lr*torch.matmul(hl3_fa.T,delta4_fa)
      delta3FA_temp = lr*torch.matmul(hl2_fa.T,delta3_fa)
      delta2FA_temp = lr*torch.matmul(hl1_fa.T,delta2_fa)
      delta1_fa_inTime = lr*torch.matmul(inputL[i,:,:].T,delta1_fa)

      delta5_bp_inTime = lr*torch.matmul(hl4_fa.T,delta5_bp)
      delta4_bp_inTime = lr*torch.matmul(hl3_fa.T,delta4_bp)
      delta3_bp_inTime = lr*torch.matmul(hl2_fa.T,delta3_bp)
      delta2_bp_inTime = lr*torch.matmul(hl1_fa.T,delta2_bp)
      delta1_bp_inTime = lr*torch.matmul(inputL[i,:,:].T,delta1_bp)

      alignmentDWFABP[0,0,j*batch_num+i] = angleFunc(delta1_bp_inTime,delta1_fa_inTime)
      alignmentDWFABP[0,1,j*batch_num+i] = angleFunc(delta2_bp_inTime,delta2FA_temp)
      alignmentDWFABP[0,2,j*batch_num+i] = angleFunc(delta3_bp_inTime,delta3FA_temp)
      alignmentDWFABP[0,3,j*batch_num+i] = angleFunc(delta4_bp_inTime,delta4FA_temp)
      alignmentDWFABP[0,4,j*batch_num+i] = angleFunc(delta5_bp_inTime,delta5FA_temp)

      B1_bp = torch.clone(w1_fa.T)
      B2_bp = torch.clone(w2_fa.T)
      B3_bp = torch.clone(w3_fa.T)
      B4_bp = torch.clone(w4_fa.T)

window_size = 60
plt.figure()
plt.plot(np.arange(1,epoch_num*batch_num+2-window_size),movingAverage(np.mean(alignmentDWFABP[:,0,:].cpu().numpy(),axis=(0)), window_size),':',color='k')
plt.plot(np.arange(1,epoch_num*batch_num+2-window_size),movingAverage(np.mean(alignmentDWFABP[:,1,:].cpu().numpy(),axis=(0)), window_size),':',color='gray')
plt.plot(np.arange(1,epoch_num*batch_num+2-window_size),movingAverage(np.mean(alignmentDWFABP[:,2,:].cpu().numpy(),axis=(0)), window_size),':',color='r')
plt.plot(np.arange(1,epoch_num*batch_num+2-window_size),movingAverage(np.mean(alignmentDWFABP[:,3,:].cpu().numpy(),axis=(0)), window_size),':',color='b')
plt.plot(np.arange(1,epoch_num*batch_num+2-window_size),np.nanmean(alignmentDWFABP[:,4,0:epoch_num*batch_num+1-window_size].cpu().numpy(),axis=(0)),':',color='g')
plt.title(r'Angle between $\Delta$ $W_{l,FA}$ & $\Delta$ $W_{l,BP}$')
plt.xlabel('Iteration')
plt.ylabel('Angle')
plt.legend(['layer 0','layer 1','layer 2','layer 3','layer 4'])

lr = 0.0005
batch_size = 1000
batch_num = 60
layer_num = 4
inputL = xTrainN.reshape((batch_num,batch_size,15*15))
epoch_num = 10
iter_needed = 600

angle_TRW = torch.zeros((1,layer_num,batch_num*epoch_num), device=DEVICE)
angle_dstrb = torch.zeros((1,layer_num,batch_num*epoch_num), device=DEVICE)

w0_TRW = torch.normal(mu,sigma,size=(225,50), device=DEVICE)
w1_TRW = torch.normal(mu,sigma,size=(50,50), device=DEVICE)
w2_TRW = torch.normal(mu,sigma,size=(50,50), device=DEVICE)
w3_TRW = torch.normal(mu,sigma,size=(50,50), device=DEVICE)
w4_TRW = torch.normal(mu,sigma,size=(50,50), device=DEVICE)
B4 = torch.normal(mu,sigma,size=(50,50), device=DEVICE)
B3 = torch.normal(mu,sigma,size=(50,50), device=DEVICE)
B2 = torch.normal(mu,sigma,size=(50,50), device=DEVICE)
B1 = torch.normal(mu,sigma,size=(50,50), device=DEVICE)
b1_TRW =  torch.normal(mu,sigma,size=(1,50), device=DEVICE)
b2_TRW =  torch.normal(mu,sigma,size=(1,50), device=DEVICE)
b3_TRW =  torch.normal(mu,sigma,size=(1,50), device=DEVICE)
b4_TRW =  torch.normal(mu,sigma,size=(1,50), device=DEVICE)
b5_TRW =  torch.normal(mu,sigma,size=(1,50), device=DEVICE)
w0_dstrb = torch.clone(w0_TRW)
w1_dstrb = torch.clone(w1_TRW)
w2_dstrb = torch.clone(w2_TRW)
w3_dstrb = torch.clone(w3_TRW)
w4_dstrb = torch.clone(w4_TRW)
w0_dstrb_0 = torch.clone(w0_TRW)
w1_dstrb_0 = torch.clone(w1_TRW)
w2_dstrb_0 = torch.clone(w2_TRW)
w3_dstrb_0 = torch.clone(w3_TRW)
w4_dstrb_0 = torch.clone(w4_TRW)
hlayer_temp_TRW = torch.zeros((iter_needed,batch_size,50), device=DEVICE)
hlayer2_temp_TRW = torch.zeros((iter_needed,batch_size,50), device=DEVICE)
hlayer3_temp_TRW = torch.zeros((iter_needed,batch_size,50), device=DEVICE)
hlayer4_temp_TRW = torch.zeros((iter_needed,batch_size,50), device=DEVICE)
delta5_temp_TRW = torch.zeros((iter_needed,batch_size,50), device=DEVICE)
delta4_temp_TRW = torch.zeros((iter_needed,batch_size,50), device=DEVICE)
delta3_temp_TRW = torch.zeros((iter_needed,batch_size,50), device=DEVICE)
delta2_temp_TRW = torch.zeros((iter_needed,batch_size,50), device=DEVICE)
delta1_temp_TRW = torch.zeros((iter_needed,batch_size,50), device=DEVICE)

for j in range(epoch_num):
    order = torch.arange(0,batch_num+1)
    for i in range(batch_num):
        z1_TRW = torch.matmul(inputL[i,:,:],w0_TRW) + b1_TRW
        hl1_TRW = torch.tanh(ReLU(z1_TRW))
        z2_TRW = torch.matmul(hl1_TRW,w1_TRW) + b2_TRW
        hl2_TRW = torch.tanh(ReLU(z2_TRW))
        z3_TRW = torch.matmul(hl2_TRW,w2_TRW) + b3_TRW
        hl3_TRW = torch.tanh(ReLU(z3_TRW))
        z4_TRW = torch.matmul(hl3_TRW,w3_TRW) + b4_TRW
        hl4_TRW = torch.tanh(ReLU(z4_TRW))
        z5_TRW = torch.matmul(hl4_TRW,w4_TRW) + b5_TRW
        outL_TRW = torch.tanh(ReLU(z5_TRW))

        E_TRW = xTrainNhot[i*batch_size:(i+1)*batch_size,:]-outL_TRW
        delta5_TRW = torch.multiply(E_TRW,tanhReLUdr(z5_TRW))
        delta4_TRW = torch.multiply(torch.matmul(delta5_TRW,B4),tanhReLUdr(z4_TRW))
        delta3_TRW = torch.multiply(torch.matmul(delta4_TRW,B3),tanhReLUdr(z3_TRW))
        delta2_TRW = torch.multiply(torch.matmul(delta3_TRW,B2),tanhReLUdr(z2_TRW))
        delta1_TRW = torch.multiply(torch.matmul(delta2_TRW,B1),tanhReLUdr(z1_TRW))

        hlayer_temp_TRW[j*batch_num+i,:,:] = hl1_TRW
        hlayer2_temp_TRW[j*batch_num+i,:,:] = hl2_TRW
        hlayer3_temp_TRW[j*batch_num+i,:,:] = hl3_TRW
        hlayer4_temp_TRW[j*batch_num+i,:,:] = hl4_TRW

        delta5_temp_TRW[j*batch_num+i,:,:] = delta5_TRW
        delta4_temp_TRW[j*batch_num+i,:,:] = delta4_TRW
        delta3_temp_TRW[j*batch_num+i,:,:] = delta3_TRW
        delta2_temp_TRW[j*batch_num+i,:,:] = delta2_TRW
        delta1_temp_TRW[j*batch_num+i,:,:] = delta1_TRW

        w4_TRW += lr*torch.matmul(hl4_TRW.T,delta5_TRW)
        w3_TRW += lr*torch.matmul(hl3_TRW.T,delta4_TRW)
        w2_TRW += lr*torch.matmul(hl2_TRW.T,delta3_TRW)
        w1_TRW += lr*torch.matmul(hl1_TRW.T,delta2_TRW)
        w0_TRW += lr*torch.matmul(inputL[i,:,:].T,delta1_TRW)

        torch.cuda.empty_cache()
        alignmentTermsK = torch.zeros((layer_num,50,50), device=DEVICE)
        for o in range(j*batch_num+i):
            alignmentTermsK[0,:,:] = alignmentTermsK[0,:,:] + (lr**2)*torch.chain_matmul(B1.T,delta2_temp_TRW[j*batch_num+i-o-1,:,:].T,\
                                                                    inputL[np.mod(j*batch_num+i-o-1,batch_num),:,:],inputL[np.mod(j*batch_num+i,batch_num),:,:].T,\
                                                                    delta2_temp_TRW[j*batch_num+i,:,:])
            alignmentTermsK[1,:,:] = alignmentTermsK[1,:,:] + (lr**2)*torch.chain_matmul(B2.T,delta3_temp_TRW[j*batch_num+i-o-1,:,:].T,\
                                                                    hlayer_temp_TRW[j*batch_num+i-o-1,:,:],hlayer_temp_TRW[j*batch_num+i,:,:].T,\
                                                                    delta3_temp_TRW[j*batch_num+i,:,:])
            alignmentTermsK[2,:,:] = alignmentTermsK[2,:,:] + (lr**2)*torch.chain_matmul(B3.T,delta4_temp_TRW[j*batch_num+i-o-1,:,:].T,\
                                                                    hlayer2_temp_TRW[j*batch_num+i-o-1,:,:],hlayer2_temp_TRW[j*batch_num+i,:,:].T,\
                                                                    delta4_temp_TRW[j*batch_num+i,:,:])
            alignmentTermsK[3,:,:] = alignmentTermsK[3,:,:] + (lr**2)*torch.chain_matmul(B4.T,delta5_temp_TRW[j*batch_num+i-o-1,:,:].T,\
                                                              hlayer3_temp_TRW[j*batch_num+i-o-1,:,:],hlayer3_temp_TRW[j*batch_num+i,:,:].T,\
                                                              delta5_temp_TRW[j*batch_num+i,:,:])
        w4_dstrb += alignmentTermsK[3,:,:] + lr*torch.matmul(torch.tanh(ReLU((torch.matmul(w3_dstrb_0.T,hl3_TRW.T) + b4_TRW.T))),delta5_TRW)
        w3_dstrb += alignmentTermsK[2,:,:] + lr*torch.matmul(torch.tanh(ReLU((torch.matmul(w2_dstrb_0.T,hl2_TRW.T) + b3_TRW.T))),delta4_TRW)
        w2_dstrb += alignmentTermsK[1,:,:] + lr*torch.matmul(torch.tanh(ReLU((torch.matmul(w1_dstrb_0.T,hl1_TRW.T) + b2_TRW.T))),delta3_TRW)
        w1_dstrb += alignmentTermsK[0,:,:] + lr*torch.matmul(torch.tanh(ReLU((torch.matmul(w0_dstrb_0.T,inputL[i,:,:].T) + b1_TRW.T))),delta2_TRW)

        J_TRW = torch.ones((1,batch_size), device=DEVICE)
        b5_TRW += torch.squeeze(lr*torch.matmul(J_TRW,delta5_TRW))
        b4_TRW += torch.squeeze(lr*torch.matmul(J_TRW,delta4_TRW))
        b3_TRW += torch.squeeze(lr*torch.matmul(J_TRW,delta3_TRW))
        b2_TRW += torch.squeeze(lr*torch.matmul(J_TRW,delta2_TRW))
        b1_TRW += torch.squeeze(lr*torch.matmul(J_TRW,delta1_TRW))

        angle_TRW[0,0,j*batch_num+i] =  angleFunc(w1_TRW,B1.T)
        angle_TRW[0,1,j*batch_num+i] =  angleFunc(w2_TRW,B2.T)
        angle_TRW[0,2,j*batch_num+i] =  angleFunc(w3_TRW,B3.T)
        angle_TRW[0,3,j*batch_num+i] =  angleFunc(w4_TRW,B4.T)

        angle_dstrb[0,0,j*batch_num+i] = angleFunc(w1_dstrb,B1.T)
        angle_dstrb[0,1,j*batch_num+i] = angleFunc(w2_dstrb,B2.T)
        angle_dstrb[0,2,j*batch_num+i] = angleFunc(w3_dstrb,B3.T)
        angle_dstrb[0,3,j*batch_num+i] = angleFunc(w4_dstrb,B4.T)

fig ,ax = plt.subplots(1,1,figsize=(10,10))
ax.plot(np.arange(0,len(angle_TRW[0,0,:])),np.mean(angle_TRW[:,0,:].cpu().numpy(),axis=(0)),color='orange')
ax.plot(np.arange(0,len(angle_TRW[0,0,:])),np.mean(angle_TRW[:,1,:].cpu().numpy(),axis=(0)),color='green')
ax.plot(np.arange(0,len(angle_TRW[0,0,:])),np.mean(angle_TRW[:,2,:].cpu().numpy(),axis=(0)),color='red')
ax.plot(np.arange(0,len(angle_TRW[0,0,:])),np.mean(angle_TRW[:,3,:].cpu().numpy(),axis=(0)),color='gray')
ax.plot(np.arange(0,len(angle_dstrb[0,0,:])),np.mean(angle_dstrb[:,0,:].cpu().numpy(),axis=(0)),':',color='orange')
ax.plot(np.arange(0,len(angle_dstrb[0,0,:])),np.mean(angle_dstrb[:,1,:].cpu().numpy(),axis=(0)),':',color='green')
ax.plot(np.arange(0,len(angle_dstrb[0,0,:])),np.mean(angle_dstrb[:,2,:].cpu().numpy(),axis=(0)),':',color='red')
ax.plot(np.arange(0,len(angle_dstrb[0,0,:])),np.mean(angle_dstrb[:,3,:].cpu().numpy(),axis=(0)),':',color='gray')
ax.set_title('Ignoring non-linearity')
ax.set_ylabel('Angle')
ax.set_xlabel('Iteration')
ax.legend([r'Angle between $W_{1}$ & ${B_1}^{T}$',r'Angle between $W_{2}$ & ${B_2}^{T}$',r'Angle between $W_{3}$ & ${B_3}^{T}$',\
           r'Angle between $W_{4}$ & ${B_4}^{T}$',r'Angle between $W_{1}disturbed$ & ${B_1}^{T}$',r'Angle between $W_{2}disturbed$ & ${B_2}^{T}$',\
           r'Angle between $W_{3}disturbed$ & ${B_3}^{T}$',r'Angle between $W_{4}disturbed$ & ${B_4}^{T}$'])
ax.set_xlim([0, 600])
ax.set_ylim([70, 95])

import argparse
import os
import sys
import time
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch

def CS(x,y):
    return np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y))
    
def TSNE_model(model, device, loader,loss_idx):
    model.eval()
    total_number = 0
    total_correct = 0
    label = []
    fea = []
    with torch.no_grad():
        for datapoint in loader:
            padded_text, attention_masks, labels = datapoint
            padded_text = padded_text.to(device)
            attention_masks = attention_masks.to(device)
            target = labels
            for i in target:
                label.append(i)
            
            labels = labels.to(device)              
            output, feature = model(padded_text, attention_masks)  
            
            feature = feature.cpu().detach() 
            for i in feature:
                fea.append(i.numpy())
                
    feature = np.array(fea)  
        
    features = (feature - np.mean(feature, axis=0)) / np.std(feature, axis=0)
    #tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne = TSNE(learning_rate = 100)
    embeddings = tsne.fit_transform(features)
    common = []
    for i in loss_idx[:10]:   
        for inx,val in enumerate(embeddings):
            temp = CS(val,embeddings[i])
            if temp > 0.6:
                common.append(inx)    

    re_common = [x for x in set(common) if common.count(x)==10]
    print(re_common)
    print(len(re_common))
    
    cm = mpl.colors.ListedColormap(['b','r'])
    plt.scatter(embeddings[:, 0], embeddings[:, 1],c = label, cmap = cm, s = 8)
    
    #plt.savefig('output.png',dpi = 500)
    
    plt.show()


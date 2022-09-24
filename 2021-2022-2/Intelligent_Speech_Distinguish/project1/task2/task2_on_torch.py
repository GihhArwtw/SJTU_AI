"""
# Voice Activities Detection - TASK 02 (on Pytorch)
Due: May 8th
"""

'''
=============================================
            Hyperparameters
=============================================
'''
"""
For frame segmentation:
    > frame_size:    the length of each frame. The unit is "second".
    
    > frame_shift:   the shift of frame segmentation, i.e. the shift of time between two consecutive frames on
                            the original sound. The unit is "second".

______________________________________________________

Windowing:      We apply window function on each frame to help STFT become finite.
                
                "Windowing" should be one of the following.
                
                        None, "hamming", "hanning", "bartlett", "blackman".
                        
                When Windowing==None, we do not apply any window mentioned above, i.e. the frame is processed
                    with rectangular window.
                

Pre-emphasis:   We preprocess the original signal by pre-emphasis. It is said that this method can enhance 
                    high-frequency components. The equation is as follows.
              
                x'[n] = x[n] - a*x[n-1].
                
______________________________________________________

For MFCC and FBank:

    > Num_filters:    the number of Mel filters. If set to None, we'll use the default, i.e. 40.
    
    > Minmax_Ceps:    a tuple or a list. We'll perserve [Minmax_Ceps[0]:(Minmax_Ceps[1]-1)] in the DCT of FBank to get MFCC.
                      If set to None, we'll use the default, i.e. (1,13).
"""


from task2_utils import frame_size, frame_shift, Windowing, Emph, alpha, Num_filters, Minmax_Ceps


"""
_______________________________________________________

For Machine Learning Traning.

    > Training_epoch:  the number of training epoches.
    
    > Learning_rate:   the learning rate in gradient descent.

    > Negative_Sample_Rate:   the ratio of negative samples generated and given samples.

"""

Training_epoch = 5

Learning_rate = 0.05

# Negative_Sample_Rate = 1

"""
_______________________________________________________

Regularization in the Loss.

    Without regularizing, the model will output a result with too many "1"s since the accuracy is high and the loss is small.
    
    Thus, we design a regularization as follows. Let the output of the model be "out".
    
        Reg(out) = tan((out-0.5)*pi)^2.
        
        Loss(out) = BCELoss(out) + gamma * Reg(out)
        
    >   gamma:   Control the scale of the regularization term. Set gamma=0 to discard the regularization term.

"""

gamma = 0.1




'''
=============================================
                Label Loading
=============================================
'''

import sys
sys.path.append('..')
from vad_utils import parse_vad_label, prediction_to_vad_label, read_label_from_file
from evaluate  import get_metrics



'''
=============================================================================
            Frame Segmentation;  STFT, MFCC, FBank Definition
=============================================================================

'''

import pydub
import numpy as np
from task2_utils import Frame_Seg, stft_power, mel_filter, 


'''
======================================================
                    Dataset Loading
======================================================

'''

from task2_utils import AudioDataset, train_fbank, train_mfcc, dev_fbank, dev_mfcc



'''
=============================================
        Neural Network Definition
=============================================
'''

import torch
import torch.nn as nn

#_______________________________________
# 
# MFCC + DNN

if (Minmax_Ceps is None):
    num_in = num_2nd = 12
else:
    num_in  = int(Minmax_Ceps[1]-Minmax_Ceps[0])
    num_2nd = int(min(num_in,24))

class CNN_mfcc(nn.Module):
    def __init__(self):
        super(DNN_mfcc, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(num_in, num_2nd),
            nn.ELU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(num_2nd, 4),
            nn.ELU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(4,1)
        )
        
    def forward(self, x):   # DNN_mfcc
        return torch.sigmoid(self.network(x))
        
        
#_______________________________________
# 
# FBank + DNN

if Num_filters is None:
    Num_filters = 40
    
Num_filters = int(Num_filters)

class DNN_fbank(nn.Module):
    def __init__(self):
        super(DNN_fbank, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(Num_filters, int(Num_filters/4)),
            nn.ELU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(int(Num_filters/4), 8),
            nn.ELU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(8,4),
            nn.ELU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(4,1)
        )
        
    def forward(self, x):   # DNN_fbank
        return torch.sigmoid(self.network(x))
    

    
#_______________________________________
# 
# FBank + pseudo-RNN

# NOT USED YET.

class RNN_fbank(nn.Module):
    def __init__(self):
        super(RNN_fbank, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(int(Num_filters*2), int(Num_filters/2)),
            nn.ELU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(int(Num_filters/2), 4),
            nn.ELU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(4,1)
        )
        
    def forward(self, x):
        return torch.sigmoid(self.network(x))




'''
================================================
================================================
            MAIN (Based on pytorch)
------------------------------------------------
          (Voice Activity Detection)
================================================
================================================
'''

def valid(model, valid_loader):
    """
    Test the current model on the validation set. Will use the GPU if such device exists.
    
    [Arguments]
    
        - model:          a neural network model to be tested on validation set.
                          Should be an instance of a certain nerual network class.
        
        - valid_loader:   the validation set on which the model will be tested.
                          Should be a torch dataset loaded by "torch.utils.data.DataLoader".
                          
    [Return]
    
        - (auc, eer):      Area under curve and Equal error rate.
        
        - accuracy:        The accuracy.
        
        - ones:            The probability that 1 occurs.
        
        - avg_loss:        The average loss on the validation set.
        
    """
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    labels = None
    preds = None
    loss = torch.tensor(0.).to(device)
    with torch.no_grad():
        for data, target in valid_loader:
            data = data[0].to(device)
            label = target[0].to(device)
            pred = model(data).reshape(-1)
            loss += nn.BCELoss()(pred, label.float())
            
            if (labels is None):
                labels = label.detach().cpu()
                preds  = pred.detach().cpu()
            else:
                labels = torch.cat([labels, label.detach().cpu()],dim=0)
                preds  = torch.cat([preds, pred.detach().cpu()],dim=0)
    
    auc_err = get_metrics(preds, labels)
    
    preds = (preds>=0.5).float()
    accuracy = torch.mean(1-torch.abs(labels - preds))
    
    loss = loss/len(valid_loader.dataset)
    
    print(f"'1' occurred with possibility {float(torch.mean(preds))}")
    
    return auc_err, accuracy, float(torch.mean(preds)), loss
      

    
def trained(network, train_data, valid_data, model_name):
    """
    Train the nerual network "network" on "train_data". Will use the GPU if such device exists.
    
    [Arguments]
    
        - network:    the class of neural network to be trained. 
                      Should be a class name.
        
        - train_data: the training dataset on which the network will be trained.
                      Should be a torch dataset, i.e. an instance of a derived class from "torch.utils.data.Dataset".
                      
        - valid_data: the validation dataset on which the network will be tested. Also might be called the "development set".
                      Should be a torch dataset, i.e. an instance of a derived class from "torch.utils.data.Dataset".
                      
        - model_name: the name used in the model storage (as a file).
    
    [Return]
        - model:      the best trained model.
        
        - (auc, err, acc, ones):    the area under curve, the equal error rate, the accuracy, and the possibility that 1 occurs
                                    of the model.

    """
    
    from torch.utils.data import DataLoader
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = DataLoader(train_data, batch_size=1, shuffle=True)
    valid_dataset = DataLoader(valid_data, batch_size=1, shuffle=True)
    #  NOTE: Since each audio file is of differnte length, we strongly recommend each batch contain one file only. 
    #        Otherwise, pytorch will draw an exception.

    model = network().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate, weight_decay=5e-4)
    
    aucs = None
    alls = []
    for epoch in range(Training_epoch):
        
        print(f"Training Epoch {epoch}: ")
        model.train()
        optimizer.zero_grad()
    
        batch_index = 0
        for batch, (data, target) in enumerate(train_dataset):
            data = data[0].to(device)
            label = target[0].to(device)
            pred = model(data)

            pred = pred.reshape(-1)
            loss = nn.BCELoss()(pred, label.float())
                
            loss.backward()
            optimizer.step()
            
            batch_index += 1
            if (batch_index%600==0):
                pred = pred.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                (auc, err) = get_metrics(pred,label)
                
                print('[{}/{} ({:2.0f}%)]\tLoss: {:.6f}\tAUC: {:.6f}\tERR: {:.6f}\tpossibility of 1: {:.6f} '.format(
                        batch_index, len(train_dataset.dataset), 100.*batch_index/len(train_dataset.dataset),
                        loss.item(), auc, err, np.mean((pred>=0.5))))

        torch.save(model, str(model_name)+'model'+str(epoch)+'.pth')
            
        model.eval()
        (auc,err), acc, ones, avg_loss = valid(model, valid_dataset)

        print("\nTesting on dev.\n\tAverage Loss: {:.6f}\n\tAUC: {:.6f}\tERR: {:.6f}\n\tAccuracy: {:.6f}\n".format(avg_loss, auc, err, acc))
        
        if aucs is None:
            aucs = auc
        else:
            aucs = np.append(aucs, auc)
            
        alls += [(auc, err, acc, ones)]
    
    index = aucs.argmax(axis=0)
    model = torch.load(str(model_name)+'model'+str(index)+'.pth')
    
    return model, alls[index], index



# mfcc + DNN:
mfcc_net, (auc_mfcc, err_mfcc, acc_mfcc, one_mfcc), epoch_m = trained(DNN_mfcc, train_mfcc, dev_mfcc, "mfcc")
torch.save(mfcc_net, "mfcc_dnn_best_on_torch.pth")

# fbank + DNN:
fbank_net, (auc_fbank, err_fbank, acc_fbank, one_fbank), epoch_f = trained(DNN_fbank, train_fbank, dev_fbank, "fbank")
torch.save(fbank_net, "fbank_dnn_best_on_torch.pth")


mfcc_net = torch.load("mfcc_dnn_best_on_torch.pth")
print("===============================\nDNN with MFCC:\n===============================\n")
from torchsummary import summary
summary(mfcc_net, (1,12))
print()
print("AUC: {:.6f}\t ERR:{:.6f}\t Accuracy:{:.6f}\t The possibility of 1's occurrence: {:.6f}".format(
        auc_mfcc, err_mfcc, acc_mfcc, one_mfcc))


fbank_net = torch.load("fbank_dnn_best_on_torch.pth")
print("===============================\nDNN with FBank:===============================\n")
from torchsummary import summary
summary(fbank_net, (1,40))
print()
print("AUC: {:.6f}\t ERR:{:.6f}\t Accuracy:{:.6f}\t The possibility of 1's occurrence: {:.6f}".format(
        auc_fbank, err_fbank, acc_fbank, one_fbank))

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
"""

frame_size = 0.032
frame_shift = 0.008

"""
______________________________________________________

Windowing:      We apply window function on each frame to help STFT become finite.
                
                "Windowing" should be one of the following.
                
                        None, "hamming", "hanning", "bartlett", "blackman".
                        
                When Windowing==None, we do not apply any window mentioned above, i.e. the frame is processed
                    with rectangular window.
                

Pre-emphasis:   We preprocess the original signal by pre-emphasis. It is said that this method can enhance 
                    high-frequency components. The equation is as follows.
              
                x'[n] = x[n] - a*x[n-1].

"""

Windowing = "hanning"
Emph = True
alpha = .9


"""
______________________________________________________

For MFCC and FBank:

    > Num_filters:    the number of Mel filters. If set to None, we'll use the default, i.e. 40.
    
    > Minmax_Ceps:    a tuple or a list. We'll perserve [Minmax_Ceps[0]:(Minmax_Ceps[1]-1)] in the DCT of FBank to get MFCC.
                      If set to None, we'll use the default, i.e. (1,13).

"""

Num_filters = None   #40

Minmax_Ceps = None   #(1,13)


"""
_______________________________________________________

For Machine Learning Traning.

    > Training_epoch:  the number of training epoches.
    
    > Learning_rate:   the learning rate in gradient descent.

"""

Training_epoch = 5

Learning_rate = 0.01




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
=============================================
            Frame Segmentation
=============================================
'''

import pydub
import numpy as np


def Frame_Seg(audio_path, frame_size=frame_size, frame_shift=frame_shift, window="hamming", emph=True, emph_alpha=.9):
    """
    Frame Segmentation.
    
    [Arguments]
    
        - audio_path:    The path of the sound file to be processed.
                         Should be a path.
                     
        - frame_size:    The length of each frame. The unit is "second".
                         Should be a float.
                     
        - frame_shift:   The shift of frame segmentation, i.e. the shift of time between two consecutive frames on the 
                             original sound. The unit is "second".
                         Should be a float.

        - window:        The window implemented on all frames. Defaultly set to "hamming". If the requirement is rectangluar
                             window, set window to "None".
                         Should be one of the following.
                             None, "hamming", "hanning", "bartlett", "blackman".
                     
        - emph:          Whether we apply pre-emphasis on the original sound or not. Defaultly set to "True". To pre-emphasize
                             the raw wave file, use "emph=True". The equation is given below in the explanation of "emph_alpha".
                         Should be a boolean.
                     
        - emph_alpha:    The scale of pre-emphasis. The pre-emphasis equation is as follow.
                        
                         x'[n] = x[n] - alpha*x[n-1].
                         
                         Should be a float between 0.9 and 1.
                     
    [Return]
    
        Returns frames, sample_rate.
    
        - frames:        A list. Each element is a frame.
    
        - sample_rate:   The sample rate of the audio file.
    
    """
    audio = pydub.AudioSegment.from_wav(audio_path)
    sample_rate = audio.frame_rate
    sound = np.array(audio.get_array_of_samples())
    
    if (emph):
        sound = np.append(sound[0], sound[1:]-emph_alpha*sound[:-1])
    
    shift = int(frame_shift * sample_rate)
    size  = int(frame_size  * sample_rate)
    frames = []
    
    if (window is "hamming"):
        wndw = np.hamming(size)
    elif (window is "hanning"):
        wndw = np.hanning(size)
    elif (window is "bartlett"):
        wndw = np.bartlett(size)
    elif (window is "blackman"):
        wndw = np.blackman(size)
    else:
        wndw = np.ones(size)
    
    for i in range(0,len(sound),shift):
        x = sound[i:i+size]
        if (len(x)<size):
            x = np.pad(x,(0,size-len(x)),'constant',constant_values=0)
        for j in range(size):
            x[j] = x[j]*wndw[j]
        x = x.tolist()
        frames = frames + [x]
        
    return frames, sample_rate




'''
=============================================
        STFT, MFCC, FBank Definition
=============================================
'''

from scipy.fftpack import dct

def stft_power(frames, nfft=512):
    """
    Compute the power spectrum of the framed audio "frames".
    
    """
    
    spec = np.fft.rfft(frames, nfft)
    spec = np.abs(spec)
    return ( (spec**2) / float(nfft) )


def mel_filter(pow_spec, sample_rate, num_filters=40, nfft=512, min_ceps=1, max_ceps=13):
    """
    MFCC (Mel-Frequency Cepstral Coefficients) and Mel-Filter FBank (Filter Bank) Computation.
    
    [Arguments]
    
        - pow_spec:      the power spectrum of the framed audio file.
        
        - sample_rate:   the sample rate of the audio file.
        
        - num_filters:   the number of Mel filters we want. Defaultly set to 40.
        
        - nfft:          the number of points in STFT.
        
        - min_ceps:      the minimum cepstral coefficient we want to use. We'll discard coefficients
                              whose cepstral is lower than min_ceps Defaultly set to 1.
        
        - max_ceps:      the maximum cepstral coefficient we want to use. We'll discard coefficients
                              whose cepstral is higher than max_ceps Defaultly set to 13.
        
    [Return]
    
        Returns mfcc, filter_banks
        
        - mfcc:         the MFCC of "frames".
        
        - filter_banks: the Mel-Filter-Moduled Filter Bank of "frames".
        
    """
      
    freq2mel = lambda n: 2595*np.log10(1.+n/700.)
    mel2freq = lambda n: 700*(np.power(10,n/2595.)-1.)
    
    mel_min = 0.
    mel_max = freq2mel(sample_rate/2.)
    mels = np.linspace(mel_min, mel_max, num_filters+2)
    
    # Mel-Filters Definition.
    freqs = mel2freq(mels)
    freqs = np.floor((nfft+1)*freqs/sample_rate)
    
    mel_filt = np.zeros( (num_filters, int(nfft/2+1)) )
    for i in range(num_filters):
        left = freqs[i]
        cntr = freqs[i+1]
        rght = freqs[i+2]
        
        for f in range(int(left),int(cntr)):
            mel_filt[i,f] = (f-left)/float(cntr-left)
        for f in range(int(cntr),int(rght)):
            mel_filt[i,f] = (rght-f)/float(rght-cntr)
            
    # FBank Computation.
    filter_banks = np.dot(pow_spec, mel_filt.T)
    filter_banks = np.where(filter_banks==0, np.finfo(float).eps, filter_banks)
    filter_banks = 20*np.log10(filter_banks)
    
    # MFC Computation.
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, min_ceps:max_ceps]
    
    # Normalization.
    filter_banks -= np.mean(filter_banks, axis=0)
    mfcc -= np.mean(mfcc, axis=0)
    
    return filter_banks, mfcc
    
    
    

'''
==========================================================
            Dataset Generation from WAV Files
==========================================================
'''

import pandas as pd


def data_gen(the_dir):
    """
    Compute the MFCC and FBank of the wav files in the development set and training set.
    Write the MFCC and FBank into files to prepare for dataset loading.
    
    [Arguments]
        
        - the_dir:    Should be either "dev" or "train". It determines which set we will generate MFCC and FBank dataset.
        
    [Return]
    
        Nothing.
        
    """
    
    dataset_labels = read_label_from_file(f"../data/{the_dir}_label.txt", frame_size=frame_size, frame_shift=frame_shift)
    
    for key in dataset_folder.keys():
        frames, fs = Frame_Seg(f"../wavs/train/{key}.wav", window=Windowing, emph=Emph, emph_alpha=alpha)
        break
    
    nfft = 512
    while (nfft<fs*frame_size):   nfft = nfft << 1;
    
    labels = []
    files  = []
    for key in dataset_labels.keys():
        frames, fs = Frame_Seg(f"../wavs/{the_dir}/{key}.wav", window=Windowing, emph=Emph, emph_alpha=alpha)

        spec = stft_power(frames, nfft=nfft)
        fbank, mfcc = mel_filter(spec, fs, nfft=nfft)
        
        label = dataset_labels[key]
        label_pad = np.pad(label, (0, np.maximum(len(frames) - len(label), 0)))[:len(frames)]
    
        df = pd.DataFrame(mfcc)
        df.to_csv(f'./{the_dir}_gen/{key}_mfcc.csv',index=False)
    
        df = pd.DataFrame(fbank)
        df.to_csv(f'./{the_dir}_gen/{key}_fbank.csv',index=False)
        
        labels = labels + [label_pad.tolist()]
        files  = files  + [key]
    
    
    df = pd.DataFrame(labels)
    df.to_csv(f'./{the_dir}_gen/{the_dir}_gen_labels.csv',index=False)
    
    df = pd.DataFrame(files)
    df.to_csv(f'./{the_dir}_gen/{the_dir}_filenames.csv',index=False)
    
    return




# Only need to run for once before training.

# Generate FBank & MFCC Dataset of the Training Dataset.
# data_gen("train")

# Generate FBank & MFCC Dataset of the Development Dataset.
# data_gen("dev")




'''
==========================================
            Dataset Preparation
==========================================

    Construct our dataset with the help of pytorch.
    
'''

import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, audio_dir, annotations_file, label_file, mode="fbank"):
        self.audio_dir = audio_dir
        self.audios = pd.read_csv(audio_dir+"/"+annotations_file).values
        self.labels = pd.read_csv(audio_dir+"/"+label_file).values
        self.mode = mode
        
    def __len__(self):
        return len(self.audios)

    def __getitem__(self, index):
        audio_path = self.audio_dir+'/'+self.audios[index][0]+"_"+self.mode+".csv"
        audio = pd.read_csv(audio_path)
        audio = torch.Tensor(audio.values)
        label = self.labels[index]
        label = torch.Tensor([label[label == label]]).long().reshape(-1)
        return audio, label
                                  

train_fbank = AudioDataset("./train_gen","train_filenames.csv","train_gen_labels.csv","fbank")
train_mfcc  = AudioDataset("./train_gen","train_filenames.csv","train_gen_labels.csv","mfcc")
                                  
dev_fbank = AudioDataset("./dev_gen","dev_filenames.csv","dev_gen_labels.csv","fbank")
dev_mfcc  = AudioDataset("./dev_gen","dev_filenames.csv","dev_gen_labels.csv","mfcc")
                                  

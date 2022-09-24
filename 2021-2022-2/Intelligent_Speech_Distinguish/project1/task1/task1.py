"""
[Voice Activities Detection] - task 01
Due: May 8th
___________________________________________
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
    > Smooth:        whether apply moving average smoothing before frame segmentation or not.
"""

frame_size = 0.032
frame_shift = 0.008
Smooth = False


"""
_________________________________________________________

For Fixed Threshold.

    We set two thresholds for short-time energy, i.e. to use two-threshold method.
    
    > alpha:  Fixed high threshold for short-time energy.
    
    > beta:   Fixed low threshold for short-time energy.
    
    For ZCR, we test one-threshold method and two-threshold method. We only consider the sound consecutive to 
            sections with high short-time energy.
    
    - To use one-threshold method,
    
        > gamma:  Fixed threshold for ZCR.
    
        > In this case, set delta to "None".
        
    - To use two-threshold method,
    
        > gamma:  Fixed high threshold for ZCR.
    
        > delta:  Fixed low threshold for ZCR.
    
_________________________________________________________

For Flexible Threshold. The threshold in this case is a ratio of the actual threshold of the "normalized" value.

    We set two thresholds for short-time energy, i.e. to use two-threshold method.

    > alpha:  The ratio of the high energy threshold of the "normalized" energy. Regard the average energy as 1
                    and all energy range from 0 to 2.     
                    
    > beta:   The ratio of the low energy threshold of the "normalized" energy. Regard the average energy as 1
                    and all energy range from 0 to 2.


    For ZCR, we test one-threshold method and two-threshold method. We only consider the sound consecutive to 
            sections with high short-time energy.
    
    - To use one-threshold method, 'delta' should be set to "None".
        > gamma:  The ratio of the ZCR threshold of the "normalized" ZCR. Regard the average ZCR as 1 and all ZCRs 
                        range from 0 to 2.
                        
        > delta:  In this case, 'delta' should be "None".
        
    - To use two-threshold method, 
        > gamma:  The ratio of the high ZCR threshold of the "normalized" ZCR. Regard the average ZCR as 1 and all
                        ZCRs range from 0 to 2.
                        
        > delta:  The ratio of the low ZCR threshold of the "normalized" ZCR. Regard the average ZCR as 1 and all
                        ZCRs range from 0 to 2.
                
"""

# Emph=False, i.e without pre-emphasis. Best flexible threshold.
'''
Method = "Flexible"

alpha = 1.0
beta = 0.8
gamma = .9
delta = None
'''

# Emph=False, i.e without pre-emphasis. Best fixed threshold.
'''
Method = "Fixed"

alpha = 300000000
beta  = 7500000
gamma = 100
delta = None
'''

# Emph=True, i.e. with pre-emphasis. Best fixed threshold.
Method = "Fixed"

alpha = 30000000
beta  = 750000
gamma = 1000
delta = None

"""
______________________________________________________


Pre-emphasis:   We preprocess the original signal by pre-emphasis. It is said that this method can enhance 
                    high-frequency components. The equation is as follows.
              
                x'[n] = x[n] - a*x[n-1].

"""

Emph = True
a = .9




'''
=============================================
                Label Loading
=============================================
'''

import sys
sys.path.append('..')
from vad_utils import parse_vad_label, prediction_to_vad_label, read_label_from_file
from evaluate  import get_metrics

targets = read_label_from_file("../data/dev_label.txt", frame_size=frame_size, frame_shift=frame_shift)




'''
=============================================
        ZCR and Frame Energy Definition
=============================================
'''

import numpy as np

def sgn(x):
    return np.round((np.sign(x)+1.5)/2)

def ZCR(frame):
    """
    Compute the zero-crossing rate of the sound frame.
    return the ZCR of the frame.
    """
    count = np.abs(sgn(frame[:-1]) - sgn(frame[1:])).sum() / 2
    return count

def Frame_Energy(frame):
    """
    Compute the energy of the sound frame.
    return the energy of the frame.
    """
    energy = 0
    for x in frame:
        energy = energy + x*x
    return energy




'''
================================================
      Voice Activity Detection Definition
================================================
'''

from math import exp

def std(x):          # in case "np.std" will overflow.
    mean_x = np.sum(x) / float(len(x))
    sigma_x = np.sqrt(np.var(x))
    if (sigma_x>0):   x = ( np.array(x) - mean_x ) / sigma_x
    x = 2./ (1. + np.exp(-x))
    return x


def Endpoint_Detect(frames, energy_alpha=0.75, energy_beta=0.2, zcr_gamma=1, method="Fixed"):
    """
    Detect the endpoint of sounds with ZCR, frame energy and base frequencies.
    We use two thresholds to determine endpoint by frame energy.
    
    [Arguments]
    - frames:        The frames of the original sound file whose voice activities we want to detect.
                     Should be a list or an ndarray.
                     
    - energy_alpha:  The high energy threshold of short-time energy.
                     Should be a float.
                     
    - energy_beta:   The low energy threshold of short-time energy.
                     Should be a float.
                     
    - zcr_gamma:     The threshold of ZCR.
                     Should be a float.
                     
    - method:        Whether the threshold is fixed or flexible. If method is "Flexible", we normalize 
                        the original signal.
                     
    [Return]
    - Activities:    A list. Activities store the activity of the current frame, =0 or =1.
    """
    
    energy = [ Frame_Energy(x) for x in frames ]
    if (method is "Flexible"):
        energy = std(energy).tolist()
        
    # for i in range(len(energy)): print(energy[i]," ",label_pad[i])
    
    threshold_high = energy_alpha
    threshold_low  = energy_beta
    
    i = 0
    endpoints = []
    Activities = np.zeros(len(frames))
    while (i<len(frames)):
        while (i<len(frames) and (energy[i]<threshold_high)):  i = i+1
        if (i==len(frames)):  break
        begin = i
        i = i+1
        while (i<len(frames) and (energy[i]>=threshold_low)):
            Activities[i] = 1
            i = i+1
        endpoints = endpoints + [[begin,i-1]]
        
    zcrs = [ ZCR(x) for x in frames ]
    if (method is "Flexible"):
        zcrs = std(zcrs)
    
    threshold_zcr = zcr_gamma
    for i in range(len(endpoints)):
        j = endpoints[i][0]-1
        while ((j>=0) and (Activities[j]==0) and (zcrs[j]>=threshold_zcr)):
            Activities[j] = 1
            j = j-1        
        
        j = endpoints[i][1]+1
        while ((j<len(frames)) and (Activities[j]==0) and (zcrs[j]>=threshold_zcr)):
            Activities[j] = 1
            j = j+1
    
    return Activities


def Endpoint_Detect_zcr2(frames, energy_alpha=0.75, energy_beta=0.2, zcr_gamma=1, zcr_delta=0.7, method=Method):
    """
    Detect the endpoint of sounds with ZCR, frame energy and base frequencies.
    We use two thresholds to determine endpoint by frame energy.
    
    [Arguments]
    - frames:        The frames of the original sound file whose voice activities we want to detect.
                     Should be a list or an ndarray.
                     
    - energy_alpha:  The high energy threshold of short-time energy.
                     Should be a float.
                     
    - energy_beta:   The low energy threshold of short-time energy.
                     Should be a float.
                     
    - zcr_gamma:     The high threshold of ZCR.
                     Should be a float.
                     
    - zcr_delta:     The high threshold of ZCR.
                     Should be a float.       
                     
    - method:        Whether the threshold is fixed or flexible. If method is "Flexible", we normalize 
                        the original signal.
                     Should be "Fixed" or "Flexible".
                     
    [Return]
    - Activities:    A list. Activities store the activity of the current frame, =0 or =1.
    """
    
    energy = [ Frame_Energy(x) for x in frames ]
    if (method is "Flexible"):
        energy = std(energy)
    # for i in range(len(energy)): print(energy[i]," ",label_pad[i])
    
    threshold_high = energy_alpha
    threshold_low  = energy_beta
    
    i = 0
    endpoints = []
    Activities = np.zeros(len(frames))
    while (i<len(frames)):
        while (i<len(frames) and (energy[i]<threshold_high)):  i = i+1
        if (i==len(frames)):  break
        begin = i
        i = i+1
        while (i<len(frames) and (energy[i]>=threshold_low)):
            Activities[i] = 1
            i = i+1
        endpoints = endpoints + [[begin,i-1]]
        
    zcrs = [ ZCR(x) for x in frames ]
    if (method is "Flexible"):
        zcrs = std(zcrs)
        
    zcr_high = zcr_gamma
    zcr_low  = zcr_delta
    i = 0
    while (i<len(frames)):
        while (i<len(frames) and (zcrs[i]<zcr_high)):  i = i+1
        if (i==len(frames)):  break
        i = i+1
        while (i<len(frames) and (zcrs[i]>=zcr_low)):
            Activities[i] = 1
            i = i+1    
    
    for i in range(len(endpoints)):
        j = endpoints[i][0]-1
        while ((j>=0) and (Activities[j]==0) and (zcrs[j]>=zcr_low) and (energy[j]>=threshold_low)):
            Activities[j] = 1
            j = j-1        
        
        j = endpoints[i][1]+1
        while ((j<len(frames)) and (Activities[j]==0) and (zcrs[j]>=zcr_low) and (energy[j]>=threshold_low)):
            Activities[j] = 1
            j = j+1
    
    return Activities




'''
=============================================
            Frame Segmentation
=============================================
'''

import pydub

def smoothen(frames, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(frames, window, 'same')

def Frame_Seg(audio_path, frame_size=frame_size, frame_shift=frame_shift, smooth=False, window_size=5, emph=True, emph_alpha=.9):
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
                     
    - smooth:        Whether we smoothen the original signal before frame segmentation or not. Defaultly set to "False".
                         To smoothen the signal before frame segmentation, use "smooth=True".
                     Should be a boolean.
                     
    - window_size:   The length of the window in the moving average smoothing.
                     Should be an int.
                     
    - emph:          Whether we apply pre-emphasis on the original sound or not. Defaultly set to "True". To pre-emphasize
                         the raw wave file, use "emph=True". The equation is given below in the explanation of "emph_alpha".
                     Should be a boolean.
                     
    - emph_alpha:    The scale of pre-emphasis. The pre-emphasis equation is as follow.
                    
                     x'[n] = x[n] - alpha*x[n-1].
                     
                     Should be a float between 0.9 and 1.
                     
    [Return]
    - frames:    A list. Each element is a frame.
    
    """
    audio = pydub.AudioSegment.from_wav(audio_path)
    sample_rate = audio.frame_rate
    sound = np.array(audio.get_array_of_samples())
    
    if (smooth):
        sound = smoothen(sound, window_size)
        
    if (emph):
        sound = np.append(sound[0], sound[1:]-emph_alpha*sound[:-1])
    
    shift = int(frame_shift * sample_rate)
    size  = int(frame_size  * sample_rate)
    frames = []
    for i in range(0,len(sound),shift):
        x = sound[i:i+size].tolist()
        frames = frames + [x]
    return frames




'''
================================================
================================================
                    MAIN
------------------------------------------------
          (Voice Activity Detection)
================================================
================================================
'''


preds = None
labels = None

for key in targets.keys():
    frames = Frame_Seg(f"../wavs/dev/{key}.wav", smooth=Smooth, emph=Emph, emph_alpha=a)
    label = targets[key]
    label_pad = np.pad(label, (0, np.maximum(len(frames) - len(label), 0)))[:len(frames)]
    
    
    if delta is None:
        # Two thresholds for short-time energy, one threshold for ZCR.
        pred = Endpoint_Detect(frames, energy_alpha=alpha, energy_beta=beta, zcr_gamma=gamma, method=Method)
        
    else:
        # Two thresholds for short-time energy, two thresholds for ZCR.
        pred = Endpoint_Detect_zcr2(frames, energy_alpha=alpha, energy_beta=beta, zcr_gamma=gamma, zcr_delta=delta, method=Method)
    
    
    pred_ones = np.mean(pred)
    # "pred_ones" are computed to see if all labels are assigned 0 or 1, which is not what we want, 
    #      even if the accuracy is super high.
    
    if preds is None:
        preds = pred
    else:
        preds = np.append(preds,pred)
        
    if labels is None:
        labels = label_pad
    else:
        labels = np.append(labels,label_pad)
    print(get_metrics(pred,label_pad)," ",pred_ones)



print(f"{Method} Threshold.")
print("Smooth:       ",Smooth)
print("Pre-emphasis: ",Emph)
print()
print("threshold_high of energy: ",alpha)
print("threshold_low  of energy: ",beta)
print("threshold_high of ZCR: ",gamma)
print("threshold_low  of ZCR: ",delta)

print(get_metrics(preds,labels))
print(np.abs(np.array(preds)-np.array(labels))/len(preds))



"""
===============================================
        VAD Prediction on Test Files
===============================================
"""


import os
files= os.listdir("../wavs/test")

output_file = open("test_label_task.txt","w")
for file in files:
    frames = Frame_Seg(f"../wavs/test/{file}", smooth=Smooth, emph=Emph, emph_alpha=a)
    
    if delta is None:
        # Two thresholds for short-time energy, one threshold for ZCR.
        pred = Endpoint_Detect(frames, energy_alpha=alpha, energy_beta=beta, zcr_gamma=gamma, method=Method)
        
    else:
        # Two thresholds for short-time energy, two thresholds for ZCR.
        pred = Endpoint_Detect_zcr2(frames, energy_alpha=alpha, energy_beta=beta, zcr_gamma=gamma, zcr_delta=delta, method=Method)
    
    pred_ones = np.mean(pred)
    print(pred_ones)
    # "pred_ones" are computed to see if all labels are assigned 0 or 1, which is not what we want, 
    #      even if the accuracy is super high.
    
    # Write the label into the output file.
    output_file.write(file[:-4])
    output_file.write(" ")
    output_file.write(prediction_to_vad_label(pred))
    output_file.write("\n")

output_file.close()
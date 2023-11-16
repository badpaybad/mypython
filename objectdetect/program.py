import os
import datetime 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)
#pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
# https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth

# https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#wav2vec-20

import torchaudio
print(torch.__version__)
print(torchaudio.__version__)
torch.random.manual_seed(0)

import numpy as np

import utils as Ultils

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


import matplotlib.pyplot as plt
SPEECH_FILE = "hello11.wav"

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

print("Sample Rate:", bundle.sample_rate)

print("Labels:", bundle.get_labels())

model = bundle.get_model().to(device)

state_dict = torch.load("/work/mypython/objectdetect/wav2vec2_fairseq_base_ls960_asr_ls960.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

print("model.__class__")
print(model.__class__)


def get_feature_wav(waveform):

    # waveform, sample_rate = torchaudio.load(file_path_wav)
    # waveform = waveform.to(device)

    # if sample_rate != bundle.sample_rate:
    #     waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        
    with torch.inference_mode():      
        
        features, _ = model.extract_features(waveform)
        # print("features")
        # print(features)
        f=features[0].detach().numpy()
        # print("f")
        # print(f.shape)
        # print(f)
        # print("f=f.flatten()")
        f=f.flatten()
        # print(f)
        
        # r= findCosineDistance(f,f)
        # print(round( r,3))
        return f


def find_subwav_in_longwav(waveform_sub, waveform_long, threshold_score=0.55,shift_duration=0.2):
    
    f0= get_feature_wav(waveform_sub)
    
    chunks= Ultils.cut_and_shift_audio(waveform_sub, waveform_long, shift_duration, bundle.sample_rate)
    
    res=[]
    for i, slice_waveform in enumerate(chunks):
        f1= get_feature_wav(slice_waveform)
        # print(waveform.size())
        # print(slice_waveform.size())
                
        s=  Ultils.findCosineDistance(f0,f1)
        if s!=None and s< threshold_score:
            res.append((i,s,slice_waveform,waveform_sub, waveform_long))
        #print([i, s])
    return res

def test():
    wf0= Ultils.load2waveform("hello11.wav", bundle.sample_rate,True)
    wf1= Ultils.load2waveform("hello2.wav", bundle.sample_rate,True)
    res=find_subwav_in_longwav(wf0, wf1)
    minS= 9999
    minR=None
    for r in res:
        i,s,w,w0,wl=r
        if s< minS:
            minS=s
            minR=r
            
        # Save the cut waveform
        output_path = f"cut_{i}_{s}.wav"
        torchaudio.save(output_path, w, sample_rate=bundle.sample_rate)

        with torch.inference_mode():
            emission, _ = model(w)
            decoder = GreedyCTCDecoder(labels=bundle.get_labels())
            transcript = decoder(emission[0])
            print("transcript")
            print(transcript)

    with torch.inference_mode():
        emission, _ = model(wf1)
        decoder = GreedyCTCDecoder(labels=bundle.get_labels())
        transcript = decoder(emission[0])
        print("transcript wf1")
        print(transcript)    
        # # Save the cut waveform
        # output_path = f"cut_{minR[0]}.wav"
        # torchaudio.save(output_path, minR[2], sample_rate=bundle.sample_rate)
    
for i in range(0,1):
    print(datetime.datetime.now())
    test()
    print(datetime.datetime.now())

# with torch.inference_mode():
#     emission, _ = model(waveform)
    
#     print(emission)
#     decoder = GreedyCTCDecoder(labels=bundle.get_labels())
    
#     transcript = decoder(emission[0])
#     print("transcript")
#     print(transcript)
#     # plt.imshow(emission[0].cpu().T, interpolation="nearest")
#     # plt.title("Classification result")
#     # plt.xlabel("Frame (time-axis)")
#     # plt.ylabel("Class")
#     # plt.tight_layout()
#     # print("Class labels:", bundle.get_labels())
    
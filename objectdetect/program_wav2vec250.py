import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import torch

device = torch.device("cpu")
print(device)
import torchaudio
print(torch.__version__)
print(torchaudio.__version__)
torch.random.manual_seed(0)


import utils as Ultils

#https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h

# load model and tokenizer
processor = Wav2Vec2Processor.from_pretrained("/data/wav2vec2-large-xlsr-vietnamese") 
model = Wav2Vec2ForCTC.from_pretrained("/data/wav2vec2-large-xlsr-vietnamese")


def load2waveform(filepath):        
    waveform, sample_rate = torchaudio.load(filepath)
    waveform = waveform.to(device)
    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            
    if waveform.shape[0] == 2:
        # Convert stereo to mono by averaging channels
        waveform = waveform.mean(dim=0, keepdim=True)
        
    outtofile=filepath .replace(".wav","")+"_16000.wav"
    torchaudio.save(outtofile,waveform, sample_rate=16000)
        
    return outtofile

# Preprocessing the datasets.
# We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
    file16000=batch["path"]
    file16000=load2waveform(file16000)
    speech_array, sampling_rate = torchaudio.load(file16000)
    batch["speech"] = speech_array.squeeze().numpy()
    return batch

# batch={
#     "path":"/work/mypython/objectdetect/hello2.wav"
# }
# speech_file_to_array_fn(batch)

# inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

# with torch.no_grad():
#     feature_extractor= model.wav2vec2.feature_extractor(inputs.input_values)
#     print(feature_extractor)
#     logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

# predicted_ids = torch.argmax(logits, dim=-1)

# print("Prediction:", processor.batch_decode(predicted_ids))

def feature_extractor( file_path):
    batch={
        "path":file_path
    }
    speech_file_to_array_fn(batch)
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)
        
    with torch.no_grad():
        feature_extractor= model.wav2vec2.feature_extractor(inputs.input_values)        
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        
    predicted_ids = torch.argmax(logits, dim=-1)
    transcript = processor.batch_decode(predicted_ids)
    
    return (feature_extractor, transcript)

   
def feature_extractor_waveform16000( waveform16000):
    batch={
        "speech":waveform16000.squeeze().numpy()
    }
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)
        
    with torch.no_grad():
        feature_extractor= model.wav2vec2.feature_extractor(inputs.input_values)
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        
    predicted_ids = torch.argmax(logits, dim=-1)
    transcript = processor.batch_decode(predicted_ids)
    
    return (feature_extractor, transcript)
# print(feature_extractor("/work/mypython/objectdetect/hello2.wav"))

def get_feature_wav(waveform16000):

    f,t= feature_extractor_waveform16000(waveform16000)
    f=f[0].detach().numpy()
    f=f.flatten()
    return f

def find_subwav_in_longwav(waveform_sub, waveform_long, threshold_score=0.3,shift_duration=0.2, SAMPLE_RATE=16000):
    
    f0= get_feature_wav(waveform_sub)
    
    chunks= Ultils.cut_and_shift_audio(waveform_sub, waveform_long, shift_duration, SAMPLE_RATE)
    
    res=[]
    for i, slice_waveform in enumerate(chunks):
        f1= get_feature_wav(slice_waveform)
        # print(waveform.size())
        # print(slice_waveform.size())
                
        s=  Ultils.findCosineDistance(f0,f1)
        if s!=None and s< threshold_score:
            res.append((i,s,slice_waveform,waveform_sub, waveform_long))
        print([i, s])
    return res


def test():
    
    SAMPLERATE=16000
    
    wf0= Ultils.load2waveform("hello11.wav",SAMPLERATE,True)
    wf1= Ultils.load2waveform("hello2.wav",SAMPLERATE,True)
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
        torchaudio.save(output_path, w, sample_rate=SAMPLERATE)

import datetime
    
for i in range(0,1):
    print(datetime.datetime.now())
    test()
    print(datetime.datetime.now())

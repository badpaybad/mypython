import os
import datetime 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)
#pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
#https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth
import torchaudio
print(torch.__version__)
print(torchaudio.__version__)
torch.random.manual_seed(0)

import numpy as np

def findCosineDistance( source_representation, test_representation):
    try:
        if type(source_representation) == list:
            source_representation = np.array(source_representation)

        if type(test_representation) == list:
            test_representation = np.array(test_representation)
        
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
    except Exception as ex:
        print("Error findCosineDistance")
        print(ex)


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

SAMPLE_RATE= bundle.sample_rate

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


def cut_and_shift_audio( input_waveform,  reference_waveform,  shift_duration=0.1):
    # Load audio files
    # input_waveform, input_sample_rate = torchaudio.load(input_audio_path)
    # reference_waveform, reference_sample_rate = torchaudio.load(reference_audio_path)
    input_sample_rate= bundle.sample_rate
    reference_sample_rate=bundle.sample_rate
    # Calculate the duration of a.wav
    duration_a = input_waveform.size(1) / input_sample_rate
    duration_a_ref= reference_waveform.size(1) / reference_sample_rate
    # print("duration_a")
    # print(duration_a)
    # print("reference_waveform duraation")    
    # print(duration_a_ref)

    # Calculate the number of frames in b.wav corresponding to the duration of a.wav
    frames_to_keep = int(duration_a * reference_sample_rate)
   
    # cut_reference_waveform = reference_waveform[:, :frames_to_keep]

    # Calculate the number of frames in each slice with the duration of a.wav
    slice_frames = input_waveform.size(1)

    # Calculate the number of frames to shift with each step
    shift_frames = int(shift_duration * reference_sample_rate)

    # print("frames_to_keep")
    # print(frames_to_keep)
    # print("slice_frames")
    # print(slice_frames)
    # print("shift_frames")
    # print(shift_frames)
    # print("cut_reference_waveform")
    # print(reference_waveform.size(1))
    # Cut and shift the waveform
    slices = []
    counter=0
    for i in range(0, reference_waveform.size(1) - slice_frames + 1, shift_frames):
        slice_end = i + slice_frames
        current_slice = reference_waveform[:, i:slice_end]
        slices.append(current_slice)
                
        # # Save the cut waveform
        # output_path = f"cut_{counter}_{i}.wav"
        # torchaudio.save(output_path, current_slice, sample_rate=reference_sample_rate)
        
        counter=counter+1

        
    """
        
    # Example usage
    input_audio_path = "a.wav"
    reference_audio_path = "b.wav"
    slices = cut_and_shift_audio(input_audio_path, reference_audio_path, shift_duration=0.1)

    # Each element in 'slices' is a waveform with the duration of a.wav
    for i, slice_waveform in enumerate(slices):
        print(f"Slice {i + 1}: {slice_waveform.size(1)} frames")
    """

    return slices


def compare_wav(waveform, waveformbig,shift_duration=0.2):
    
    f0= get_feature_wav(waveform)
    
    chunks= cut_and_shift_audio(waveform, waveformbig, shift_duration)
    
    res=[]
    for i, slice_waveform in enumerate(chunks):
        f1= get_feature_wav(slice_waveform)
        # print(waveform.size())
        # print(slice_waveform.size())
                
        s= findCosineDistance(f0,f1)
        if s< 0.4:
            res.append((i,s,slice_waveform,waveform))
        #print([i, s])
    return res

def load2waveform(filepath):        
    waveform, sample_rate = torchaudio.load(filepath)
    waveform = waveform.to(device)

    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        
    return waveform

def test():
    res=compare_wav(load2waveform("hello11.wav"), load2waveform("hello2.wav"))
    minS= 9999
    minR=None
    for r in res:
        i,s,w,w0=r
        if s< minS:
            minS=s
            minR=r

    print(minR)
    with torch.inference_mode():
        emission, _ = model(minR[2])
        decoder = GreedyCTCDecoder(labels=bundle.get_labels())
        transcript = decoder(emission[0])
        print("transcript")
        print(transcript)
    
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
    
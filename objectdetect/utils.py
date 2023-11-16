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


def findCosineDistance( source_representation, test_representation):
    """min is better

    Args:
        source_representation (_type_): _description_
        test_representation (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        
        if type(source_representation) == list:
            source_representation = np.array(source_representation)

        if type(test_representation) == list:
            test_representation = np.array(test_representation)
        
        #a = np.matmul(np.transpose(source_representation), test_representation)
        a = np.matmul(source_representation, test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return  1- (a / (np.sqrt(b) * np.sqrt(c)))
    except Exception as ex:
        print("Error findCosineDistance")
        print(ex)


def cut_and_shift_audio( input_waveform,  reference_waveform,  shift_duration=0.2, SAMPLERATE=16000):
    # Load audio files
    # input_waveform, input_sample_rate = torchaudio.load(input_audio_path)
    # reference_waveform, reference_sample_rate = torchaudio.load(reference_audio_path)
    input_sample_rate= SAMPLERATE
    reference_sample_rate=SAMPLERATE
    # Calculate the duration of a.wav
    duration_a = input_waveform.size(1) / input_sample_rate
    #duration_a_ref= reference_waveform.size(1) / reference_sample_rate
    # print("duration_a")
    # print(duration_a)
    # print("reference_waveform duraation")    
    # print(duration_a_ref)

    # Calculate the number of frames in b.wav corresponding to the duration of a.wav
    #frames_to_keep = int(duration_a * reference_sample_rate)
   
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


def load2waveform(filepath, SAMPLERATE=16000, to_mono=False):        
    waveform, sample_rate = torchaudio.load(filepath)
    waveform = waveform.to(device)

    if sample_rate != SAMPLERATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLERATE)
        
    if to_mono==True and waveform.shape[0] == 2:
        # Convert stereo to mono by averaging channels
        waveform = waveform.mean(dim=0, keepdim=True)
        
    return waveform

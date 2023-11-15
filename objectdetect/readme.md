# convert 

ffmpeg -y -i hello.m4a hello1.wav


# trim from sec # -ss 60 -to 70

ffmpeg -y -i hello.m4a -ss 00:00:00.500 -to  00:00:02.000  hello11.wav

# model 

https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#wav2vec-20 

download wav2vec_large.pt at https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt
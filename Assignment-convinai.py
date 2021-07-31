#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install datasets')
get_ipython().system('pip install soundfile')
get_ipython().system('pip install datasets')


# In[2]:


get_ipython().system('pip install pydub')


# In[3]:


get_ipython().system('pip install librosa')


# In[4]:


get_ipython().system('pip install os-sys')


# In[5]:


import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Tokenizer
import librosa
from pydub import AudioSegment
import os


# In[6]:


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
librispeech_samples_ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")


# In[7]:


file_list=[]
audio = AudioSegment.from_file("C:/Users/Aniruddh Aithal/Downloads/Bush_Addresses_Congress_9-20-01.wav")
lengthaudio = len(audio)
print("Length of Audio File", lengthaudio)
start = 0
# In Milliseconds, this will cut 10 Sec of audio
threshold = 10000
end = 0
counter = 0
while start < len(audio):
    end += threshold
    #print(start , end)
    chunk = audio[start:end]
    filename = f'C:/Users/Aniruddh Aithal/Downloads/Chunks/{counter}.wav'
    file_list.append(filename)
    chunk.export(filename, format="wav")
    counter +=1
    start += threshold


# In[10]:


# iterate over files in list
for i in file_list:  
#load audio snippets
    audio, rate = librosa.load(i, sr = 16000)
    file1 = open("C:/Users/Aniruddh Aithal/Downloads/Transcription.txt", "a")
# pad input values and return pt tensor
    input_values = processor(audio, sampling_rate=rate, return_tensors="pt").input_values

# retrieve logits & take argmax
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

# transcribe
    transcription = processor.decode(predicted_ids[0])
    print(transcription)
    file1.write(transcription)
    file1.close()

# FINE-TUNE

    target_transcription = "----------------------------"#Audio is too long to have target transcript.

# encode labels
    with processor.as_target_processor():
        labels = processor(target_transcription, return_tensors="pt").input_ids

# compute loss by passing labels
    loss = model(input_values, labels=labels).loss
    loss.backward()


# In[ ]:





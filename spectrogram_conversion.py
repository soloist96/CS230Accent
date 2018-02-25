
# coding: utf-8

# In[1]:


import sys
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa
from librosa import display
import glob


# In[2]:


"""
# Function that parse audio from audio clip
# Input: wav file name, t1 (start of phoeme), t2 (end of phoeme)
# Returns: audio segments of individual phoeme between t1 and t2
"""
def parse_out_segment(audio_clip, t1, t2):

	# Grab audio segment between t1 and t2
	# First grab the first t2 milliseconds
	first_audio_segment = audio_clip[:t2]

	# Then grab the last t2-t1 milliseconds
	phoeme_length = t2 - t1
	audio_segment = first_audio_segment[-phoeme_length:]

	return audio_segment


# In[3]:


"""
# Function that segments audio clip into smaller segments
# Input: wav file name
# Returns: dictionary of audio segments
"""
def segment_audio_clip(audio_file_name):

	# Length is 1000ms = 1sec
	segment_length = 500

	# Read the audio file
	audio_clip = AudioSegment.from_wav(audio_file_name)
	#print audio_clip.duration_seconds
	# Calculate the number of segments based on audio clip duration and segment length
	audio_duration_ms = (audio_clip.duration_seconds)*1000
	num_segments = int(audio_duration_ms / segment_length)
	#print num_segments
	
	# Parse out wav file name
	# Read in label file into dictionary
	split_wav_file = audio_file_name.split('/wav\\')
	#print str(split_wav_file)
	split_wav_file_name = split_wav_file[1].split('.wav')
	wav_file_name = split_wav_file_name[0]

	# Segment the audio clip and save in dictionary
	segment_dict = {}

	for i in range(num_segments):
		key = wav_file_name + '_' + str(i)
		segment_dict[key] = parse_out_segment(audio_clip, i*1000, i*1000+segment_length)

	return segment_dict


# In[4]:


"""
# Function that exports each audio segment to individual WAV files
# Input: dictionary of phoemes audio clips
# Returns: nothing
"""
def export_audio_segments(audio_segments_dict, wav_file_name, accent_id):

	for timestamp in audio_segments_dict:
		export_file_name = accent_id + '_spectrograms\\' + timestamp + '.wav'
		#export_file_name = timestamp + '.wav'

		#Exports to a wav file
		audio_segment = audio_segments_dict[timestamp]
		audio_segment.export(export_file_name, format="wav")


# In[11]:


"""
# Function that creates spectrogram of each WAV file
# Input: none
# Returns: nothing
"""
def convert_audio_to_spectrograms(accent_id):
    reg_ex = accent_id + '_spectrograms\*.wav'
    wav_file_list = glob.glob(reg_ex)
    #print wav_file_list
    
    spectrogram_list = []
    
    for wav_file in wav_file_list:
        #print wav_file
        
        # Convert wav file to spectrogram (FFT)
        samples, sampling_rate = librosa.load(wav_file)
        D = librosa.stft(samples)
        D_magnitude = np.abs(D)
        #print (D_magnitude).shape
        
        D_reshape = np.reshape(D_magnitude,(205,110))
        #print D_reshape.shape
        
        spectrogram_list.append(D_reshape)
        
        D_amp_to_db = librosa.amplitude_to_db(D_magnitude, ref=np.max)
        #plt.pcolormesh(D_amp_to_db)
        #librosa.display.specshow(D_amp_to_db, y_axis='log', x_axis='time')
        
        # Find clip name
        #split_str = accent_id + '_spectrograms\\'
        #split_file_name = wav_file.split(split_str)
        #wav_file_name = split_file_name[len(split_file_name)-1]
        #split_wav_file = wav_file_name.split('.wav')
        #wav_clip_name = split_wav_file[0]
        
        #plot_title_str = 'Power spectrogram of ' + accent_id + ' ' + wav_clip_name
        #plt.title(plot_title_str)
        #plt.colorbar(format='%+2.0f dB')
        #plt.tight_layout()
        #plt.show()
        
    return spectrogram_list


# In[14]:


"""
# MAIN FUNCTION
"""
if __name__ == "__main__":
    
    # Grab all the audio clip files
    #accent_id_list = ['bdl','aew','rms','clb','eey','ljm','lnh','slt','ahw','awb','fem','jmk','rxr','axb','slp','aup','gka','ksp']
    accent_id_list = ['aup']
    for accent_id in accent_id_list:
        cmd_arg = '../dataset/cmu_us_' + accent_id + '_arctic/wav/arctic_*.wav'
        clip_file_list = glob.glob(cmd_arg)

        #print clip_file_list

        #--------------------------------------------------
        # Split audio into segments
        #--------------------------------------------------
        for wav_file in clip_file_list:
            #print wav_file
            audio_segments_dict = segment_audio_clip(wav_file)
            #print audio_segments_dict

            split_wav_file = wav_file.split('/wav\\')
            #print str(split_wav_file)
            split_wav_file_name = split_wav_file[1].split('.wav')
            wav_file_name = split_wav_file_name[0]

            #--------------------------------------------------
            # EXPORT AS WAV FILES
            #--------------------------------------------------
            export_audio_segments(audio_segments_dict, wav_file_name, accent_id)

        #--------------------------------------------------
        # CONVERT TO SPECTROGRAM
        #--------------------------------------------------
        spectrogram_list = convert_audio_to_spectrograms(accent_id)
        spectrogram_array = np.array(spectrogram_list)
        print accent_id + str(spectrogram_array.shape)

        #--------------------------------------------------
        # PRINT OUT SPECTROGRAM TO FILE
        #--------------------------------------------------
        spectrogram_file_name = accent_id + '_spectrogram_array.npy'
        np.save(spectrogram_file_name, spectrogram_array)
    
   


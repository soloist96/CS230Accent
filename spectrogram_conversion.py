
# coding: utf-8

# In[3]:


import sys
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa
from librosa import display
import glob


# ### Function to parse out audio from clip

# In[4]:


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


# ### Function to change volume (magnitude) of audio clip

# In[5]:


"""
# Function tha changes the volume (magnitude) of the audio clip
# Input: pydub audio clip
# Returns: audio increased and decreased
"""
def volume_change_audio_clip(audio_clip):
    
    db_shift = 10
    
    # boost volume by 10dB
    louder_audio = audio_clip + db_shift

    # reduce volume by 10dB
    quieter_audio = audio_clip - db_shift
    
    return louder_audio,quieter_audio


# ### Function to segment the audio clip

# Has flags to shift the audio by 250ms and change volume

# In[19]:


"""
# Function that segments audio clip into smaller segments
# Input: wav file name, flag to shift by 250ms, flag to change volume
# Returns: dictionary of audio segments
"""
def segment_audio_clip(audio_file_name, wav_name, shift_250ms, flag_change_volume):
    
    # Length is 1000ms = 1sec
    segment_length = 500
    shift_length = shift_250ms
    
    # Read the audio file
    audio_clip = AudioSegment.from_wav(audio_file_name)
    #print audio_clip.duration_seconds
    
    # Calculate the number of segments based on audio clip duration and segment length
    audio_duration_ms = (audio_clip.duration_seconds)*1000    
    num_segments = int(audio_duration_ms / segment_length)
    #print num_segments
    
    # Segment the audio clip and save in dictionary
    segment_dict = {}
    
    for i in range(num_segments):
        key = wav_name + '_' + str(i)
        segment_audio = parse_out_segment(audio_clip, i*1000, i*1000+segment_length)
        segment_dict[key] = segment_audio
        
        # Check if volume change is request
        if (flag_change_volume):
            louder_audio_seg, quieter_audio_seg = volume_change_audio_clip(segment_audio)
            segment_dict[key + '_louder'] = louder_audio_seg
            segment_dict[key + '_quieter'] = quieter_audio_seg
    
        
    # Cut out the first 250ms of the audio clip
    if (shift_250ms):        
        new_audio_duration_ms = audio_duration_ms - shift_length
        num_segments = int(new_audio_duration_ms / segment_length)
        
        wav_name = split_wav_file_name[0] + '_shifted'
    
        for i in range(num_segments):
            key = wav_name + '_' + str(i)
            segment_audio = parse_out_segment(audio_clip, i*1000, i*1000+segment_length)
            segment_dict[key] = segment_audio
            
            # Check if volume change is request
            if (flag_change_volume):
                louder_audio_seg, quieter_audio_seg = volume_change_audio_clip(segment_audio)
                segment_dict[key + '_louder'] = louder_audio_seg
                segment_dict[key + '_quieter'] = quieter_audio_seg
    
    return segment_dict


# ### Function to export audio segment to individual WAV files

# In[20]:


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


# ### Function to change the pitch of the audio

# In[21]:


"""
# Function that changes the pitch of given audio
# Input: Librosa audio data
# Returns: audio pitch shifted up and pitch shifted down
"""
def pitch_shift_audio(librosa_samples, sampling_rate):
    # Shift up by a major third (four half-steps)
    samples_third = librosa.effects.pitch_shift(librosa_samples, sampling_rate, n_steps=4)
    
    # Shift down by a tritone (six half-steps)
    samples_tritone = librosa.effects.pitch_shift(librosa_samples, sampling_rate, n_steps=-6)
    
    return samples_third, samples_tritone


# ### Function to create spectrograms of each WAV file

# In[22]:


"""
# Function that creates spectrogram of each WAV file
# Input: none
# Returns: nothing
"""
def convert_audio_to_spectrograms(accent_id, flag_pitch_shift):
    reg_ex = accent_id + '_spectrograms\*.wav'
    wav_file_list = glob.glob(reg_ex)
    #print wav_file_list
    
    spectrogram_list = []
    
    for wav_file in wav_file_list:
        #print wav_file
        
        # Convert wav file to spectrogram (FFT)
        samples, sampling_rate = librosa.load(wav_file)
        
        # Comput STFT of the audio
        D = librosa.stft(samples)
            
        D_magnitude = np.abs(D)
        #print (D_magnitude).shape
        
        D_reshape = np.reshape(D_magnitude,(205,110))
        #print D_reshape.shape
        
        # Append to spectrogram list
        spectrogram_list.append(D_reshape)
        
        # Check if pitch shifting is requested
        if (flag_pitch_shift):
            pitched_up_samples, pitched_down_samples = pitch_shift_audio(samples, sampling_rate)
            
            # Compute STFT
            D_pitched_up = librosa.stft(pitched_up_samples)
            D_pitched_down = librosa.stft(pitched_down_samples)
            
            # Reshape
            D_pitched_up_reshape = np.reshape(np.abs(D_pitched_up),(205,110))
            D_pitched_down_reshape = np.reshape(np.abs(D_pitched_down),(205,110))
            
            # Append to spectrogram list
            spectrogram_list.append(D_pitched_up_reshape)
            spectrogram_list.append(D_pitched_down_reshape)
        
        # Plot the spectrom
        D_amp_to_db = librosa.amplitude_to_db(D_magnitude, ref=np.max)
        plt.pcolormesh(D_amp_to_db)
        librosa.display.specshow(D_amp_to_db, y_axis='log', x_axis='time')
        """
        # Find clip name
        split_str = accent_id + '_spectrograms\\'
        split_file_name = wav_file.split(split_str)
        wav_file_name = split_file_name[len(split_file_name)-1]
        split_wav_file = wav_file_name.split('.wav')
        wav_clip_name = split_wav_file[0]
        
        plot_title_str = 'Power spectrogram of ' + accent_id + ' ' + wav_clip_name
        plt.title(plot_title_str)
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()
        """
        
        # Plot spectrogram of pitch shifted audio
        if (flag_pitch_shift):
            D_amp_to_db = librosa.amplitude_to_db(np.abs(D_pitched_up), ref=np.max)
            plt.pcolormesh(D_amp_to_db)
            librosa.display.specshow(D_amp_to_db, y_axis='log', x_axis='time')
            """
            plot_title_str = 'Power spectrogram of ' + accent_id + ' ' + wav_clip_name + ' pitched UP'
            plt.title(plot_title_str)
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.show()
            
            D_amp_to_db = librosa.amplitude_to_db(np.abs(D_pitched_down), ref=np.max)
            plt.pcolormesh(D_amp_to_db)
            librosa.display.specshow(D_amp_to_db, y_axis='log', x_axis='time')
                    
            plot_title_str = 'Power spectrogram of ' + accent_id + ' ' + wav_clip_name + ' pitched DOWN'
            plt.title(plot_title_str)
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.show()
            """
    return spectrogram_list


# # Running the data augmentation

# In[28]:


"""
# MAIN FUNCTION
"""
if __name__ == "__main__":
    
    # Grab all the audio clip files
    accent_id_list = ['bdl','aew','rms','clb','eey','ljm','lnh','slt','ahw','awb','fem','jmk','rxr','axb','slp','aup','gka','ksp']
    #accent_id_list = ['bdl']
    for accent_id in accent_id_list:
        cmd_arg = '../dataset/cmu_us_' + accent_id + '_arctic/wav/arctic_*.wav'
        clip_file_list = glob.glob(cmd_arg)

       # print clip_file_list

        #--------------------------------------------------
        # Split audio into segments
        #--------------------------------------------------
        for wav_file in clip_file_list:
            #print wav_file
            
            #--------------------------------------------------
            # WAV NAME & NUMBER
            #--------------------------------------------------
            split_wav_file = wav_file.split('/wav\\')
            #print str(split_wav_file)
            split_wav_file_name = split_wav_file[1].split('.wav')
            wav_name = split_wav_file_name[0]
            
            #--------------------------------------------------
            # SEGMENT WAV FILE
            #--------------------------------------------------
            audio_segments_dict = segment_audio_clip(wav_file, wav_name, shift_250ms=True, flag_change_volume=True)
            #print audio_segments_dict



            #--------------------------------------------------
            # EXPORT AS WAV FILES
            #--------------------------------------------------
            export_audio_segments(audio_segments_dict, wav_name, accent_id)

        #--------------------------------------------------
        # CONVERT TO SPECTROGRAM
        #--------------------------------------------------
        spectrogram_list = convert_audio_to_spectrograms(accent_id, flag_pitch_shift=True)
        spectrogram_array = np.array(spectrogram_list)
        print accent_id + str(spectrogram_array.shape)

        #--------------------------------------------------
        # PRINT OUT SPECTROGRAM TO FILE
        #--------------------------------------------------
        spectrogram_file_name = accent_id + '_spectrogram_array.npy'
        np.save(spectrogram_file_name, spectrogram_array)
    
   


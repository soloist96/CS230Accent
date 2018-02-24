import sys
import numpy as np
import math
from pydub import AudioSegment
import matplotlib.pyplot as plt
import glob

"""
# Function that reads in the label file
# Input: label file name
# Returns: dictionary of phoemes ('time': 'phoeme')
"""
def read_label_into_phoeme_dict(file_name):

	phoeme_dict = {}

	f_label_file = open(file_name, "r")

	for line in iter(f_label_file):
		#print line
		if line != "#\n":
			split_line = line.split()
			phoeme_dict[split_line[0]] = split_line[2]

	return phoeme_dict


"""
# Function that parse audio of individual phoeme from audio clip
# Input: wav file name, t1 (start of phoeme), t2 (end of phoeme)
# Returns: audio segments of individual phoeme between t1 and t2
"""
def parse_out_phoeme(audio_clip, t1, t2):

	# Grab audio segment between t1 and t2
	# First grab the first t2 milliseconds
	first_audio_segment = audio_clip[:t2]

	# Then grab the last t2-t1 milliseconds
	phoeme_length = t2 - t1
	audio_segment = first_audio_segment[-phoeme_length:]

	return audio_segment


"""
# Function that segments entire audio clip into indidivual phoemes
# Input: wav file name, dictionary of phoemes with timestamps
# Returns: dictionary of audio segments of phoemes ('time': 'phoeme audio segment')
"""
def segment_audio_into_phoemes(audio_file_name, label_phoemes_dict):

	# Open the WAV file
	audio_clip = AudioSegment.from_wav(audio_file_name)

	print "Audio clip duration = " + str(audio_clip.duration_seconds)

	# Sort the timestamps
	timestamp_keys = label_phoemes_dict.keys()
	timestamp_keys.sort()

	# Initialize variables
	segment_dict = {}
	prev_phoeme = ""
	t1 = 0.0
	t2 = 0.0
	#print str(label_phoemes_dict)
	#print str(timestamp_keys)

	# Loop over the phoemes in the dictionary
	for timestamp in timestamp_keys:

		curr_phoeme = label_phoemes_dict[timestamp]

		# pydub works in milliseconds
		t1 = t2
		t2 = float(timestamp)*1000.0

		# Skip if phoeme is a pause ("pau")
		if curr_phoeme != "pau" and prev_phoeme != "pau":
			#print "PHOEME = " + prev_phoeme
			#print "T1 = " + str(t1)
			#print "T2 = " + str(t2)

			# Parse out audio segment between t1 and t2, and save phoeme audio clip to dictionary
			segment_dict[str(t1)] = parse_out_phoeme(audio_clip, t1, t2)

		prev_phoeme = curr_phoeme

	return segment_dict


"""
# Function that exports each phoeme audio clip to indidivual WAV files
# Input: dictionary of phoemes audio clips
# Returns: nothing
"""
def export_audio_segments(audio_segments_dict, wav_file_name):

	for timestamp in audio_segments_dict:
		export_file_name = wav_file_name + '_' + timestamp + '.wav'

		#Exports to a wav file
		audio_segment = audio_segments_dict[timestamp]
		audio_segment.export(export_file_name, format="wav")


"""
# Function that creates spectrogram of each phoeme WAV file
# Input: none
# Returns: nothing
"""
def	convert_phoemes_to_spectrograms(wav_file_name):

	reg_ex = wav_file_name + '*.wav'
	wav_file_list = glob.glob(reg_ex)

	for wav_file in wav_file_list:
		#print wav_file

		# Convert wav file to spectrogram (FFT)
		sample_rate, samples = wavfile.read(wav_file)
		frequencies, times, spectogram = signal.spectrogram(samples, sample_rate)

		# Create spectrogram plot
		plt.pcolormesh(times, frequencies, spectogram)
		fig = plt.imshow(spectogram)
		plt.axis('off')
		#plt.show()

		# Create file name and save spectrogram plot
		split_wav_file_name = wav_file.split('wav')
		plot_name = './spectrograms/' + split_wav_file_name[0] + 'png'
		#print plot_name
		fig.axes.get_xaxis().set_visible(False)
		fig.axes.get_yaxis().set_visible(False)
		plt.savefig(plot_name, bbox_inches='tight', transparent=True, pad_inches=0)


"""
# MAIN FUNCTION
"""
if __name__ == "__main__":

	os.system('mkdir spectrograms')

	# Grab all the audio clip files
	#clip_file_list = glob.glob('../../cmu_arctic/cmu_us_bdl_arctic/wav/*.wav')
	clip_file_list = glob.glob('../../cmu_arctic/cmu_us_ksp_arctic/wav/*.wav')

	for wav_clip in clip_file_list:
		print wav_clip
		#--------------------------------------------------
		# READ IN DATA
		#--------------------------------------------------
		# Read in label file into dictionary
		split_wav_file = wav_clip.split('/wav/')
		#print str(split_wav_file)
		split_wav_file_name = split_wav_file[1].split('.wav')
		wav_file_name = split_wav_file_name[0]
		#print str(wav_file_name)
		label_file_name = split_wav_file[0] + '/lab/' + wav_file_name + '.lab'
		label_phoemes_dict = read_label_into_phoeme_dict(label_file_name)
		#print str(phoeme_dict)

		#--------------------------------------------------
		# SPLIT INTO PHOEMES
		#--------------------------------------------------
		audio_segments_dict = segment_audio_into_phoemes(wav_clip, label_phoemes_dict)
	
		#--------------------------------------------------
		# EXPORT AS WAV FILES
		#--------------------------------------------------
		export_audio_segments(audio_segments_dict, wav_file_name)

		#--------------------------------------------------
		# CONVERT TO SPECTROGRAM
		#--------------------------------------------------
		convert_phoemes_to_spectrograms(wav_file_name)

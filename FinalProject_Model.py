from pydub import AudioSegment

def wavConversion(file_name):
  if file_name.path.endswith('.mp3') or file_name.path.endswith('flac'):
    file = AudioSegment.from_file(file_name)
    new_file_name = file_name.split('.')[0] + ".wav"
    file.export(new_file_name, format='wav')


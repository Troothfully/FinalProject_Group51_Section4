import re
from pydub import AudioSegment

class FileInput:
    
    def __init__(self, file_name):
        self.file_name = file_name

    @property
    def file(self):
        return self.file_name
    
    @file.setter
    def file(self, file_name):
        pattern = r'\b[A-Za-z0-9._%+-]+.wav\b'
        if re.fullmatch(pattern, file_name):
            self.file_name = file_name
        elif file_name.path.endswith('.mp3') or file_name.path.endswith('.flac'):
            file = AudioSegment.from_file(file_name)
            new_file_name = file_name.split('.')[0] + ".wav"
            file.export(new_file_name, format='wav')
            self.file_name = new_file_name
        else:
            raise ValueError(f'Invalid file type (must be .mp3, .wav, or .flac): {file_name}')

    def save(self):
        with open('file_name.txt', 'a') as f:
            f.write(self.file_name + '\n')

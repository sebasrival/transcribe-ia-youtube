import pytube
import whisper
import os
from whisper.utils import get_writer

#Config Inicial
filename = 'audio.mp4'
output_directory = './'

# Config Youtube
youtubeVideoURL = ''
videoTube = pytube.YouTube(youtubeVideoURL)
audio = videoTube.streams.get_audio_only()

# Se Descarga en formato audio
audio.download(filename=filename)

#Configuración del modelo
model = whisper.load_model('small')

#Transcription
result = model.transcribe(filename, fp16=False, language='es')

#Se guarda en el archivo
txt_write = get_writer("txt", output_directory)
txt_write(result, filename, {})

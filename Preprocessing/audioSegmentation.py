from pydub import AudioSegment
import contextlib
import wave

SEGMENT_LENGTH = 5000
AUDIO_FILE = 'CoffeeMachine.wav'
audio = AudioSegment.from_file(AUDIO_FILE)
fileLength = 0
fileName = 'CoffeeMachine'

with contextlib.closing(wave.open(AUDIO_FILE, 'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    fileLength = (frames / float(rate)) * 1000
    print(fileLength)


t1 = 0
counter = 0
while t1 <= fileLength - SEGMENT_LENGTH:
    counter += 1
    t2 = t1 + SEGMENT_LENGTH
    newAudio = audio[t1:t2]
    newName = fileName + str(counter) + '.wav'
    newAudio.export(newName, format="wav")
    t1 += SEGMENT_LENGTH
    print(t1)
    print(fileLength)
    print(SEGMENT_LENGTH)
    print(fileLength - SEGMENT_LENGTH)

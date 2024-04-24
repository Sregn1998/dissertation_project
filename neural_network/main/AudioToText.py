import os
import speech_recognition as sr

audio_save_path = "sample_video/speech1.wav"

class AudioToText:
    def __init__(self, audio_path):
        self.audio_path = audio_path
    
    def recognize_audio(self):
        recognizer = sr.Recognizer()
        with sr.AudioFile(self.audio_path) as source:
            recorded_audio = recognizer.listen(source)
            print("Audio recording done")
        try:
            print("Recognizing the text")
            text = recognizer.recognize_google(recorded_audio, language="en-US")
            print("Decoded Text : {}".format(text))
            if text is not None:
                with open(text_save_path, "w") as text_file:
                    text_file.write(text)
                print("Результат распознавания сохранен в файл:", text_save_path)
            else:
                print("Невозможно сохранить результат распознавания, так как текст не был распознан.")
        except Exception as ex:
            print(ex)
            return None

text_save_path = os.path.join(os.path.dirname(audio_save_path), "result.txt")

audio_to_text_converter = AudioToText(audio_save_path)
text = audio_to_text_converter.recognize_audio()

from moviepy.editor import VideoFileClip

class VideoToAudio:
    def __init__(self, video_path, save_path):
        self.video_path = video_path
        self.save_path = save_path
    
    def extract_audio(self):
        video = VideoFileClip(self.video_path)
        audio = video.audio
        audio.write_audiofile(self.save_path)
        return audio

from faster_whisper import WhisperModel


class ASRHandler:
    def __init__(self):
        self.asr_model = WhisperModel("small" , device="cuda", compute_type="float16")
    
    def transcribe(self, audio_path: str):
        segments, info = self.asr_model.transcribe(audio_path, beam_size=1, vad_filter=True)
        transcribed_text = "".join(segment.text for segment in segments)
        return transcribed_text.strip()


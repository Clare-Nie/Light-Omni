from typing import Optional

import torch
import numpy as np


class VADHandler:
    def __init__(
        self,
        sampling_rate: int = 16000,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 1000, # ms
        speech_pad_ms: int = 100
    ):
        self.sampling_rate = sampling_rate
        self.threshold = threshold

        print("📥 [VAD] Loading Silero VAD model...")
        self.model, utils = torch.hub.load(
            repo_or_dir='./scripts/tools/snakers4_silero-vad_master',
            model='silero_vad',
            source='local',
            force_reload=False,
            onnx=False,
            skip_validation=True
        )
        (_, _, _, VADIterator, _) = utils
        
        self.vad_iterator = VADIterator(
            self.model,
            threshold=threshold,
            sampling_rate=sampling_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms
        )
        self.current_time_offset = 0.0
        self._is_speaking = False
        self._start_sec = None

    def reset(self):
        self.vad_iterator.reset_states()
        self.current_time_offset = 0.0
        self._is_speaking = False
        self._start_sec = None
        print("   🔄 [VAD] State Reset")

    def is_speaking(self):
        return self._is_speaking

    def push_audio(self, audio_chunk: np.ndarray) -> bool:
        if not torch.is_tensor(audio_chunk):
            audio_tensor = torch.from_numpy(audio_chunk)
        else:
            audio_tensor = audio_chunk
        
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.squeeze()

        window_size = 512
        for i in range(0, len(audio_tensor), window_size):
            chunk = audio_tensor[i: i + window_size]
            if len(chunk) < window_size:
                chunk = torch.nn.functional.pad(chunk, (0, window_size - len(chunk)))
            
            speech_event = self.vad_iterator(chunk, return_seconds=True)
            
            if speech_event:                
                if 'start' in speech_event:
                    self._is_speaking = True
                elif 'end' in speech_event:
                    self._is_speaking = False
        
        return self._is_speaking


class AudioSlice:
    def __init__(self, start: float, end: Optional[float]):
        self.start = start
        self.end = end
    
    def duration(self):
        if self.end is None: return 0.0
        return self.end - self.start
    
    def __repr__(self):
        end_str = f"{self.end:.2f}" if self.end else "None"
        return f"<Slice start={self.start:.2f} end={end_str}>"


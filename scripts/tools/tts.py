import subprocess
import random, os, sys

import torchaudio

sys.path.append('CosyVoice')
sys.path.append('CosyVoice/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel

# 提示: 运行此代码前，请先安装 CosyVoice 项目: https://github.com/FunAudioLLM/CosyVoice
# 注意: 必须保证 transformers 的版本为 4.51.3 (可运行 pip install transformers==4.51.3)


class VoiceGenerator:
    def __init__(self):      
        self.cosyvoice = AutoModel(model_dir='CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B')
        self.prompt_speech_path = 'CosyVoice/asset/zero_shot_prompt.wav'
        self.prompt_text_content = 'You are a helpful assistant. <|endofprompt|>' 
        self.has_set_voice = False

    def set_anthor_voice(self, voice_path):
        self.prompt_speech_path = voice_path
        self.has_set_voice = True

    def generate(self, text, output_path):
        results = self.cosyvoice.inference_instruct2(
            text, 
            self.prompt_text_content, 
            self.prompt_speech_path, 
            stream=False
        )
        generated = False
        for i, j in enumerate(results):
            torchaudio.save(output_path, j['tts_speech'], self.cosyvoice.sample_rate)
            generated = True
            break             
        if not generated:
            print("Error: No audio generated.")

    def _get_duration(self, file_path):
        try:
            cmd = [
                'ffprobe', 
                '-v', 'error', 
                '-show_entries', 'format=duration', 
                '-of', 'default=noprint_wrappers=1:nokey=1', 
                file_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return float(result.stdout.strip())
        except Exception as e:
            print(f"Error getting duration for {file_path}: {e}")
            return 0.0
        
    def synthesize(self, ori_path, input_path, output_path):
        # Merge the input audio (input_path) into the original audio (ori_path) and save it to the specified path (output_path)
        
        if not os.path.exists(ori_path) or not os.path.exists(input_path):
            print("Error: Input files not found.")
            return

        abs_ori = os.path.abspath(ori_path)
        abs_inp = os.path.abspath(input_path)
        abs_out = os.path.abspath(output_path)
        
        target_output = output_path
        if abs_out == abs_ori or abs_out == abs_inp:
            target_output = output_path + ".tmp.wav"

        dur_ori = self._get_duration(ori_path)
        dur_inp = self._get_duration(input_path)
        
        if dur_inp >= dur_ori:
            cmd = f'ffmpeg -y -i "{input_path}" -c copy "{target_output}" -v error'
        else:
            split_point = dur_ori - dur_inp
            mode = random.randint(0, 1)
            
            if mode == 0:
                filter_cmd = f'[0:a]atrim=0:{split_point},asetpts=PTS-STARTPTS[head];[head][1:a]concat=n=2:v=0:a=1[out]'
                print("[FFmpeg] Mode: Concatenate (Original Head + Input)")
            else:
                delay_ms = int(split_point * 1000)
                filter_cmd = f'[1:a]adelay={delay_ms}|{delay_ms}[delayed];[0:a][delayed]amix=inputs=2:duration=first:dropout_transition=0[out]'
                print("[FFmpeg] Mode: Mix Overlay (Original + Input Overlay at end)")
            
            cmd = (f'ffmpeg -y -i "{ori_path}" -i "{input_path}" '
                f'-filter_complex "{filter_cmd}" -map "[out]" '
                f'-c:a pcm_s16le "{target_output}" -v error')

        try:
            subprocess.run(cmd, shell=True, check=True)
            if target_output != output_path:
                os.replace(target_output, output_path)
            print(f"[FFmpeg] Successfully saved to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"[FFmpeg] Error executing command: {e}")
            if target_output != output_path and os.path.exists(target_output):
                os.remove(target_output)

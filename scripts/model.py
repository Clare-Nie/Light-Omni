import copy
import re
import time
import os

import soundfile as sf, soxr
import cv2
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from google import genai
from google.genai import types


class QwenOmni2_5Model():
    def __init__(self,
                model_path = "/data1/niechang/Memory/omnimemory/nie_omni/TrainingDatasetConstruction/ms-swift/0313_sft_retrieve/v1-20260313-143837/checkpoint-5793",
                base_model="Qwen/Qwen2.5-Omni-7B",
                device_map=None):
        if device_map is None:
            device_map = "balanced_low_0" 
        print("Loading model from:", model_path)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device_map,
                                                                         attn_implementation="flash_attention_2", trust_remote_code=True)
        self.model.disable_talker()        
        self.processor = Qwen2_5OmniProcessor.from_pretrained(base_model)

        self.media_files_map = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def process_mm_info(self, conversation, sr=16000):
        audios, images = [], []
        for item in conversation[-1]["content"]:
            if item["type"] == "image":
                if item["image"] in self.media_files_map:
                    images.append(copy.deepcopy(self.media_files_map[item["image"]]))
                    continue
                images.append(Image.open(item["image"]).convert("RGB"))
                self.media_files_map[item["image"]] = images[-1]
            elif item["type"] == "audio":
                if item["audio"] in self.media_files_map:
                    audios.append(copy.deepcopy(self.media_files_map[item["audio"]]))
                    continue
                y, fs = sf.read(item["audio"], dtype='float32')
                if y.ndim > 1: y = y.mean(axis=1)
                if fs != sr: y = soxr.resample(y, fs, sr)
                audios.append(y)
                self.media_files_map[item["audio"]] = audios[-1]
        return (audios if audios else None, 
                images if images else None, 
                None)

    def clear_media_files(self):
        self.media_files_map = {}

    def send_message(self, message, mode="response"):
        if mode == "memory" or mode == "retrieve":
            system_prompt = """You are a helpful assistant."""
        else:
            system_prompt = """You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."""

        conversation = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_prompt}
                    ],
                }
            ]
        conversation.append(message)
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = self.process_mm_info(conversation)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=False, use_audio_in_video=False)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        input_len = inputs["input_ids"].shape[1]    
        inputs["meta"] =  {"type": mode, "info": []}
        text_ids = self.model.generate(**inputs, use_audio_in_video=False, use_cache=True, return_audio=False, max_new_tokens=1024)
        text = self.processor.batch_decode(text_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return text


    def get_response_state_parallel(self, conv, raw_message, asr_model):
        if raw_message["audio_path"]:
            future_text = self.executor.submit(asr_model.transcribe, raw_message["audio_path"])
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            conv
        ]
        text_template = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = self.process_mm_info(conversation)
        inputs = self.processor(
            text=text_template, audio=audios, images=images, videos=videos, 
            return_tensors="pt", padding=True, use_audio_in_video=False
        )

        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                inputs[k] = v.to(self.model.dtype)
        inputs["meta"] = {"type": "retrieve", "info": {}, "inference_mode": True}

        with torch.no_grad():
            results = self.model.generate(
                **inputs, 
                use_audio_in_video=False, 
                use_cache=True, 
                return_audio=False, 
                max_new_tokens=1
            )
        
        asr_text = future_text.result() if raw_message["audio_path"] else ""
        text_input = raw_message.get("text", "null")
        final_text = f"{asr_text}\n{text_input}".replace("null", "").strip()

        soft_prompt = results['soft_prompt']
        final_embedding = self.model.thinker.get_emb_input_embedding_from_text(
            final_text, 
            is_query=True, 
            soft_prompt=soft_prompt
        )
        return {
            "is_response": results.get("is_response", False),
            "is_retrieve": results.get("is_retrieve", False),
            "retrieve_embedding": final_embedding.detach().cpu().float().numpy()
        }

    def get_texts_embedding(self, texts):
        return self.model.thinker.get_emb_input_embedding_from_text(texts)
    
    def start_feature_cache(self):
        self.model.thinker.start_feature_cache()
    
    def clear_feature_cache(self):
        self.model.thinker.clear_feature_cache()

    def prepare_input(self, prompt, files=[]):
        assert len(files) == prompt.count("<img>") + prompt.count("<audio>") + prompt.count("<video>")
        inputs = []
        segments = re.split(r'(<img>|<audio>|<video>)', prompt)
        file_index = 0
        for segment in segments:
            if segment == "<img>":
                if file_index < len(files):
                    inputs.append({"type": "image", "image": files[file_index]})
                    file_index += 1
                else:
                    print(f"警告: 发现 <img> 标签，但文件列表已空")
            elif segment == "<audio>":
                if file_index < len(files):
                    inputs.append({"type": "audio", "audio": files[file_index]})
                    file_index += 1
                else:
                    print(f"警告: 发现 <audio> 标签，但文件列表已空")
            elif segment == "<video>":
                if file_index < len(files):
                    inputs.append({"type": "video", "video": files[file_index]})
                    file_index += 1
                else:
                    print(f"警告: 发现 <video> 标签，但文件列表已空")
            else:
                if segment: 
                    inputs.append({"type": "text", "text": segment})
        return {
            "role": "user",
            "content": inputs
        }


class GeminiClient:
    def __init__(self,
                api_key="",
                base_url="http://automl.aiserverai.online",
                model_name="gemini-3-flash-preview"
                ):
        self.client = genai.Client(
            http_options=types.HttpOptions(
            base_url=base_url),
            api_key=api_key)
        self.api_key = api_key
        self.model_name = model_name
        self.chat = self.client.chats.create(model=model_name)

    def clear(self):
        self.chat = self.client.chats.create(model=self.model_name)

    def send_message(self, message, clear=True):
        for _ in range(3):
            try:
                response = self.chat.send_message(message)
                break
            except Exception as e:
                print("gemini api error: ", e)
                time.sleep(2)
                continue
        if clear:
            self.chat = self.client.chats.create(model=self.model_name)
        return response.text

    def image_process(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图片: {img_path}")
        h, w = img.shape[:2]
        if max(h, w) > 720:
            scale = 720 / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
        _, buffer = cv2.imencode('.png', img)
        return buffer.tobytes()
    
    def audio_process(self, audio_path):
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        return audio_bytes
    
    def video_process(self, video_path):
        if not os.path.exists(video_path):
            raise ValueError(f"无法找到视频文件: {video_path}")
        # 获取文件大小并检查（可选）
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        if file_size_mb > 20:
            print(f"警告: 视频文件较大 ({file_size_mb:.2f}MB)，通过 Part.from_bytes 发送可能导致请求超时或失败。")
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
        return video_bytes
    
    def _optimize_continuous_images(self, prompt, files):
        segments = re.split(r'(<img>|<audio>|<video>)', prompt)
        media_info = []
        f_idx = 0
        for i, seg in enumerate(segments):
            if seg in ("<img>", "<audio>", "<video>"):
                media_info.append({"seg_idx": i, "tag": seg, "file_idx": f_idx})
                f_idx += 1

        if not media_info:
            return prompt, files

        to_keep_file_indices = []
        to_remove_seg_indices = set()

        idx = 0
        while idx < len(media_info):
            j = idx
            while (j + 1 < len(media_info) and 
                   media_info[j]['tag'] == '<img>' and 
                   media_info[j+1]['tag'] == '<img>' and 
                   not "".join(segments[media_info[j]['seg_idx']+1 : media_info[j+1]['seg_idx']]).strip()):
                j += 1
            
            curr_block = media_info[idx : j+1]
            if curr_block[0]['tag'] == '<img>' and len(curr_block) >= 9:
                sampled_block = curr_block[::2]
                kept_seg_idxs = {m['seg_idx'] for m in sampled_block}

                to_keep_file_indices.extend([m['file_idx'] for m in sampled_block])
                for m in curr_block:
                    if m['seg_idx'] not in kept_seg_idxs:
                        to_remove_seg_indices.add(m['seg_idx'])
            else:
                to_keep_file_indices.extend([m['file_idx'] for m in curr_block])
            
            idx = j + 1

        new_prompt = "".join([s for i, s in enumerate(segments) if i not in to_remove_seg_indices])
        new_files = [files[i] for i in to_keep_file_indices]
        return new_prompt, new_files

    def prepare_input(self, prompt, files=[], optimize=True):
        if optimize and len(files) > 16:
            prompt, files = self._optimize_continuous_images(prompt, files)

        assert len(files) == prompt.count("<img>") + prompt.count("<audio>") + prompt.count("<video>")
        inputs = []
        segments = re.split(r'(<img>|<audio>|<video>)', prompt)
        file_index = 0
        for segment in segments:
            if segment == "<img>":
                if file_index < len(files):
                    inputs.append(genai.types.Part.from_bytes(
                        data=self.image_process(files[file_index]), 
                        mime_type="image/png"
                    ))
                    file_index += 1
                else:
                    print(f"警告: 发现 <img> 标签，但文件列表已空")
            elif segment == "<audio>":
                if file_index < len(files):
                    inputs.append(genai.types.Part.from_bytes(
                        data=self.audio_process(files[file_index]), 
                        mime_type="audio/wav"
                    ))
                    file_index += 1
                else:
                    print(f"警告: 发现 <audio> 标签，但文件列表已空")
            elif segment == "<video>":
                if file_index < len(files):
                    inputs.append(genai.types.Part.from_bytes(
                        data=self.video_process(files[file_index]), 
                        mime_type="video/mp4"
                    ))
                    file_index += 1
                else:
                    print(f"警告: 发现 <video> 标签，但文件列表已空")
            else:
                if segment: 
                    inputs.append(segment)
        return inputs
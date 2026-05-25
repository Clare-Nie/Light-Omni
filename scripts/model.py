import copy
import re
import time
import os

import soundfile as sf, soxr
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


class QwenOmni2_5Model():
    def __init__(self,
                model_path = "sft_retrieve/v1-20260313-143837/checkpoint-5793",
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


import librosa
import torch, os, re
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info 

class GemmaClient:
    def __init__(self, 
                 model_id="google/gemma-4-E4B-it", 
                 device="auto",
                 dtype=torch.bfloat16):
        print(f"Loading Multimodal Model: {model_id}...")
        from transformers import AutoProcessor, AutoModelForMultimodalLM
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True
        )
        self.device = self.model.device
        self.history = []
        self.system_prompt = "You are a helpful assistant."

    def clear(self):
        self.history = []

    def image_process(self, img_path):
        return img_path
    
    def audio_process(self, audio_path):
        return audio_path


    def prepare_input(self, prompt, files=[]):
        content = []
        segments = re.split(r'(<img>|<audio>)', prompt)
        file_idx = 0
        
        for seg in segments:
            if seg == "<img>":
                if file_idx < len(files):
                    content.append({"type": "image", "image": self.image_process(files[file_idx])})
                    file_idx += 1
            elif seg == "<audio>":
                if file_idx < len(files):
                    content.append({"type": "audio", "audio": self.audio_process(files[file_idx])})
                    file_idx += 1
            elif seg.strip():
                content.append({"type": "text", "text": seg})
        
        return content

    def send_message(self, current_content, clear=True, enable_thinking=False):
        if clear:
            self.clear()
        self.history.append({"role": "user", "content": current_content})
        messages = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}] + self.history
        print("Prepared messages for model:", messages)
        inputs = self.processor.apply_chat_template(
            messages, 
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.device)
        input_len = inputs["input_ids"].shape[-1]
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        self.processor.parse_response(response)
        # print("Raw response:", response.strip('<turn|>'))
        return response


class Qwen3_Omni_Client:
    def __init__(self, 
                 model_path="/data1/niechang/huggingface_cache/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695/",
                 use_audio_in_video=True):
        print(f"[Qwen3-Omni] Loading from {model_path} with device_map='auto'...")
        
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        
        self.use_audio_in_video = use_audio_in_video
        self.history = []
        self.system_prompt = "You are a helpful assistant."

    def clear(self):
        self.history = []

    def prepare_input(self, prompt, files=[]):
        """
        与 GemmaClient 保持一致，将文本标签和文件列表转换为 Qwen3-Omni 的 content 格式
        支持 <img>, <audio>, <video>
        """
        content = []
        # 增加对 <video> 标签的支持
        segments = re.split(r'(<img>|<audio>|<video>)', prompt)
        file_idx = 0
        
        for seg in segments:
            if seg == "<img>":
                if file_idx < len(files):
                    content.append({"type": "image", "image": files[file_idx]})
                    file_idx += 1
            elif seg == "<audio>":
                if file_idx < len(files):
                    content.append({"type": "audio", "audio": files[file_idx]})
                    file_idx += 1
            elif seg == "<video>":
                if file_idx < len(files):
                    content.append({"type": "video", "video": files[file_idx]})
                    file_idx += 1
            elif seg.strip():
                content.append({"type": "text", "text": seg})
        
        return content

    def send_message(self, prompt_or_content, clear=False, max_new_tokens=512):
        if clear:
            self.clear()
        current_content = prompt_or_content

        self.history.append({"role": "user", "content": current_content})
        # messages = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}] + self.history

        messages = self.history
        audios, images, videos = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(
            text=text_prompt, 
            audio=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt", 
            padding=True, 
            use_audio_in_video=self.use_audio_in_video
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                speaker="Ethan", # Qwen3 特有参数
                thinker_return_dict_in_generate=True,
                use_audio_in_video=self.use_audio_in_video,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_audio=False
            )
        if isinstance(output_ids, tuple):
            output_obj = output_ids[0]
        else:
            output_obj = output_ids
        
        # 使用 output_obj 而不是 output_ids
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_obj.sequences[:, input_len:]

        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
            return_audio=False
        )[0]
        self.history = []
        
        return response





# if __name__ == "__main__":
#     client = GemmaClient(model_id="google/gemma-4-E4B-it")
#     my_prompt = "<audio>描述这段音频的内容一句话分析。"
#     my_files = [
#         # "/data1/niechang/Memory/omnimemory/nie_omni/TrainingDatasetConstruction/dataset/training_retrieve/0/profiles/data/processed_2025-01-10 17:07:01.png", 
#         "/data1/niechang/Memory/omnimemory/nie_omni/TrainingDatasetConstruction/dataset/training_retrieve/0/raw/2025-01-10 17:07:01—2025-01-10 17:07:20.wav"
#     ]
#     result = client.send_message(client.prepare_input(my_prompt, my_files))





from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()
# client = GemmaClient(model_id="google/gemma-4-E4B-it")
# client = GemmaClient(model_id="google/gemma-4-31B-it")
client = Qwen3_Omni_Client()

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    files = data.get("files", [])
    clear = data.get("clear", False)
    
    # 调用你之前的 send_message 逻辑
    result = client.send_message(client.prepare_input(prompt, files), clear=clear)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
# cd /data1/niechang/Memory/omnimemory/nie_omni/TrainingDatasetConstruction/Light-Omni; conda activate gemma
# CUDA_VISIBLE_DEVICES=0,1,2,3 python /data1/niechang/Memory/omnimemory/nie_omni/TrainingDatasetConstruction/Light-Omni/scripts/gemma.py

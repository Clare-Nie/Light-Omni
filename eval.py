import os
import time
import json
import shutil
import random
random.seed(42)

import torch
import librosa
import numpy as np
import torch.multiprocessing as mp
from datasets import load_dataset

import scripts
from scripts import config
from scripts import utils
from scripts import tools


def get_lvbench_datasets():
    global LVBENCH_DATASET_PATH, LVBENCH_VIDEO_ROOT
    path = LVBENCH_DATASET_PATH
    video_root = LVBENCH_VIDEO_ROOT
    result = {}
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            v_path = os.path.join(video_root, f"{item['key']}.mp4")
            result[v_path] = []
            
            for qa in item['qa']:
                parts = qa['question'].split('\n')
                q_text = parts[0].strip()
                opts = [
                    "A. " + parts[1].split(')')[-1].strip(),
                    "B. " + parts[2].split(')')[-1].strip(),
                    "C. " + parts[3].split(')')[-1].strip(),
                    "D. " + parts[4].split(')')[-1].strip()
                ]
                result[v_path].append({
                    "question": q_text,
                    "options": opts,
                    "answer": qa['answer'],
                    "other": qa
                })
    print(f"LVBench: {len(result)} videos, {sum(len(v) for v in result.values())} questions.")
    return result


def get_mmevideolong_datasets():
    global VIDEOMME_DATASET_PATH, VIDEOMME_VIDEO_ROOT
    dataset = load_dataset(VIDEOMME_DATASET_PATH, split="test")
    def filter_long_videos(example):
        return example.get('duration') == 'long'
    long_dataset = dataset.filter(filter_long_videos)
    print(f"长视频数据量: {len(long_dataset)}")
    # 打印示例
    print(long_dataset[0])
    result = {}
    for sample in long_dataset:
        video_path = os.path.join(VIDEOMME_VIDEO_ROOT, sample["videoID"] + ".mp4")
        if video_path not in result:
            result[video_path] = []
        result[video_path].append({
            "question": sample["question"],
            "options": sample["options"],
            "answer": sample["answer"],
            "other": sample
        })
    print(len(result), "long videos found.")
    print("Total questions: ", sum([len(x) for x in result.values()]))
    return result


def get_hippovlog_datasets():
    global HIPPOVLOG_DATASET_PATH, HIPPOVLOG_VIDEO_ROOT
    path = HIPPOVLOG_DATASET_PATH
    video_path = HIPPOVLOG_VIDEO_ROOT
    result = {}
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            v_path = os.path.join(video_path, f"{item['video_id']}.mp4")
            if v_path not in result:
                result[v_path] = []
            result[v_path].append({
                "question": item["question_text"],
                "options": [f'A. {item["options"]["A"]}', f'B. {item["options"]["B"]}', f'C. {item["options"]["C"]}', f'D. {item["options"]["D"]}'],
                "answer": item["correct_answer"],
                "other": item
            })
    print(f"HippoVlog: {len(result)} videos found, {sum(len(v) for v in result.values())} questions.")
    return result


def simulate_stream_processing(session_id, video_data, save_dir, samples, eval_model):
    agent = scripts.lightomni.LightOmniAgent(save_dir, client=eval_model)
    vad_handler = tools.vad.VADHandler(min_silence_duration_ms=1000)
    vad_handler.reset()

    buffer_data = []
    is_speaking = False
    force_cut = False
    MAX_FRAMES = 30 / config.FRAME_DURATION
    MIN_FRAMES = 5 / config.FRAME_DURATION

    print(f"🚀 Processing stream ({len(video_data)} frames)...", len(agent.data), save_dir)
    for item in video_data:
        ts, v_path, a_path = item
        if len(agent.data) > 0 and len(agent.data[-1].get("convs", [])) > 0:
            latest_time = agent.data[-1].get("convs", [])[-1]["message"]["end_time"]
            if utils.compare_time_strings(latest_time, ts) >= 0:
                continue
        v_path = agent.process_image(ts, v_path)
        # VAD Detection
        try:
            audio_chunk, _ = librosa.load(a_path, sr=16000)
        except:
            audio_chunk = np.zeros(16000)
        
        currently_speaking = vad_handler.push_audio(audio_chunk)
        force_cut = False

        if is_speaking:
            if not currently_speaking and len(buffer_data) < MIN_FRAMES:
                currently_speaking = True
            if len(buffer_data) >= (MAX_FRAMES-1):
                currently_speaking = False
                force_cut = True
        
        # Case 1: Speech Starts (Idle -> Speaking)
        if not is_speaking and currently_speaking:
            is_speaking = True
            # Handle excessive background
            if len(buffer_data) > config.MIN_BACKGROUND_BEFORE_SPEECH / config.FRAME_DURATION:
                bg_to_flush = buffer_data[:int(-2 / config.FRAME_DURATION)]
                buffer_data = buffer_data[int(-2 / config.FRAME_DURATION):] # Keep 2 seconds as context
                _process_and_send(agent, bg_to_flush, save_dir, "eval_memory")

        # Case 2: Speech Ends (Speaking -> Idle)
        elif is_speaking and not currently_speaking:
            is_speaking = False
            buffer_data.append((ts, v_path, a_path)) # Include last frame
            _process_and_send(agent, buffer_data, save_dir, "eval_memory")
            buffer_data = []
            if not force_cut:
                vad_handler.reset()
            continue

        # Case 3: Background Timeout (Idle -> Idle)
        elif not is_speaking:
            if len(buffer_data) >= config.BACKGROUND_INTERVAL / config.FRAME_DURATION:
                _process_and_send(agent, buffer_data, save_dir, "eval_memory")
                buffer_data = []

        # Append current frame to buffer
        buffer_data.append((ts, v_path, a_path))

    # Process remaining buffer
    if buffer_data:
        _process_and_send(agent, buffer_data, save_dir, "eval_memory")
    res = []
    for sample in samples:
        message = {
            "start_time": agent.data[-1].get("convs", [])[-1]["message"]["end_time"],
            "end_time": utils.start_time_add(agent.data[-1].get("convs", [])[-1]["message"]["end_time"]),
            "image_path": [],
            "audio_path": "",
            "text": sample["question"] + "\nOptions:\n" + "\n".join(sample["options"]),
            "tag": "eval_final_question"
        }
        response, _ = agent.send_message(message)
        res.append({"response": response,
            "ground_truth": sample["answer"],
            "sample": sample})
    return res


def _process_and_send(agent, buffer_data, save_dir, tag):
    if not buffer_data: return
    start_ts = buffer_data[0][0]
    end_ts = buffer_data[-1][0]
    v_paths = [x[1] for x in buffer_data]
    a_paths = [x[2] for x in buffer_data]
    
    os.makedirs(os.path.join(save_dir, config.RAW_DIALOGUE_DATA_PATH), exist_ok=True)
    if config.VIDEO_OUPUT_MODE != "image":
        tv = os.path.join(save_dir, config.RAW_DIALOGUE_DATA_PATH, f"{start_ts}—{end_ts}.mp4")
    else:
        tv = None
    ta = os.path.join(save_dir, config.RAW_DIALOGUE_DATA_PATH,f"{start_ts}—{end_ts}.wav")
    
    utils.merge_files(tv, a_paths, tv, ta)    
    message = {
        "start_time": start_ts,
        "end_time": end_ts,
        "image_path": v_paths,
        "audio_path": ta,
        "text": "null",
        "tag": tag
    }
    statr_time = time.time()
    response, _ = agent.send_message(message)
    print(f"time after message response: {time.time() - statr_time}", message["start_time"], message["end_time"])
    return response


def main_worker(gpu_id, total_gpus, eval_dataset):
    if eval_dataset == "mmevideolong":
        long_dataset = get_mmevideolong_datasets()
    elif eval_dataset == "lvbench":
        long_dataset = get_lvbench_datasets()
    elif eval_dataset == "hippovlog":
        long_dataset = get_hippovlog_datasets()
    else:
        raise ValueError("Invalid dataset name")
    save_path = f"{SAVE_DIR}/eval_{eval_dataset}_results"
    torch.cuda.set_device(gpu_id)
    eval_model = scripts.model.QwenOmni2_5Model(device_map={"": gpu_id})
    os.makedirs(save_path, exist_ok=True)
    all_keys = list(long_dataset.keys())
    my_keys = all_keys[gpu_id::total_gpus] 
    save_json = os.path.join(save_path, f"results_{gpu_id}.json")
    
    existing_data = []
    
    _N = len(existing_data)
    for idx, video_file in enumerate(my_keys):            
        samples = long_dataset[video_file]
        global_idx = all_keys.index(video_file)
        print(f"[GPU {gpu_id}] 处理: {video_file}")
        save_sample_path = os.path.join(save_path, f"{global_idx}")
        os.makedirs(save_sample_path, exist_ok=True)
        
        start_time = '2026-01-01 08:00:00'
        tmp_path = os.path.join(save_sample_path, f'frame_0.json')
        if os.path.exists(tmp_path):
            video_data = utils.load_json(tmp_path)
        else:
            video_data = utils.video_file_process_batch(start_time, video_file, os.path.join(save_sample_path, config.INPUT_DATA_PATH), clip_duration=config.FRAME_DURATION, video_output_mode=config.VIDEO_OUPUT_MODE,
                                                        long_video_sample=False)
            utils.save_json(video_data, tmp_path)
        if len(video_data) == 0:
            shutil.rmtree(save_sample_path)
            continue
        try:
            res = simulate_stream_processing(0, video_data, save_sample_path, samples, eval_model)
            existing_data.append(res)
            utils.save_json(existing_data, save_json)
        except Exception as e:
            print(e)
            assert 1==2

def merge_results(eval_dataset):
    save_path = f"{SAVE_DIR}/eval_{eval_dataset}_results"
    final_results = []
    for i in range(8):
        part_path = os.path.join(save_path, f"results_{i}.json")
        if os.path.exists(part_path):
            data = utils.load_json(part_path)
            for j in data:
                final_results.extend(j)
            os.remove(part_path)
    
    final_save_path = os.path.join(save_path, "results.json")
    if final_results:
        utils.save_json(final_results, final_save_path)
    print(f"合并完成，共 {len(final_results)} 条数据，保存至: {final_save_path}")

def eval_final_result(eval_dataset):
    from tqdm import tqdm
    base_dir = f"{SAVE_DIR}/eval_{eval_dataset}_results"
    result_path = os.path.join(base_dir, "results.json")
    output_path = os.path.join(base_dir, "results_scored.json")
    
    if not os.path.exists(result_path):
        print(f"Error: 找不到结果文件 {result_path}")
        return
    results = utils.load_json(result_path)
    print(f"正在评估 {len(results)} 条数据...")

    correct_count = 0
    total_count = 0
    
    for item in tqdm(results):
        question = item['sample']['question']
        options = item['sample']['options']
        model_response = item['response']
        ground_truth = item['ground_truth']
        
        option_letters = ['A', 'B', 'C', 'D', 'E', 'F'][:len(options)]
        options_str = "\n".join([f"{label}. {opt}" for label, opt in zip(option_letters, options)])
        pred_letter = model_response[0]

        is_correct = (pred_letter == ground_truth)
        if is_correct:
            correct_count += 1
        total_count += 1
        
        item["eval_extraction"] = pred_letter
        item["is_correct"] = is_correct

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(f"\n========================================")
    print(f"Evaluation Finished for {eval_dataset}")
    print(f"Total Samples: {total_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Results saved to: {output_path}")
    print(f"========================================")

    summary = {"dataset": eval_dataset, "accuracy": accuracy, "total": total_count, "correct": correct_count}
    results.append(summary)
    utils.save_json(results, output_path)


def eval_final_result_llm(eval_dataset):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm
    base_dir = f"{SAVE_DIR}/eval_{eval_dataset}_results"
    result_path = os.path.join(base_dir, "results.json")
    output_path = os.path.join(base_dir, "results_scored.json")
    
    if not os.path.exists(result_path):
        print(f"Error: 找不到结果文件 {result_path}")
        return
    results = utils.load_json(result_path)
    print(f"正在评估 {len(results)} 条数据...")

    model_id = "Qwen/Qwen3-8B" 
    print(f"Loading Evaluator Model: {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "/data1/cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218", 
        torch_dtype="auto",
        device_map="auto"
    ).eval()

    correct_count = 0
    total_count = 0
    
    for item in tqdm(results):
        question = item['sample']['question']
        options = item['sample']['options']
        model_response = item['response']
        ground_truth = item['ground_truth']
        
        option_letters = ['A', 'B', 'C', 'D', 'E', 'F'][:len(options)]
        options_str = "\n".join([f"{label}. {opt}" for label, opt in zip(option_letters, options)])
        prompt = f"""You are an intelligent answer extractor.
Review the user's question, the provided options, and the model's response.
Determine which option the model's response matches best.

Question: {question}

Options:
{options_str}

Model's Response: {model_response}

Task: Output ONLY the capital letter (A, B, C, D, etc.) of the chosen option. Do not output any explanation or other text.
The Correct Option Letter is:"""

        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=10,
                do_sample=False,
            )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        extraction = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        pred_letter = extraction[0].upper() if len(extraction) > 0 else "FAIL"
        if pred_letter not in option_letters:
            found = False
            for char in extraction:
                if char.upper() in option_letters:
                    pred_letter = char.upper()
                    found = True
                    break
            if not found:
                pred_letter = "FAIL"

        is_correct = (pred_letter == ground_truth)
        if is_correct:
            correct_count += 1
        total_count += 1
        
        item["eval_extraction"] = pred_letter
        item["is_correct"] = is_correct

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(f"\n========================================")
    print(f"Evaluation Finished for {eval_dataset}")
    print(f"Total Samples: {total_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Results saved to: {output_path}")
    print(f"========================================")

    summary = {"dataset": eval_dataset, "accuracy": accuracy, "total": total_count, "correct": correct_count}
    results.append(summary)
    utils.save_json(results, output_path)


LVBENCH_DATASET_PATH = "/data1/yangyan/benchmark/LVBCaption/LVBench_data/video_info.meta.jsonl"
LVBENCH_VIDEO_ROOT = "/data1/yangyan/benchmark/LVBCaption/LVBench_data/all_videos"

VIDEOMME_DATASET_PATH = "/data1/niechang/Memory/omnimemory/nie_omni/benchmarkDataset/videomme/videomme"
VIDEOMME_VIDEO_ROOT = "/data1/niechang/Memory/omnimemory/nie_omni/benchmarkDataset/videomme/video"

HIPPOVLOG_DATASET_PATH = "/data1/cache/datasets/HippoVlog/questions.jsonl"
HIPPOVLOG_VIDEO_ROOT = "/data1/cache/datasets/HippoVlog/videos"

SAVE_DIR = "/data1/niechang/Memory/omnimemory/nie_omni/benchmarkDataset"


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    eval_datasets = ['mmevideolong', 'lvbench', 'hippovlog']
    for eval_dataset in eval_datasets:
        WORLD_SIZE = 8
        print("start eval: ", eval_dataset)
        mp.spawn(main_worker, args=(WORLD_SIZE, eval_dataset), nprocs=WORLD_SIZE)    
        merge_results(eval_dataset)
        eval_final_result(eval_dataset)


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval.py



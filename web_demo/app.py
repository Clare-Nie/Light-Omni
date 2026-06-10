import os
import time
import base64
import json
import gc
from datetime import datetime
import multiprocessing as mp

import torch
import numpy as np
import librosa
import soundfile as sf
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from scripts import config
from scripts import utils
from scripts import tools
from scripts.lightomni import LightOmniAgent
from scripts.model import QwenOmni2_5Model


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

Q1_input = None     # 进程1发送给进程2的每一帧的数据包
Q2_memory = None    # 进程 2 发送给进程3的完整的一轮对话数据 
Q3_output = None    # 进程 2 发送给进程1的回复文本
Q_Sync = None       # 进程 3给进程2的同步信号

BASE_SAVE_DIR = "output"
CURRENT_SAVE_DIR = None
IS_STREAMING = False
IS_AGENT_READY = False

# ========================= Helper Functions for Audio =========================
def concatenate_audio_chunks(audio_paths, output_path, sr=16000):
    if not audio_paths:
        return None
    
    all_audio_data = []
    for path in audio_paths:
        try:
            audio, _ = librosa.load(path, sr=sr, mono=True)
            all_audio_data.append(audio)
        except Exception as e:
            print(f"Error loading audio chunk {path}: {e}")
            all_audio_data.append(np.zeros(sr * config.FRAME_DURATION))

    if not all_audio_data:
        return None

    concatenated_audio = np.concatenate(all_audio_data)
    sf.write(output_path, concatenated_audio, sr)
    return output_path


def process_2_inference(q1, q2, q3, q_sync, save_dir_base):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    current_session_save_dir = None
    agent = None
    vad_handler = None
    
    buffer_data = []
    is_speaking = False
    
    MAX_FRAMES = 30 / config.FRAME_DURATION
    MIN_FRAMES = 3 / config.FRAME_DURATION

    eval_model = QwenOmni2_5Model(device_map={"": 0})
    print("[进程2-GPU0] 交互推理引擎已启动，等待连接请求...")

    while True:
        while not q_sync.empty():
            sync_msg = q_sync.get()
            if sync_msg == "MEMORY_UPDATED":
                if agent:
                    agent.data = json.load(open(agent.data_file_path, 'r', encoding='utf-8'))
                    agent.retriever.reload()
                    print("[进程2-GPU0] 已同步最新记忆！")

        # 2. 从 Q1 获取多模态流或等待连接
        if not q1.empty():
            item = q1.get()
            if item == "CONNECT_AGENT":
                username = q1.get()
                current_session_save_dir = os.path.join(save_dir_base, username)
                os.makedirs(current_session_save_dir, exist_ok=True)
                os.makedirs(os.path.join(current_session_save_dir, "raw"), exist_ok=True)
                
                if agent is not None:
                    del agent
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                agent = LightOmniAgent(current_session_save_dir, client=eval_model, cache=True)
                vad_handler = tools.vad.VADHandler(min_silence_duration_ms=500)
                vad_handler.reset()
                
                print(f"[进程2-GPU0] Agent已为用户'{username}'连接，数据保存至: {current_session_save_dir}")
                q3.put({"type": "agent_status", "status": "ready"})
                continue

            if agent is None:
                print("[进程2-GPU0] 未连接 Agent，跳过流数据...")
                continue

            ts, v_path, a_path = item
            if len(agent.data) > 0 and len(agent.data[-1].get("convs", [])) > 0:
                latest_conv_time = agent.data[-1]["convs"][-1]["message"]["end_time"]
                if utils.compare_time_strings(latest_conv_time, ts) > 0:
                    print(f"[进程2-GPU0] 警告: 当前帧时间戳 {ts} 早于上一次对话的结束时间 {latest_conv_time}。跳过此帧。")
                    continue

            v_path_processed = agent.process_image(ts, v_path) # 假设这个方法返回处理后的路径
            
            try:
                audio_chunk, _ = librosa.load(a_path, sr=16000, mono=True)
            except Exception as e:
                print(f"[进程2-GPU0] Error loading audio chunk {a_path}: {e}")
                audio_chunk = np.zeros(int(16000 * config.FRAME_DURATION), dtype=np.float32) # 使用静音填充
            
            currently_speaking = vad_handler.push_audio(audio_chunk)
            force_cut = False

            if is_speaking:
                # 1. 强制延长：如果 VAD 说说话结束了，但当前攒的片段不足 MIN_FRAMES，强制维持 speaking 状态继续录制
                if not currently_speaking and len(buffer_data) < MIN_FRAMES:
                    currently_speaking = True
                # 2. 强制截断：如果片段已经超过 MAX_FRAMES，强制认为结束，触发 Case 2 发送
                if len(buffer_data) >= (MAX_FRAMES - 1): # MAX_FRAMES-1 是因为当前帧还没加入
                    currently_speaking = False
                    force_cut = True
            
            # Case 1: Speech Starts (Idle -> Speaking)
            if not is_speaking and currently_speaking:
                is_speaking = True
                print(f"[进程2-GPU0] VAD Detected: Speech Starts at {ts}")
                    
                if len(buffer_data) > config.MIN_BACKGROUND_BEFORE_SPEECH / config.FRAME_DURATION:
                    split_idx = int(-config.MIN_BACKGROUND_BEFORE_SPEECH / config.FRAME_DURATION)
                    bg_to_memory = buffer_data[:split_idx]
                    buffer_data = buffer_data[split_idx:] 

                    if bg_to_memory:
                        bg_a_paths = [x[2] for x in bg_to_memory]
                        bg_combined_a_path = os.path.join(current_session_save_dir, "raw", f"{bg_to_memory[0][0]}-{bg_to_memory[-1][0]}.wav")
                        bg_concatenated_file = concatenate_audio_chunks(bg_a_paths, bg_combined_a_path)
                        bg_message = {
                            "tag": "inference",
                            "start_time": bg_to_memory[0][0],
                            "end_time": bg_to_memory[-1][0],
                            "image_path": [x[1] for x in bg_to_memory],
                            "audio_path": bg_concatenated_file,
                            "text": ""
                        }
                        q2.put({"message": bg_message, "response": "", "retrieve_state": {}})

            # Case 2: Speech Ends (Speaking -> Idle)
            elif is_speaking and not currently_speaking:
                is_speaking = False
                buffer_data.append((ts, v_path_processed, a_path)) # Include last frame
                
                print(f"[进程2-GPU0] VAD Detected: Speech Ends at {ts}. Triggering inference.")
                q3.put({"type": "system_event", "event": "vad_detected"})
                
                # 触发推理
                if buffer_data:
                    current_segment_v_paths = [item[1] for item in buffer_data]
                    current_segment_a_paths = [item[2] for item in buffer_data]
                    

                    start_time_segment = buffer_data[0][0]
                    end_time_segment = buffer_data[-1][0]
                    combined_audio_path = os.path.join(os.path.join(current_session_save_dir, "raw"), f"{start_time_segment}-{end_time_segment}.wav")
                    concatenated_audio_file = concatenate_audio_chunks(current_segment_a_paths, combined_audio_path)

                    message = {
                        "tag": "inference",
                        "start_time": start_time_segment,
                        "end_time": end_time_segment,
                        "image_path": current_segment_v_paths,
                        "audio_path": concatenated_audio_file,
                        "text": "" # ASR is run internally by process_single_message
                    }
                    inference_start_wall = time.time()
                    response_text, turn_data = agent.send_message(message, update_memory=False)
                    end_dt = datetime.strptime(end_time_segment, "%Y-%m-%d %H:%M:%S")
                    total_latency = time.time() - end_dt.timestamp()
                    pure_inference_time = time.time() - inference_start_wall
                    print(f"[进程2-GPU0] {time.strftime('%Y-%m-%d %H:%M:%S')}", f"对话结束于: {end_time_segment}", f"  - 响应总延迟 (距第一帧): {total_latency:.2f}s", f"  - 纯模型推理耗时: {pure_inference_time:.2f}s")
                    print(turn_data['retrieve_state'])

                    if response_text:
                        q3.put({"type": "text_response", "text": response_text})
                    
                    if turn_data: 
                        q2.put(turn_data)

                    buffer_data = []
                
                if not force_cut:
                    vad_handler.reset()
                continue

            # Case 3: Background Timeout (Idle -> Idle)
            elif not is_speaking:
                if len(buffer_data) >= config.BACKGROUND_INTERVAL / config.FRAME_DURATION:
                    print(f"[进程2-GPU0] VAD Detected: Background Timeout at {ts}. Flushing background data.")
                    if buffer_data:
                        bg_a_paths = [x[2] for x in buffer_data]
                        bg_v_paths = [x[1] for x in buffer_data]
                        start_time = buffer_data[0][0]
                        end_time = buffer_data[-1][0]
                        
                        bg_combined_a_path = os.path.join(current_session_save_dir, "raw", f"{start_time}-{end_time}_bg.wav")
                        bg_concatenated_file = concatenate_audio_chunks(bg_a_paths, bg_combined_a_path)
                        
                        bg_message = {
                            "tag": "inference",
                            "start_time": start_time,
                            "end_time": end_time,
                            "image_path": bg_v_paths,
                            "audio_path": bg_concatenated_file,
                            "text": ""
                        }
                        inference_start_wall = time.time()
                        response_text, turn_data = agent.send_message(bg_message, update_memory=False)
                        pure_inference_time = time.time() - inference_start_wall
                        print(f"[进程2-GPU0] 背景感知完成 - 耗时: {pure_inference_time:.2f}s")

                        if response_text and response_text.strip():
                            print(f"[进程2-GPU0] Agent 主动发言: {response_text}")
                            q3.put({"type": "text_response", "text": response_text})
                        
                        if turn_data:
                            q2.put(turn_data)
                    buffer_data = []

            # Append current frame to buffer
            buffer_data.append((ts, v_path_processed, a_path))

        else:
            time.sleep(0.02)


def process_3_memory(q2, q3, q_sync, save_dir_base):
    # 确保在进程启动时设置 CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print(f"[进程3-GPU1] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    memory_agent = None
    current_session_save_dir = None

    memory_model = QwenOmni2_5Model(device_map={"": 1})
    print("[进程3-GPU1] 离线记忆引擎已启动，等待连接请求...")

    while True:
        if not q2.empty():
            mem_start_time = time.perf_counter()

            item = q2.get()
            if item == "CONNECT_AGENT":
                username = q2.get() # 下一个 item 是 username
                current_session_save_dir = os.path.join(save_dir_base, username)
                os.makedirs(current_session_save_dir, exist_ok=True)

                if memory_agent is not None:
                    del memory_agent
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # 初始化独立的记忆Agent
                memory_agent = LightOmniAgent(current_session_save_dir, client=memory_model, cache=False)
                print(f"[进程3-GPU1] Agent已为用户'{username}'连接，数据保存至: {current_session_save_dir}")
                continue

            if memory_agent is None:
                print("[进程3-GPU1] 未连接 Agent，跳过记忆数据...")
                continue

            turn_data = item
            q3.put({"type": "system_event", "event": "memory_start"})
            print("[进程3-GPU1] 收到新对话，开始长短时记忆整合...")
            
            # 确保 memory_agent 的 data 结构与 P2 同步
            if not memory_agent.data or not memory_agent.data[-1].get("convs"):
                memory_agent.data.append({"convs": []})
            
            # 检查时间戳是否需要开启新 session (P3 也要独立判断)
            last_conv_time = None
            if memory_agent.data and memory_agent.data[-1].get("convs"):
                last_conv_time = memory_agent.data[-1]["convs"][-1]["message"]["end_time"]
                if utils.is_time_diff_over_minutes(turn_data['message']['start_time'], last_conv_time, memory_agent.session_interval_time):
                    memory_agent.update_long_term_memory()
                    memory_agent.data.append({"convs": []})

            memory_agent.data[-1]["convs"].append(turn_data)
            memory_agent.memory_consolidation()
            memory_agent.update()

            q3.put({"type": "system_event", "event": "memory_done"})
            q_sync.put("MEMORY_UPDATED")

            mem_duration = time.perf_counter() - mem_start_time
            last_frame_time = turn_data['message'].get('end_time', 'Unknown')
            print(f"[进程3-GPU1] {time.strftime('%Y-%m-%d %H:%M:%S')} 记忆整合完毕 (截至时间: {last_frame_time})！耗时: {mem_duration:.2f}s。同步信号已发出。")
        else:
            time.sleep(0.02)


@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect_agent')
def handle_connect_agent(data):
    global CURRENT_SAVE_DIR
    username = data.get('username')
    if not username:
        emit('error', {'message': 'Username is required to connect agent.'})
        return

    CURRENT_SAVE_DIR = os.path.join(BASE_SAVE_DIR, username)
    os.makedirs(CURRENT_SAVE_DIR, exist_ok=True)

    Q1_input.put("CONNECT_AGENT")
    Q1_input.put(username)
    Q2_memory.put("CONNECT_AGENT")
    Q2_memory.put(username)

    emit('agent_status', {'status': 'connecting'})
    print(f"[主进程] 连接请求已发送给Agent，用户: {username}, 存储路径: {CURRENT_SAVE_DIR}")


@socketio.on('start_chat')
def handle_start_chat():
    global IS_STREAMING
    if IS_AGENT_READY:
        IS_STREAMING = True
        print("[主进程] 对话已开始，流数据将发送至推理引擎。")
        emit('chat_status', {'status': 'started'})
    else:
        emit('error', {'message': '请先连接 Agent 并等待初始化完成。'})

@socketio.on('stop_chat')
def handle_stop_chat():
    global IS_STREAMING
    IS_STREAMING = False
    print("[主进程] 对话已暂停。")
    emit('chat_status', {'status': 'stopped'})


@socketio.on('stream_chunk')
def handle_stream(data):
    """
    接收前端每 1 秒发来的数据: { 'timestamp': '...', 'image_base64': '...', 'audio_base64': '...' }
    """
    if CURRENT_SAVE_DIR is None or not IS_STREAMING:
        return
    ts_str = data.get('timestamp')
    safe_ts = ts_str # .replace(" ", "_").replace(":", "-")
    save_dir_img = os.path.join(CURRENT_SAVE_DIR, "input", "sampled_images")
    save_dir_audio = os.path.join(CURRENT_SAVE_DIR, "input", "audio_clips")
    os.makedirs(save_dir_img, exist_ok=True)
    os.makedirs(save_dir_audio, exist_ok=True)

    try:
        img_b64 = data.get('image_base64', '')
        if ',' in img_b64:
            img_data = base64.b64decode(img_b64.split(',')[1])
            v_path = os.path.join(save_dir_img, f"image_{safe_ts}.jpg")
            with open(v_path, "wb") as f:
                f.write(img_data)
        else:
            return

        a_path = os.path.join(save_dir_audio, f"audio_{safe_ts}.wav")
        audio_b64 = data.get('audio_base64', '')
        
        if audio_b64 and ',' in audio_b64:
            audio_data = base64.b64decode(audio_b64.split(',')[1])
            with open(a_path, "wb") as f:
                f.write(audio_data)
        else:
            empty_audio = np.zeros(16000, dtype=np.float32)
            sf.write(a_path, empty_audio, 16000)
        Q1_input.put((ts_str, v_path, a_path))
        print(f"[主进程] 成功保存数据帧 -> image_{safe_ts}.jpg / audio_{safe_ts}.wav")

    except Exception as e:
        print(f"[主进程] 保存流数据时发生错误: {e}")
def output_listener():
    """监听 Q3_output，将模型的回复通过 WebSocket 推送给前端"""
    global IS_AGENT_READY
    print("[主进程-监控] 消息监听线程已启动...")
    while True:
        if not Q3_output.empty():
            response_data = Q3_output.get()

            if response_data.get("type") == "agent_status" and response_data.get("status") == "ready":
                IS_AGENT_READY = True
                
            print(f"[主进程-监控] 准备推送消息到前端: {response_data.get('type')}")
            socketio.emit('model_response', response_data)
        time.sleep(0.05)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) 
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    Q1_input = mp.Queue()
    Q2_memory = mp.Queue()
    Q3_output = mp.Queue()
    Q_Sync = mp.Queue()


    p2 = mp.Process(target=process_2_inference, args=(Q1_input, Q2_memory, Q3_output, Q_Sync, BASE_SAVE_DIR))
    p2.start()
    p3 = mp.Process(target=process_3_memory, args=(Q2_memory, Q3_output, Q_Sync, BASE_SAVE_DIR))
    p3.start()
    
    socketio.start_background_task(target=output_listener)
    socketio.run(app, host='127.0.0.1', port=19090, allow_unsafe_werkzeug=True)
                #  ssl_context=('./cert.pem', './key.pem')) # http --> https

# CUDA_VISIBLE_DEVICES=6,7 python -m web_demo.app
# openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
# Common Name (e.g. server FQDN or YOUR name) []:192.168.31.234 # ipconfig getifaddr en0
# https://10.49.5.24:19090/
# git add . && git commit -m "update" && git push




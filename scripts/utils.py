import os
import uuid
import json
import time
import random
import functools
import shutil

import cv2
import librosa
import numpy as np
import json_repair
import subprocess
import soundfile as sf

from moviepy.editor import VideoFileClip, concatenate_videoclips
from typing import List, Tuple, Union
from datetime import datetime, timedelta


def get_world_time():
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return current_time_str


def start_time_add(start_time, add_day=0, add_hour=0, add_min=0, add_sec=1):
    dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    delta = timedelta(
        days=add_day,
        hours=add_hour,
        minutes=add_min,
        seconds=add_sec
    )
    new_dt = dt + delta
    return new_dt.strftime("%Y-%m-%d %H:%M:%S")


def generate_64bit_id():
    '''Generate a unique ID using UUID version 4 (randomly generated numbers), and truncating it to the first 64 bits.'''
    full_uuid = uuid.uuid4().int
    return full_uuid & 0xFFFFFFFFFFFFFFFF


def get_random_2025_timestamp():
    import random
    start_date = datetime(2025, 1, 1, 0, 0, 0)
    end_date = datetime(2025, 12, 31, 23, 59, 59)
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())
    random_timestamp = random.randint(start_timestamp, end_timestamp)
    random_date = datetime.fromtimestamp(random_timestamp)
    return random_date.strftime("%Y-%m-%d %H:%M:%S")


def add_random_offset(base_time_str: str, max_days: int = 5) -> str:
    time_format = "%Y-%m-%d %H:%M:%S"
    try:
        base_date = datetime.strptime(base_time_str, time_format)
    except ValueError:
        raise ValueError(f"输入的时间字符串 '{base_time_str}' 格式不正确，应为 'YYYY-MM-DD HH:MM:SS'")
    random_days = random.randint(0, max_days)
    random_seconds = random.randint(0, 24 * 3600 - 1)
    time_offset = timedelta(days=random_days, seconds=random_seconds)
    new_date = base_date + time_offset
    return new_date.strftime(time_format)


def is_time_diff_over_minutes(time1, time2, threshold_minutes):
    '''Check if two times are more than threshold_minutes apart'''
    if time1 is None or time2 is None:
        return True
    time_format = "%Y-%m-%d %H:%M:%S"
    t1 = datetime.strptime(time1, time_format)
    t2 = datetime.strptime(time2, time_format)    
    diff_minutes = abs((t2 - t1).total_seconds()) / 60
    return diff_minutes > threshold_minutes


def compare_time_strings(time_str1, time_str2, time_format="%Y-%m-%d %H:%M:%S"):
    '''
    return
        -1: time_str1 < time_str2
         0: time_str1 == time_str2
         1: time_str1 > time_str2
    '''
    if time_str1 is None or time_str2 is None:
        return 0
    dt1 = datetime.strptime(time_str1, time_format)
    dt2 = datetime.strptime(time_str2, time_format)

    if dt1 < dt2:
        return -1
    elif dt1 > dt2:
        return 1
    else:
        return 0


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def format_normalize(text):
    text = text.strip().strip("json").strip("```").strip("'''").strip()
    return text


def video_file_process(
    start_time: str, 
    video_path: str, 
    save_dir: str = None,
    clip_duration: float = 1.0,
    video_output_mode: str = 'image',
    image_sample_point: str = 'middle'
) -> List[Tuple[str, Union[str, List[np.ndarray], np.ndarray], Union[str, np.ndarray]]]:

    if video_output_mode not in ['clip', 'image']:
        raise ValueError("video_output_mode must be 'clip' or 'image'")
    if image_sample_point not in ['first', 'middle', 'last']:
        raise ValueError("image_sample_point must be 'first', 'middle', or 'last'")

    if save_dir:
        if video_output_mode == 'clip':
            video_save_dir = os.path.join(save_dir, "video_clips")
        else:
            video_save_dir = os.path.join(save_dir, "sampled_images")
        
        audio_save_dir = os.path.join(save_dir, "audio_clips")
        os.makedirs(video_save_dir, exist_ok=True)
        os.makedirs(audio_save_dir, exist_ok=True)

    try:
        y, sr = librosa.load(video_path, sr=16000)
    except Exception as e:
        raise ValueError("视频读取错误")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        print(f"Error opening video file: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError(f"Error: Video FPS is 0 for {video_path}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    duration = min(librosa.get_duration(y=y, sr=sr), cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
    
    frames_per_clip = int(round(fps * clip_duration))
    samples_per_clip = int(round(sr * clip_duration))
    
    result = []
    current_sec = 0.0
    clip_idx = 0

    while current_sec + clip_duration <= duration:
        timestamp = start_time_add(start_time, add_sec=current_sec)
        s_start = int(current_sec * sr)
        audio_chunk = y[s_start : s_start + samples_per_clip]
        
        if save_dir:
            audio_path = os.path.join(audio_save_dir, f"audio_{timestamp}.wav")
            sf.write(audio_path, audio_chunk, sr)
            a_out = audio_path
        else:
            a_out = audio_chunk

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_sec * fps))
        frames = []
        for _ in range(frames_per_clip):
            ret, frame = cap.read()
            if ret: 
                frames.append(frame)
            else: 
                break
            
        if not frames or (video_output_mode == 'clip' and len(frames) < frames_per_clip): 
            break

        if video_output_mode == 'clip':
            if save_dir:
                video_clip_path = os.path.join(video_save_dir, f"clip_{timestamp}.mp4")
                out = cv2.VideoWriter(video_clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                for f in frames: out.write(f)
                out.release()
                v_out = video_clip_path
            else:
                v_out = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        
        else:
            if image_sample_point == 'first':
                sampled_frame = frames[0]
            elif image_sample_point == 'last':
                sampled_frame = frames[-1]
            else: # 'middle'
                sampled_frame = frames[len(frames) // 2]
            
            if save_dir:
                image_path = os.path.join(video_save_dir, f"image_{timestamp}.png")
                cv2.imwrite(image_path, sampled_frame)
                v_out = image_path
            else:
                v_out = cv2.cvtColor(sampled_frame, cv2.COLOR_BGR2RGB)
        result.append((timestamp, v_out, a_out))

        current_sec += clip_duration
        clip_idx += 1

    cap.release()
    return result


def video_file_process_batch(
    start_time: str, 
    video_path: str, 
    save_dir: str,
    clip_duration: float = 1.0,
    video_output_mode: str = 'image',
    long_video_sample = True,
) -> List[Tuple[str, str, str]]:
    
    if not save_dir:
        raise ValueError("save_dir must be provided for batch processing mode.")
    if video_output_mode not in ['clip', 'image']:
        raise ValueError("video_output_mode must be 'clip' or 'image'")

    temp_dir = os.path.join(save_dir, "temp_" + os.path.basename(video_path))
    video_temp_dir = os.path.join(temp_dir, "video")
    audio_temp_dir = os.path.join(temp_dir, "audio")
    
    os.makedirs(video_temp_dir, exist_ok=True)
    os.makedirs(audio_temp_dir, exist_ok=True)

    final_video_dir = os.path.join(save_dir, "sampled_images" if video_output_mode == 'image' else "video_clips")
    final_audio_dir = os.path.join(save_dir, "audio_clips")
    os.makedirs(final_video_dir, exist_ok=True)
    os.makedirs(final_audio_dir, exist_ok=True)

    duration = get_video_duration_ffprobe(video_path)
    t_args = []
    if duration and duration > 600 and long_video_sample:
        t_args = ['-t', str(random.triangular(180, 1800, 180))]
    try:
        audio_command = ['ffmpeg'] + t_args + [
            '-i', video_path,
            '-f', 'segment', '-segment_time', str(clip_duration),
            '-c:a', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            '-reset_timestamps', '1',
            os.path.join(audio_temp_dir, 'audio_%05d.wav')
        ]
        try:
            subprocess.run(audio_command, check=True, capture_output=True)
        except:
            limit_args = t_args if t_args else ['-t', str(duration)]
            silence_command = ['ffmpeg'] + limit_args + [
                '-f', 'lavfi', '-i', 'anullsrc=r=16000:cl=mono', # 生成静音源
                '-f', 'segment', '-segment_time', str(clip_duration),
                '-c:a', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                '-reset_timestamps', '1',
                os.path.join(audio_temp_dir, 'audio_%05d.wav')
            ]
            subprocess.run(silence_command, check=True, capture_output=True)

        # print("Batch processing video...")
        if video_output_mode == 'image':
            # 每 clip_duration 秒提取一帧
            video_command = ['ffmpeg'] + t_args + [
                '-i', video_path,
                '-vf', f'fps=1/{clip_duration}',
                '-q:v', '2', # 保证图片质量
                os.path.join(video_temp_dir, 'image_%05d.png')
            ]
        else: # 'clip'
            # 批量切分视频片段
            video_command = ['ffmpeg'] + t_args + [
                '-i', video_path,
                '-f', 'segment',
                '-segment_time', str(clip_duration),
                '-c', 'copy', # 尝试使用极速的流复制
                '-reset_timestamps', '1',
                os.path.join(video_temp_dir, 'clip_%05d.mp4')
            ]
        
        try:
             subprocess.run(video_command, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            if video_output_mode == 'clip' and '-c' in video_command and 'copy' in video_command:
                print("Stream copy failed, falling back to re-encoding...")
                video_command[video_command.index('-c')] = '-c:v'
                video_command[video_command.index('copy')] = 'libx264' # 或者其他编码器
                subprocess.run(video_command, check=True, capture_output=True)
            else:
                raise e

        result = []
        audio_files = sorted(os.listdir(audio_temp_dir))
        video_files = sorted(os.listdir(video_temp_dir))
        
        num_clips = min(len(audio_files), len(video_files))
        
        for i in range(num_clips):
            current_sec = i * clip_duration
            timestamp = start_time_add(start_time, add_sec=current_sec)

            new_audio_name = f"audio_{timestamp}.wav"
            new_video_name = f"image_{timestamp}.png" if video_output_mode == 'image' else f"clip_{timestamp}.mp4"

            final_audio_path = os.path.join(final_audio_dir, new_audio_name)
            final_video_path = os.path.join(final_video_dir, new_video_name)
            
            shutil.move(os.path.join(audio_temp_dir, audio_files[i]), final_audio_path)
            shutil.move(os.path.join(video_temp_dir, video_files[i]), final_video_path)
            
            result.append((timestamp, final_video_path, final_audio_path))
            
    except Exception as e:
        print(f"An error occurred during batch processing: {e}")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"FFmpeg Error Output: {e.stderr.decode()}")
        return []
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            
    print("Batch processing complete.")
    return result


def merge_files(video_paths, audio_paths, out_v, out_a):
    if audio_paths:
        try:
            combined_audio = []
            sr = 16000
            for ap in audio_paths:
                y, _ = librosa.load(ap, sr=sr)
                combined_audio.append(y)
            
            if combined_audio:
                final_audio = np.concatenate(combined_audio)
                sf.write(out_a, final_audio, sr)
            else:
                sf.write(out_a, np.zeros(16000), 16000)
        except Exception as e:
            raise ValueError("音频合并错误")

    if video_paths:
        try:
            clips = []
            for vp in video_paths:
                clip = VideoFileClip(vp)
                clips.append(clip)
            
            if clips:
                final_clip = concatenate_videoclips(clips, method="compose")
                final_clip.write_videofile(out_v, codec="libx264", audio=False, verbose=False, logger=None)
                for clip in clips: clip.close()
            else:
                shutil.copy(video_paths[0], out_v)
        except Exception as e:
            raise ValueError("视频合并错误")


def parse_response(response_data):
    if isinstance(response_data, str):
        try:
            clean_text = response_data.replace("```json", "").replace("```", "").strip()
            response_data = json.loads(clean_text)
        except Exception:
            return {
                "action": "answer", 
                "answer": "null", 
                "thought": "JSON parsing failed."
            }
    
    if not isinstance(response_data, dict):
         return {"action": "answer", "answer": "null", "thought": "Invalid output format."}

    res = {}
    raw_action = response_data.get("action", "answer")
    content = response_data.get("content", "null")
    thought = response_data.get("thought", "No thought provided.")

    res["thought"] = thought
    if raw_action == "retrieve":
        res["action"] = "retrieve"
        res["retrieve"] = content
    else:
        res["action"] = "answer"
        res["answer"] = content

    return res


def time_str_to_seconds(time_str: str) -> float:
    if time_str is None:
        return 0.0
    try:
        dt_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        return dt_obj.timestamp()
    except ValueError:
        pass
    try:
        parts = time_str.split(':')
        h = int(parts[0])
        m = int(parts[1])
        
        if '.' in parts[2]:
            s, ms = parts[2].split('.')
            total_seconds = h * 3600 + m * 60 + int(s) + int(ms) / 1000
        else:
            total_seconds = h * 3600 + m * 60 + int(parts[2])
            
        return float(total_seconds)
    except (ValueError, IndexError):
        print(f"警告：无法解析未知的时间格式 '{time_str}'，返回 0.0")
        return 0.0


def get_video_duration_ffprobe(video_path):
    try:
        command = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None
    except Exception:
        return None


def get_audio_duration(path):
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return float(result.stdout)


def safe_json_repair(response_text):
    if not response_text:
        return {}
    cleaned_text = "".join(
        ch for ch in response_text 
        if ord(ch) >= 32 or ch in '\n\r\t'
    )
    try:
        decoded = json_repair.loads(cleaned_text)
        if isinstance(decoded, str) and (decoded.startswith('{') or decoded.startswith('[')):
            return json.loads(decoded)
        return decoded
    except Exception as e:
        print(f"JSON Repair 失败: {e}\n原始文本片段: {cleaned_text[:100]}...")
        return response_text


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Found function [{func.__name__}] finished in {end_time - start_time:.4f}s")
        return result
    return wrapper


def time_all_methods(cls):
    for name, method in vars(cls).items():
        if callable(method):
            setattr(cls, name, timer(method))
    return cls

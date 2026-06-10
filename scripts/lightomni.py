import os
import json
import copy
import re
import random
import time

import torch
from collections import Counter
import transformers
import logging, warnings
logging.basicConfig(level=logging.ERROR, force=True); warnings.filterwarnings('ignore')
transformers.utils.logging.set_verbosity_error()

from scripts import tools
from scripts import config
from scripts import utils
from scripts import model
from scripts import prompts as prompts_inf


class LightOmniAgent:
    def __init__(self,
                 root_path,
                 session_interval_time=15,
                 trigger_summary=12,
                 client=None,
                 cache=True,
                 merge_factor=8,
                 use_gemini=False
                 ):
        self.root_path = root_path
        if client is None:
            self.client = model.QwenOmni2_5Model()
        else:
            self.client = client
        self.use_gemini = use_gemini
        if self.use_gemini:
            self.gemini = model.GeminiClient()
        if cache:
            self.client.start_feature_cache()
        self.merge_factor = merge_factor

        self.profile_tool = tools.profile_manager.ProfileTool(root_path=root_path)
        self.retriever = tools.retriever.Retriever(root_path, omni_model=self.client)
        self.asr_model = tools.asr.ASRHandler()

        self.current_turn_faces = set()
        self.accumulated_faces_for_ltm = set()
        self.session_interval_time = session_interval_time
        self.trigger_summary=trigger_summary
        self.data_file_path = os.path.join(root_path, config.DATASET_CONVERSATION_PATH)
        if os.path.exists(self.data_file_path):
            self.data = json.load(open(self.data_file_path))
        else:
            self.data = []
        self.data_log_path = os.path.join(root_path, "log.json")
        if os.path.exists(self.data_log_path):
            self.data_log = json.load(open(self.data_log_path))
        else:
            self.data_log = {}

    def update(self):
        with open(self.data_file_path, 'w') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
        with open(self.data_log_path, 'w') as f:
            json.dump(self.data_log, f, ensure_ascii=False, indent=4)

    def update_log(self):
        with open(self.data_log_path, 'w') as f:
            json.dump(self.data_log, f, ensure_ascii=False, indent=4)
        
        if "log_qa" not in self.data[-1]:
            self.data[-1]["log_qa"] = len(self.data_log["qa_generation"])
            with open(self.data_file_path, 'w') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=4)
    
    def process_image(self, timestamp, image_path):
        resave_path, faces = self.profile_tool.process_image(timestamp, image_path, long_edge=480)
        self.current_turn_faces.update(faces)
        return resave_path


    def get_last_conv_time(self):
        res = None
        if self.data and not self.data[-1].get("convs", []):
            self.data = self.data[:-1]
        if self.data:
            res = self.data[-1]["convs"][-1]["message"]["end_time"]
        else:
            self.data.append({"convs": []})
        return res


    def get_model_response(self, prompt, files, json_output=False, clear_context=True, log_tag="unknown"):
        _prompt = copy.deepcopy(prompt)
        _files = copy.deepcopy(files)
        if files:
            assert len(files) == prompt.count("<img>") + prompt.count("<audio>")
        if log_tag=="response_generation" and self.use_gemini:
            response = self.gemini.send_message(self.gemini.prepare_input(copy.deepcopy(_prompt), copy.deepcopy(_files)), clear=True)
        else:
            response = self.client.send_message(self.client.prepare_input(copy.deepcopy(_prompt), copy.deepcopy(_files)), mode="response" if log_tag=="response_generation" else "memory")
        if json_output:
            try:
                response = utils.format_normalize(response)
                response = utils.safe_json_repair(response)
            except:
                print("Error parsing response:", utils.format_normalize(response))
                raise Exception("Error parsing response")
        
        if log_tag not in self.data_log:
            self.data_log[log_tag] = []

        self.data_log[log_tag].append({
            "prompt": copy.deepcopy(_prompt),
            "response": response,
            "files": copy.deepcopy(_files),
        })
        return response


    def get_unified_history(self):
        all_logs = []
        raw_imgs = []
        
        _N = sum([len(session.get("logs", [])) for session in self.data])
        for session in self.data:
            logs = session.get("logs", [])
            for idx, log in enumerate(logs):
                entry = (
                    f"[Time: {log.get('start_time')} - {log.get('end_time')}]\n"
                    f"Visual: {log.get('visual', 'null')}\n"
                    f"Audio: {log.get('auditory', 'null')}\n"
                    f"Assistant Response: {log.get('assistant', 'null')}\n"
                )
                threshold = min(1.0, 24 / (_N + 1))
                if random.random() < threshold:
                    imgs = session["convs"][idx]["message"]["image_path"]
                    if imgs:
                        input_img = imgs[len(imgs) // 2]
                        entry += (f"Relevant image: <img>\n")
                        raw_imgs.append(input_img)
                all_logs.append(entry)
        if not all_logs:
            return "No history available.", raw_imgs
        return "\n".join(all_logs), raw_imgs

    
    def get_response_state_parallel(self, message):
        prompt = prompts_inf.OMNI_MEMORY_STAGE_1_PROMPT_INF
        prompt = prompt.replace("{START_TIME}", message["start_time"])
        prompt = prompt.replace("{END_TIME}", message["end_time"])
        prompt = prompt.replace("{INPUT_IMAGE_SEQUENCE}", "<img>" * len(message["image_path"]))
        prompt = prompt.replace("{INPUT_AUDIO_STREAM}", "<audio>" if message["audio_path"] else "null")
        prompt = prompt.replace("{INPUT_TEXT_STREAM}", message.get("text", "null"))# .split("\n\n\n")[-1]
        prompt = prompt.replace("{INPUT_FACES}", self.profile_tool.get_faces_profile(self.current_turn_faces))
        stm_text, stm_imgs = self.get_short_term_memory(img_ratio=0.1)
        prompt = prompt.replace("{SHORT_TERM_MEMORY}", stm_text)
        files = copy.deepcopy(message["image_path"] +[message["audio_path"]] if message["audio_path"] else message["image_path"])
        files = stm_imgs + files
        assert len(files) == prompt.count("<img>") + prompt.count("<audio>")
        result = self.client.get_response_state_parallel(self.client.prepare_input(copy.deepcopy(prompt), copy.deepcopy(files)), message, self.asr_model)
        return result


    def process_single_message(self, message):
        if message["tag"] == "eval_memory":
            return "", {}
        if message["tag"] == "inference":
            retrieve_state = self.get_response_state_parallel(message)
            if not retrieve_state.get("is_response", False):
                retrieve_state.pop("retrieve_embedding", None)
                return "", retrieve_state
            if retrieve_state.get("is_retrieve", False):
                semantic_memory_str, episodic_memory_str, ret_files = self.retrieve_memory(retrieve_state.get("retrieve_embedding", message["text"]))
            else:
                semantic_memory_str, episodic_memory_str, ret_files = "", "", []
        else:
            retrieve_state = self.get_response_state_parallel(message)
            if message["tag"] == "eval_final_question":
                retrieve_state["is_retrieve"] = True
            if retrieve_state.get("is_retrieve", False):
                semantic_memory_str, episodic_memory_str, ret_files = self.retrieve_memory(retrieve_state.get("retrieve_embedding", message["text"]))
            else:
                semantic_memory_str, episodic_memory_str, ret_files = "", "", []
        if self.use_gemini:
            prompt = prompts_inf.OMNI_MEMORY_STAGE_2_PROMPT_INF_gemini
        else:
            prompt = prompts_inf.OMNI_MEMORY_STAGE_2_PROMPT_INF
        prompt = prompt.replace("{START_TIME}", message["start_time"])
        prompt = prompt.replace("{END_TIME}", message["end_time"])
        prompt = prompt.replace("{INPUT_IMAGE_SEQUENCE}", "<img>" * len(message["image_path"]))
        prompt = prompt.replace("{INPUT_AUDIO_STREAM}", "<audio>" if message["audio_path"] else "null")
        prompt = prompt.replace("{INPUT_TEXT_STREAM}", message.get("text", "null"))
        prompt = prompt.replace("{INPUT_FACES}", self.profile_tool.get_faces_profile(self.current_turn_faces))
        stm_text, stm_imgs = self.get_short_term_memory(img_ratio=0.1)
        prompt = prompt.replace("{SHORT_TERM_MEMORY}", stm_text)
        files = copy.deepcopy(message["image_path"] +[message["audio_path"]] if message["audio_path"] else message["image_path"])
        files = stm_imgs + files
        prompt = prompt.replace("{RETRIEVED_SEMANTIC_MEMORY}", semantic_memory_str)
        prompt = prompt.replace("{RETRIEVED_EPISODIC_MEMORY}", episodic_memory_str)
        files = ret_files + files
        response = self.get_model_response(prompt, files, json_output=False, clear_context=True, log_tag="response_generation")

        self.retriever.reset()
        retrieve_state.pop("retrieve_embedding", None)
        return response, retrieve_state


    def retrieve_memory(self, query, img_ratio=0.1,
                        top_k_semantic=config.RetrieverConfig.TOP_K_SEMANTIC,
                        top_k_episodic=config.RetrieverConfig.TOP_K_EPISODIC,
                        add_num=False):
        if isinstance(query, str) and query == "":
            print("[Memory] Model did not provide a thought/query for retrieval.")
            return "", "", []

        retrieved_data = self.retriever.retrieve(
            query, 
            top_k_semantic=top_k_semantic, 
            top_k_episodic=top_k_episodic
        )

        semantic_results = retrieved_data.get("semantic", [])
        if semantic_results:
            if add_num:
                semantic_str = "\n".join(f"Entry {i}:\n{item['content']}" for i, item in enumerate(semantic_results))
            else:
                semantic_str = "\n".join([f"{item['content']}" for _, item in enumerate(semantic_results)])
        else:
            semantic_str = "No relevant semantic memory found."

        episodic_results = retrieved_data.get("episodic", [])
        raw_imgs = []
        if episodic_results:
            episodic_str_list = []
            for i, item in enumerate(episodic_results):
                content = item['content']
                res = f"Entry {i}:\n" if add_num else ""
                res += (
                    f"{content['start_time']}——{content['end_time']}\n"
                    f"Visual description: {content['visual description']}\n"
                    f"Audio description: {content['audio description']}\n"
                    f"Assistant Response: {content.get('assistant', 'null')}\n"
                )
                if config.RetrieverConfig.RETRIEVE_RAW_DATA:
                    imgs = []
                    for conv in content["raw_convs"]:
                        imgs.extend(conv["message"]["image_path"])
                    if imgs:
                        k = max(1, int(len(imgs) * img_ratio))
                        indices = [int(i * len(imgs) / k) for i in range(k)]
                        sampled_imgs = [imgs[i] for i in indices]
                        tmp = "Relevant image: " + "<img>" * len(sampled_imgs) + "\n"
                        res += tmp
                        raw_imgs.extend(sampled_imgs)

                episodic_str_list.append(res)
            episodic_str = "\n".join(episodic_str_list)
        else:
            episodic_str = "No relevant episodic memory found."
            
        return semantic_str, episodic_str, raw_imgs


    def update_short_term_memory(self):
        prompt = prompts_inf.OMNI_MEMORY_LOG_GENERATE_PROMPT_INF
        message = self.data[-1]["convs"][-1]["message"]
        response = self.data[-1]["convs"][-1]["response"]
        prompt = prompt.replace("{START_TIME}", message["start_time"])
        prompt = prompt.replace("{END_TIME}", message["end_time"])
        prompt = prompt.replace("{INPUT_IMAGE_SEQUENCE}", "<img>" * len(message["image_path"]))
        prompt = prompt.replace("{INPUT_AUDIO_STREAM}", "<audio>" if message["audio_path"] else "null")
        prompt = prompt.replace("{INPUT_TEXT_STREAM}", message.get("text", "null"))
        prompt = prompt.replace("{INPUT_FACES}", self.profile_tool.get_faces_profile(self.current_turn_faces))
        prompt = prompt.replace("{SHORT_TERM_MEMORY}", self.get_short_term_memory(img_ratio=0)[0])
        assistant_response = "null" if not response else response
        
        files = copy.deepcopy(message["image_path"] +[message["audio_path"]] if message["audio_path"] else message["image_path"])
        response = self.get_model_response(prompt, files, json_output=True, clear_context=True, log_tag="short_term_memory_update")
        response["start_time"] = message["start_time"]
        response["end_time"] = message["end_time"]
        response['assistant'] = assistant_response
        response["level"] = 0

        self.data[-1]["logs"] = self.data[-1].get("logs", [])
        response["turn_idx"] = len(self.data[-1]["logs"])
        self.data[-1]["logs"].append(copy.deepcopy(response))

        self.data[-1]["STM"] = self.data[-1].get("STM", [])
        self.data[-1]["STM"].append(copy.deepcopy(response))

        if "semantic_memory" in response:
            response["semantic_memory"] = list(set(response["semantic_memory"]))
            update_semantic_memorys = []
            for idx in range(len(response["semantic_memory"])):
                update_semantic_memorys.append(f"{response['start_time']}—{response['end_time']}: {response['semantic_memory'][idx]}")
            if update_semantic_memorys:
                self.retriever.update(semantic_list=update_semantic_memorys)

        txt = str(response.get('visual', '')) + str(response.get('auditory', ''))
        self.accumulated_faces_for_ltm.update({f"<face_{k}>" for k in re.findall(r'<face_(\d+)>', txt)})
        
        bottom_level_merged = self.summarize_short_term_memory()
        if bottom_level_merged:
            self.update_long_term_memory()
        

    def get_short_term_memory(self, img_ratio=0.1, start_idx=0, end_idx=None):
        STM = self.data[-1].get("STM", [])
        num = end_idx or len(STM)
        contents = []
        for item in STM[start_idx:num]:
            content = (
                f"Start time: {item['start_time']}——End time: {item['end_time']}\n"
                f"Visual description: {item.get('visual', 'null')}\n"
                f"Audio description: {item.get('auditory', 'null')}\n"
                f"Assistant: {item.get('assistant', 'null')}\n"
            )
            contents.append(content)
        if img_ratio <= 0:
            return "\n".join(contents), []

        stm_text = "\n".join(contents)
        LTM = self.data[-1].get("LTM", [])
        start_idx = max([max(t["turn_idx"]) for t in LTM if t.get("turn_idx")] or [-1]) + 1
        Convs = self.data[-1].get("convs", [])
        raw_image_pool = []
        for i in range(start_idx, len(Convs)):
            raw_image_pool.extend(Convs[i]["message"].get("image_path", []))
        stm_images = []
        if raw_image_pool:
            k = max(1, int(len(raw_image_pool) * img_ratio))
            stm_images = [raw_image_pool[int(i * len(raw_image_pool) / k)] for i in range(k)]
            stm_text += f"\n[Recent Visual History]: " + "<img>" * len(stm_images)

        return stm_text, stm_images


    def summarize_short_term_memory(self):
        f = self.merge_factor
        level_0_merged = False
        
        while True:
            stm = self.data[-1]["STM"]
            level_counts = Counter([node.get("level", 0) for node in stm])
            target_lvl = -1
            for lvl in sorted(level_counts.keys()):
                if level_counts[lvl] >= f + 1:
                    target_lvl = lvl
                    break
            
            if target_lvl == -1:
                break
            
            if target_lvl == 0:
                level_0_merged = True
            indices = [i for i, node in enumerate(stm) if node.get("level", 0) == target_lvl]
            to_merge_indices = indices[:f]
            start_idx = to_merge_indices[0]
            end_idx = to_merge_indices[-1] + 1
            memory_sequence, _ = self.get_short_term_memory(img_ratio=0, start_idx=start_idx, end_idx=end_idx)
            
            prompt = prompts_inf.OMNI_MEMORY_LOG_MERGE_PROMPT_INF
            prompt = prompt.replace("{MEMORY_LOG_SEQUENCE}", memory_sequence)
            response = self.get_model_response(prompt, [], json_output=True, clear_context=True, log_tag=f"short_term_memory_summary")
            
            subset = stm[start_idx : end_idx]
            new_node = copy.deepcopy(response)
            new_node.update({
                "start_time": subset[0]["start_time"],
                "end_time": subset[-1]["end_time"],
                "level": target_lvl + 1, # 等级进位
                "turn_idx": sum([x["turn_idx"] if isinstance(x["turn_idx"], list) else [x["turn_idx"]] for x in subset], [])
            })
            self.data[-1]["STM"] = stm[:start_idx] + [new_node] + stm[end_idx:]
        return level_0_merged

    def update_long_term_memory(self):
        N = len(self.data[-1]['logs'])
        if N == 0:
            return
        topics = self.data[-1].get("LTM", [])
        if topics:
            M = max([max(topic["turn_idx"]) for topic in topics]) + 1
        else:
            M = 0
        input_logs = copy.deepcopy(self.data[-1]['logs'][M:])
        if len(input_logs) == 0:
            return
        atomic_logs = []
        for i in range(len(input_logs)):
            item = input_logs[i]
            atomic_logs.append({
                "start_time": item["start_time"],
                "end_time": item["end_time"],
                "visual description": item.get("visual", "null"),
                "audio description": item.get("auditory", "null"),
                "assistant": item.get("assistant", "null"),
                "turn_idx": [item["turn_idx"]],
                "raw_convs": [self.data[-1]["convs"][item["turn_idx"]]]
            })
        topics = self.data[-1].get("LTM", [])
        self.data[-1]["LTM"] = topics + [{"turn_idx": i["turn_idx"]} for i in atomic_logs]
        if atomic_logs:
            self.retriever.update(episodic_list=atomic_logs)

        contents = []
        for idx, log in enumerate(input_logs):
            content = (
                f"Log index: {idx}\n"
                f"Start time: {log['start_time']}——End time: {log['end_time']}\n"
                f"Visual description: {log.get('visual', 'null')}\n"
                f"Audio description: {log.get('auditory', 'null')}\n"
                f"Assistant: {log['assistant']}\n"
            )
            contents.append(content)

        face_counts = Counter(re.findall(r'<face_(\d+)>', ''.join(contents)))
        faces_to_update = {f"<face_{k}>" for k, v in face_counts.items() if v >= 2}
        if faces_to_update:
            profile_prompt = prompts_inf.OMNI_MEMORY_PROFILE_UPDATE_PROMPT_INF
            profile_prompt = profile_prompt.replace("{CURRENT_PROFILES}", self.profile_tool.get_faces_profile(faces_to_update))
            profile_prompt = profile_prompt.replace("{MEMORY_LOG_SEQUENCE}", "\n".join(contents))
            profiles = self.get_model_response(profile_prompt, [], json_output=True, clear_context=True, log_tag="profile_update")
            self.profile_tool.update_faces_profile(profiles, faces_to_update)
        self.accumulated_faces_for_ltm = set()
        

    def send_message(self, message, update_memory=True):
        start_ts = message.get('start_time')
        last_conv_time = self.get_last_conv_time()
        if last_conv_time and utils.compare_time_strings(start_ts, last_conv_time) == -1:
            return ""
        if last_conv_time and utils.is_time_diff_over_minutes(start_ts, last_conv_time, self.session_interval_time):
            if update_memory:
                self.update_long_term_memory()
            self.data.append({"convs": []})
        response, retrieve_state = self.process_single_message(message)
        self.client.clear_media_files()
        update = True if message["tag"] != "eval_final_question" else False
        if update:
            self.data[-1]["convs"] = self.data[-1].get("convs", [])
            self.data[-1]["convs"].append({"message": message, "response": response, "retrieve_state": retrieve_state})
            if update_memory:
                self.memory_consolidation()
                self.update()
            self.current_turn_faces = set()
        self.client.clear_feature_cache()
        turn_data = {
            "message": message, 
            "response": response, 
            "retrieve_state": retrieve_state
        }
        return response, turn_data


    def memory_consolidation(self):
        self.update_short_term_memory()

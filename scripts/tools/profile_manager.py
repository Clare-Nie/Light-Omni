import os
import json

import faiss
import cv2
import numpy as np
from insightface.app import FaceAnalysis

from scripts import config


class ProfileTool:
    def __init__(self, root_path):
        self.face_config = config.FaceConfig
        self.root_path = root_path
        self.profile_dir = os.path.join(root_path, self.face_config.PROFILE_PATH)
        self.data_dir = os.path.join(root_path, self.face_config.PROFILE_DATA_PATH)
        self.avatar_dir = os.path.join(root_path, self.face_config.PROFILE_AVATAR_PATH)
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.avatar_dir, exist_ok=True)
        
        self.json_path = os.path.join(self.profile_dir, self.face_config.PROFILE_META_JSON)
        self.vectors_path = os.path.join(self.profile_dir, self.face_config.PROFILE_VECTORS_FILE)

        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                self.meta = json.load(f)
            if "qualities" not in self.meta:
                self.meta["qualities"] = [100.0] * len(self.meta["mapping"])
        else:
            self.meta = {"face_counter": 0, "mapping": [], "qualities": []}

        self.dimension = 512
        if os.path.exists(self.vectors_path):
            self.known_embeddings = np.load(self.vectors_path).tolist()
        else:
            self.known_embeddings = []
        self.index = faiss.IndexFlatIP(self.dimension)
        if len(self.known_embeddings) > 0:
            self.index.add(np.array(self.known_embeddings, dtype='float32'))
        print("Loading face detection model...")
        self.app = FaceAnalysis(name='buffalo_s', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']) # buffalo_s buffalo_l
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.active_tracks = {}

    def get_faces_profile(self, faces):
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                self.meta.update(json.load(f))
        contents = []
        for face in faces:
            if face in self.meta:
                contents.append(f"{face}: {str(self.meta[face])}")
            else:
                contents.append(f"{face}: {'unknown'}")
        return "\n".join(contents)

    def update_faces_profile(self, profiles, detected_faces):
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                disk_meta = json.load(f)
                self.meta.update(disk_meta)
        
        updated = False
        for face in detected_faces:
            if face in profiles:
                try:
                    new_data = profiles[face]
                    new_data = {k: v for k, v in new_data.items() if v != ""}
                    self.meta[face] = new_data
                    updated = True
                except:
                    pass
        if updated:
            self._save_data()


    def _save_data(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=4)
        embeddings_np = np.array(self.known_embeddings, dtype='float32')
        np.save(self.vectors_path, embeddings_np)
        self.index.reset()
        if len(self.known_embeddings) > 0:
            self.index.add(embeddings_np)

    def get_yaw_deviation(self, kps):
        """
        计算侧脸偏差值 (归一化算法)。
        返回值范围: 0.0 ~ 1.0
        0.0 : 绝对正脸
        0.1 ~ 0.2 : 轻微侧脸 (推荐注册阈值 < 0.2)
        > 0.5 : 大侧脸
        """
        if kps is None: return 1.0 # 没关键点视为最差情况
        left_eye_x = kps[0][0]
        right_eye_x = kps[1][0]
        nose_x = kps[2][0]

        dist_left = abs(nose_x - left_eye_x)
        dist_right = abs(right_eye_x - nose_x)
        
        numerator = abs(dist_left - dist_right)
        denominator = dist_left + dist_right
        
        if denominator == 0:
            return 1.0
        deviation = numerator / denominator
        return float(deviation)

    def get_confident_face(self, faces):
        valid_faces = []
        for f in faces:
            if f.det_score < 0.7: continue 
            bbox = f.bbox.astype(int)
            if (bbox[3] - bbox[1]) < 75: continue 
            valid_faces.append(f)
        return valid_faces
    def get_confident_face(self, faces):
        valid_faces = []
        for f in faces:
            if f.det_score < 0.7: continue 
            
            bbox = f.bbox.astype(int)
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            if face_height < 80: continue 

            yaw_dev = self.get_yaw_deviation(f.kps)
            left_eye = f.kps[0]
            right_eye = f.kps[1]
            eye_dist = np.linalg.norm(left_eye - right_eye)
            eye_ratio = eye_dist / face_width
            if eye_ratio < 0.20: 
                continue
            f.yaw_dev = float(yaw_dev)
            f.eye_ratio = float(eye_ratio)
            valid_faces.append(f)
            
        return valid_faces

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def process_image(self, timestamp, image_path, long_edge=None):
        return_img_array = False
        if isinstance(image_path, str):
            if not os.path.exists(image_path): return None, []
            frame = cv2.imread(image_path)
        else:
            frame = image_path
            return_img_array = True
            
        if frame is None: return None, []

        detected_names = []
        faces = self.app.get(frame)
        valid_faces = self.get_confident_face(faces)
        
        next_active_tracks = {}
        db_updated = False
        
        TRACK_THRESH = 0.5
        DB_THRESH = 0.5
        UPDATE_ALPHA = 0.2

        for face in valid_faces:
            current_emb = face.embedding.astype('float32').reshape(1, -1)
            faiss.normalize_L2(current_emb)
            
            current_yaw_dev = self.get_yaw_deviation(face.kps)
            found_name = "Unknown"
            best_score = 0
            found_idx = -1 # 记录是在 mapping 里的第几个
            
            # === Step A: 查短期记忆 (Track) ===
            for name, hist_emb in self.active_tracks.items():
                score = np.dot(current_emb, hist_emb.T)[0][0]
                if score > TRACK_THRESH and score > best_score:
                    best_score = score
                    found_name = name
                    # 这里我们需要反查 ID 索引，稍微有点慢，但数据量小没问题
                    if name in self.meta["mapping"]:
                        found_idx = self.meta["mapping"].index(name)
            
            # === Step B: 查全局数据库 (DB) ===
            if found_name == "Unknown":
                if len(self.known_embeddings) > 0:
                    D, I = self.index.search(current_emb, 1)
                    score = D[0][0]
                    idx = I[0][0]
                    if score > DB_THRESH:
                        if 0 <= idx < len(self.meta["mapping"]):
                            found_name = self.meta["mapping"][idx]
                            found_idx = idx
            
            # === Step C: 注册新用户 ===
            if found_name == "Unknown":
                # 偏差 < 0.5 大概对应 Yaw 0.5~1.5，算比较正
                if current_yaw_dev < 0.5:
                    found_name = self._register_new_face(current_emb, frame, face.bbox, current_yaw_dev)
                    db_updated = True
                    found_idx = len(self.known_embeddings) - 1

            # === Step D: 择优更新数据库
            if found_idx != -1:
                if found_idx < len(self.meta["qualities"]) and found_idx < len(self.known_embeddings):
                    old_quality = self.meta["qualities"][found_idx]
                    if current_yaw_dev < (old_quality - 0.1):
                        print(f"Update Best Shot for {found_name}: Quality {old_quality:.2f} -> {current_yaw_dev:.2f}")
                        self.known_embeddings[found_idx] = current_emb.flatten()
                        self.meta["qualities"][found_idx] = float(current_yaw_dev)
                        self._save_avatar(found_name, frame, face.bbox)
                        db_updated = True
                else:
                    pass

            # === Step E: 更新短期记忆
            if found_name != "Unknown":
                if found_name in self.active_tracks:
                    history_emb = self.active_tracks[found_name]
                    new_emb = (UPDATE_ALPHA * current_emb) + ((1 - UPDATE_ALPHA) * history_emb)
                    faiss.normalize_L2(new_emb)
                    next_active_tracks[found_name] = new_emb
                else:
                    next_active_tracks[found_name] = current_emb

                detected_names.append(found_name)
                h, w = frame.shape[:2]
                f_scale = (w / 1000.0) * 1.5
                f_thick = max(1, int((w / 1000.0) * 4))
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), f_thick)
                cv2.putText(frame, found_name, (bbox[0], bbox[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, f_scale, (0, 255, 0), f_thick)

        self.active_tracks = next_active_tracks
        if db_updated:
            self._save_data()

        if long_edge is not None:
            h, w = frame.shape[:2]
            if max(h, w) > long_edge:
                scale = long_edge / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        save_path = os.path.join(self.data_dir, f"processed_{timestamp}.png")
        cv2.imwrite(save_path, frame)
        if return_img_array:
            return frame, detected_names
        return save_path, detected_names

    def _register_new_face(self, embedding_np, original_img, bbox, yaw_dev):
        current_id_num = self.meta["face_counter"]
        name = f"<face_{current_id_num}>"
        self.meta["mapping"].append(name)
        self.meta["qualities"].append(float(yaw_dev))
        self.meta["face_counter"] += 1
        self.known_embeddings.append(embedding_np.flatten())
        self._save_avatar(name, original_img, bbox)
        return name
    def _save_avatar(self, name, img, bbox):
        try:
            x1, y1, x2, y2 = bbox.astype(int)
            h, w, _ = img.shape
            x1 = max(0, x1 - 10); y1 = max(0, y1 - 10)
            x2 = min(w, x2 + 10); y2 = min(h, y2 + 10)
            face_crop = img[y1:y2, x1:x2]
            if face_crop.size > 0:
                avatar_path = os.path.join(self.avatar_dir, f"{name}.png")
                cv2.imwrite(avatar_path, face_crop)
        except: pass
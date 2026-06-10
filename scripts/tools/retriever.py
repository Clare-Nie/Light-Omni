import os
import json

import faiss


class Retriever:
    def __init__(self, root_path, dimension=1024, omni_model=None):
        self.root_path = root_path
        self.storage_dir = os.path.join(root_path, "retriever_storage")
        
        self.sem_dir = os.path.join(self.storage_dir, "semantic")
        self.sem_data_path = os.path.join(self.sem_dir, "data.json")
        self.sem_index_path = os.path.join(self.sem_dir, "vector.index")
        
        self.episodic_dir = os.path.join(self.storage_dir, "episodic")
        self.episodic_data_path = os.path.join(self.episodic_dir, "data.json")
        self.episodic_index_path = os.path.join(self.episodic_dir, "vector.index")

        self.dimension = dimension

        os.makedirs(self.sem_dir, exist_ok=True)
        os.makedirs(self.episodic_dir, exist_ok=True)

        self.sem_data, self.sem_index = self._load_memory(self.sem_data_path, self.sem_index_path)
        self.episodic_data, self.episodic_index = self._load_memory(self.episodic_data_path, self.episodic_index_path)

        self.seen_sem_indices = set()
        self.seen_episodic_indices = set()

        self.omni_model = omni_model

        self.reload()
    
    def reload(self):
        self.sem_data, self.sem_index = self._load_memory(self.sem_data_path, self.sem_index_path)
        self.episodic_data, self.episodic_index = self._load_memory(self.episodic_data_path, self.episodic_index_path)
        self.reset()
    
    def reset(self):
        self.seen_sem_indices.clear()
        self.seen_episodic_indices.clear()
    
    def _get_text_embedding(self, texts):
        embeddings = self.omni_model.get_texts_embedding(texts)
        embeddings = embeddings.float().cpu().numpy()
        return embeddings

    def _load_memory(self, json_path, index_path):
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
        else:
            index = faiss.IndexFlatIP(self.dimension)
        return data, index
    
    def _save_memory(self, data, index, json_path, index_path):
        """内部方法：保存特定的记忆库"""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        faiss.write_index(index, index_path)
    
    def _add_to_index(self, new_items, current_data, current_index, key=None):
        if not new_items:
            return False
        
        existing_ids = set()
        for x in current_data:
            if isinstance(x, dict) and "start_time" in x and "end_time" in x:
                existing_ids.add(f"{x['start_time']}_{x['end_time']}")
            elif isinstance(x, str):
                existing_ids.add(x.strip())
        
        texts_to_embed = []
        valid_items_to_add = []
        for item in new_items:
            if isinstance(item, dict) and "start_time" in item:
                if f"{item['start_time']}_{item['end_time']}" in existing_ids:
                    continue
            elif isinstance(item, str):
                if item.strip() in existing_ids:
                    continue
            
            text_to_embed = ""
            if key is not None and isinstance(item, dict):
                _key = key if isinstance(key, list) else [key]
                # 过滤掉 None 或空值，确保全是字符串
                parts = [str(item.get(dd, "")) for dd in _key if item.get(dd) is not None]
                text_to_embed = "\n".join(parts)
            elif isinstance(item, str):
                text_to_embed = item

            if text_to_embed and text_to_embed.strip():
                texts_to_embed.append(text_to_embed)
                valid_items_to_add.append(item)
        
        if not texts_to_embed:
            return False

        batch_emb = self._get_text_embedding(texts_to_embed)
        if batch_emb is None or len(batch_emb) == 0:
            return False

        current_index.add(batch_emb)
        current_data.extend(valid_items_to_add)
        return True
        
    def update(self, semantic_list=None, episodic_list=None):
        sem_updated = False
        episodic_updated = False

        if semantic_list and len(semantic_list) > 0:
            if self._add_to_index(semantic_list, self.sem_data, self.sem_index, key=None):
                self._save_memory(self.sem_data, self.sem_index, self.sem_data_path, self.sem_index_path)
                sem_updated = True
                # print(f"[Retriever] Added {len(semantic_list)} semantic memories.")

        if episodic_list and len(episodic_list) > 0:
            if self._add_to_index(episodic_list, self.episodic_data, self.episodic_index, key=["visual description", "audio description"]):
                self._save_memory(self.episodic_data, self.episodic_index, self.episodic_data_path, self.episodic_index_path)
                episodic_updated = True
                # print(f"[Retriever] Added {len(episodic_list)} episodic memories.")

        if not sem_updated and not episodic_updated:
            print("[Retriever] No valid data to update.")

    def retrieve(self, query, top_k_semantic=6, top_k_episodic=2):
        if isinstance(query, str):
            query_emb = self._get_text_embedding(query)
        else:
            query_emb = query

        def search_unique(index, data, k, seen_indices):
            total_items = index.ntotal
            if total_items == 0:
                return []
            
            fetch_k = min(total_items, k + len(seen_indices) + 5)
            
            D, I = index.search(query_emb, fetch_k)
            
            results = []
            for score, idx in zip(D[0], I[0]):
                if len(results) >= k:
                    break
                if idx != -1 and idx < len(data) and idx not in seen_indices:
                    results.append({
                        "id": int(idx),
                        "content": data[idx],
                        "score": float(score)
                    })
                    seen_indices.add(idx)
            return results
        
        sem_results = search_unique(self.sem_index, self.sem_data, top_k_semantic, self.seen_sem_indices)
        episodic_results = search_unique(self.episodic_index, self.episodic_data, top_k_episodic, self.seen_episodic_indices)

        sem_results.sort(key=lambda x: x['id'])
        episodic_results.sort(key=lambda x: x['id'])

        return {
            "semantic": sem_results,
            "episodic": episodic_results
        }
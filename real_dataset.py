"""
Real Text Dataset: Headlines for Sports, Tech, Politics, and Health (Novelty).
Uses Sentence-Transformers to generate embeddings.
"""

import torch
import os
import pickle
from sentence_transformers import SentenceTransformer

# Categorias
SPORTS = [
    "The team won the championship after a thrilling final match.",
    "Golden State Warriors secure victory in the NBA playoffs.",
    "Real Madrid celebrates another Champions League title.",
    "The Olympic Games will feature new sports this year.",
    "Tennis world number one retires after a record-breaking career.",
    "Formula 1 race ends with a dramatic overtake on the last lap.",
    "World Cup qualifying matches are heating up across the globe.",
    "The marathon runner broke the world record in Berlin.",
    "Basketball stars sign massive contracts during free agency.",
    "The local soccer team was promoted to the first division."
]

TECH = [
    "New AI model released with unprecedented reasoning capabilities.",
    "Smartphone manufacturers reveal the latest foldable screen technology.",
    "GPU architecture improvements double performance for gaming and AI.",
    "The software update fixes critical security vulnerabilities.",
    "Startup launches a new satellite into low Earth orbit.",
    "Cloud computing provider announces a massive expansion of data centers.",
    "Researchers develop a more efficient battery for electric vehicles.",
    "The tech giant acquires a small VR gaming studio.",
    "New programming language promises better memory safety.",
    "Quantum computing prototype achieves a major milestone."
]

POLITICS = [
    "The senate voted on the new bill after days of debate.",
    "Diplomatic relations strained between the two neighboring countries.",
    "Elections are scheduled for the end of the year in the region.",
    "The president announced a new economic policy to fight inflation.",
    "International summit focuses on climate change agreements.",
    "The prime minister reshuffles the cabinet ahead of the session.",
    "Protests erupt in the capital following the government's decision.",
    "Trade agreement signed to reduce tariffs between major economies.",
    "The mayor outlines a new plan for urban infrastructure development.",
    "Peace talks continue as both sides seek a permanent ceasefire."
]

HEALTH_NOVELTY = [
    "A new vaccine was discovered to prevent a rare tropical disease.",
    "Healthy diet and regular exercise reduce the risk of heart disease.",
    "Researchers identify a new genetic marker for early cancer detection.",
    "Mental health awareness month promotes wellness and support.",
    "The hospital opens a new wing for advanced surgical procedures.",
    "New study reveals the benefits of sleep for cognitive function.",
    "Global health organization declares the outbreak over.",
    "Advances in telemedicine improve access for remote communities.",
    "The nutrition guide highlights the importance of vitamins and minerals.",
    "Doctors successfully perform a complex organ transplant surgery."
]

class TextDataset:
    def __init__(self, model_name='all-MiniLM-L6-v2', cache_file='embeddings_cache.pkl'):
        self.model = SentenceTransformer(model_name)
        self.cache_file = cache_file
        
    def get_data(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Generar embeddings
        console_log("Encoding sentences (this may take a few seconds)...")
        emb_sports = self.model.encode(SPORTS)
        emb_tech = self.model.encode(TECH)
        emb_politics = self.model.encode(POLITICS)
        emb_health = self.model.encode(HEALTH_NOVELTY)
        
        data = {
            'train_x': torch.tensor([*emb_sports, *emb_tech, *emb_politics]),
            'train_y': torch.tensor([0]*len(SPORTS) + [1]*len(TECH) + [2]*len(POLITICS)),
            'novel_x': torch.tensor(emb_health),
            'centers': torch.tensor([emb_sports.mean(0), emb_tech.mean(0), emb_politics.mean(0)])
        }
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump(data, f)
            
        return data

def console_log(msg):
    # Helper simple si rich no esta disponible o para evitar problemas de importacion
    print(f"[TextDataset] {msg}")

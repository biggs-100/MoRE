import torch
from benchmark.metrics import MetricEngine

def run_experiment(model, stream, feature_extractor, n_tasks):
    engine = MetricEngine(n_tasks=n_tasks)
    
    # Lista de tareas para evaluación posterior (i < j)
    # En realidad, necesitamos evaluar TODAS las tareas vistas hasta ahora (j <= i)
    task_data_cache = []
    
    for i, (X, y, tid) in enumerate(stream):
        print(f"--- Training on Task {i} ---")
        
        # Extraer features
        features = feature_extractor.extract(X)
        
        # Entrenar modelo
        if hasattr(model, 'train_task'):
            model.train_task(features, y, task_id=i)
        
        # Consolidar (para EWC)
        if hasattr(model, 'consolidate'):
            model.consolidate(features, y)
            
        # Cachear para evaluación
        task_data_cache.append((features, y, tid))
        
        # Evaluar en todas las tareas vistas hasta i
        for j in range(i + 1):
            eval_features, eval_y, _ = task_data_cache[j]
            
            output = model.predict(eval_features)
            preds = torch.argmax(output, dim=1)
            acc = (preds == eval_y).float().mean().item()
            
            engine.update(i, j, acc)
            print(f"Task {i} -> Accuracy on Task {j}: {acc:.4f}")
            
    return engine

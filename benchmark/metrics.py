import numpy as np

class MetricEngine:
    def __init__(self, n_tasks):
        self.n_tasks = n_tasks
        self.R = np.zeros((n_tasks, n_tasks))
        
    def update(self, current_task_id, eval_task_id, accuracy):
        self.R[current_task_id, eval_task_id] = accuracy
        
    def calculate_acc(self):
        """Average Accuracy over all seen tasks at the end."""
        T = self.n_tasks
        return np.mean(self.R[T-1, :T])
        
    def calculate_bwt(self):
        """
        Backward Transfer: measures the average change in accuracy 
        on task i after learning tasks i+1 to T.
        BWT = 1/(T-1) * sum_{i=1}^{T-1} (R[T,i] - R[i,i])
        """
        T = self.n_tasks
        if T <= 1:
            return 0.0
            
        forgetting = 0.0
        for i in range(T - 1):
            forgetting += (self.R[T-1, i] - self.R[i, i])
            
        return forgetting / (T - 1)
        
    def get_matrix(self):
        return self.R.tolist()

import os
from typing import List, Iterable

from unify_eval.model.deep_model import DeepModel

"""
Maintains a queue of a given size of paths to saved models. 
"""


class QueuedModelSaver:
    def __init__(self,
                 path_to_folder: str,
                 model_name: str,
                 queue_size: int = 3):
        self.path_to_folder = path_to_folder
        if not os.path.exists(self.path_to_folder):
            print(f"creating new model folder {self.path_to_folder}")
            os.makedirs(self.path_to_folder)
        self.model_name = model_name
        self.paths_to_models: List[str] = []
        self.queue_size: int = queue_size

    def get_latest_model_path(self) -> str:
        return self.paths_to_models[-1]

    def _remove_last_model(self):
        path = self.paths_to_models.pop(0)
        os.remove(path)

    def add_model(self, model: DeepModel, tags: Iterable[str]):
        path = self._make_full_model_path(tags)
        self.paths_to_models.append(path)
        while len(self.paths_to_models) > self.queue_size:
            self._remove_last_model()
        model.to_cpu().save(path=self._make_full_model_path(tags))

    def _make_full_model_path(self, tags: Iterable[str]):
        tags = "_".join(str(tag) for tag in tags)
        return os.path.join(self.path_to_folder, f"{self.model_name}_{tags}.model")

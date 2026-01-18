import json
from typing import Any

class JSONConfig:

    def __init__(self, file_path:str):
        self.file_path = file_path
        self.data = self._load_json()

    def _load_json(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save(self)->None:
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def get(self, key:str, default=None) -> Any:
        return self.data.get(key, default)

    def set(self, key:str, value:Any)->None:
        self.data[key] = value
        self.save()

    def delete(self, key:str)->None:
        if key in self.data:
            del self.data[key]
            self.save()
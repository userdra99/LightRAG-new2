import os
from pathlib import Path
class LightRAGConfig:
    WORKING_DIR = "./data"
    DOCSSOURCE_PATH = "./DOCSSOURCE"
    EMBEDDING_DIM = 768
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 100
    SUPPORTED_FORMATS = [".pdf", ".txt", ".docx", ".xlsx", ".csv"]
    
    @classmethod
    def get_working_dir(cls):
        return Path(cls.WORKING_DIR).absolute()
    
    @classmethod
    def get_docssource_path(cls):
        return Path(cls.DOCSSOURCE_PATH).absolute()
    
    @classmethod
    def ensure_directories(cls):
        cls.get_working_dir().mkdir(parents=True, exist_ok=True)
        cls.get_docssource_path().mkdir(parents=True, exist_ok=True)
        Path("./logs").mkdir(exist_ok=True)
    
    @classmethod
    def is_supported_file(cls, file_path):
        return any(file_path.lower().endswith(ext) for ext in cls.SUPPORTED_FORMATS)

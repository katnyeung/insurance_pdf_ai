from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    # RAGFlow configuration
    RAGFLOW_API_URL: str = os.environ.get("RAGFLOW_API_URL", "http://localhost:9380")
    RAGFLOW_API_KEY: str = os.environ.get("RAGFLOW_API_KEY", "ragflow-I4NTJhODhlMGFhZDExZjBiNzE2MzI1ZT")
    
    # App configuration
    APP_HOST: str = os.environ.get("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.environ.get("APP_PORT", "5000"))
    APP_DEBUG: bool = os.environ.get("APP_DEBUG", "True").lower() == "true"
    
    # LLM configuration
    LLM_MODEL: str = os.environ.get("LLM_MODEL", "deepseek-r1:latest")
    LLM_API_BASE: str = os.environ.get("LLM_API_BASE", "http://localhost:11434")
    
    # Path configuration
    TEMPLATES_DIR: str = os.environ.get("TEMPLATES_DIR", "templates")
    STATIC_DIR: str = os.environ.get("STATIC_DIR", "static")
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
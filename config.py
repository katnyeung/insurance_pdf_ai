from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    RAGFLOW_API_URL: str = "http://localhost:9380"
    RAGFLOW_API_KEY: str = "ragflow-I4NTJhODhlMGFhZDExZjBiNzE2MzI1ZT"

settings = Settings()

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    openai_api_key: str

    model_config = SettingsConfigDict(
        env_file=(".env"),
        env_nested_delimiter="__",
        env_file_encoding="utf-8",
        extra="ignore",
    )


config = Config()  # type: ignore

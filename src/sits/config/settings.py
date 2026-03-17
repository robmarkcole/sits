"""
Configurações globais do sistema.
"""

from functools import lru_cache

import torch
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Configurações globais do SITS.

    Valores podem ser sobrescritos por variáveis de ambiente com prefixo SITS_.
    Exemplo: SITS_LOG_LEVEL=DEBUG
    """

    # Device
    device: str = Field(
        default="auto",
        description="Device para PyTorch: 'auto', 'cuda' ou 'cpu'"
    )

    # Defaults
    default_batch_size: int = Field(
        default=4096,
        description="Tamanho de batch padrão"
    )

    random_seed: int = Field(
        default=42,
        description="Seed para reprodutibilidade"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Nível de logging: DEBUG, INFO, WARNING, ERROR"
    )

    # Processamento
    num_workers: int = Field(
        default=0,
        description="Workers para DataLoader (0 = main thread)"
    )

    class Config:
        env_prefix = "SITS_"
        env_file = ".env"

    def get_device(self) -> torch.device:
        """
        Retorna torch.device baseado na configuração.

        Returns:
            torch.device configurado
        """
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


@lru_cache()
def get_settings() -> Settings:
    """
    Retorna instância singleton das configurações.

    Returns:
        Settings configurado
    """
    return Settings()

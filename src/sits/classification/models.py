"""
Registry de modelos para classificacao de series temporais.

Usa modelos do tsai com interface unificada.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from loguru import logger

# Registry de modelos tsai
# Formato: (nome_tsai, aceita_seq_len)
TSAI_MODELS = {
    # Inception family
    "inception_time": ("InceptionTime", True),
    "inception_time_plus": ("InceptionTimePlus", True),
    # ResNet
    "resnet": ("ResNet", False),
    "resnet_plus": ("ResNetPlus", True),
    # LSTM
    "lstm": ("LSTM", False),
    "lstm_plus": ("LSTMPlus", True),
    "lstm_attention": ("LSTMAttention", True),
    # GRU
    "gru": ("GRU", False),
    "gru_plus": ("GRUPlus", True),
    "gru_attention": ("GRUAttention", True),
    # RNN
    "rnn": ("RNN", False),
    "rnn_plus": ("RNNPlus", True),
    "rnn_attention": ("RNNAttention", True),
    # FCN
    "fcn": ("FCN", False),
    "fcn_plus": ("FCNPlus", False),
    # TCN
    "tcn": ("TCN", False),
    # Transformers
    "tst": ("TST", True),
    "tst_plus": ("TSTPlus", True),
    "tsit_plus": ("TSiTPlus", True),
    "conv_tran_plus": ("ConvTranPlus", True),
    # Perceiver
    "ts_perceiver": ("TSPerceiver", True),
    # Hybrid Transformer-RNN
    "transformer_gru_plus": ("TransformerGRUPlus", True),
    "transformer_lstm_plus": ("TransformerLSTMPlus", True),
    "transformer_rnn_plus": ("TransformerRNNPlus", True),
    # XCM
    "xcm": ("XCM", True),
    "xcm_plus": ("XCMPlus", True),
    # Xception
    "xception_time": ("XceptionTime", False),
    "xception_time_plus": ("XceptionTimePlus", True),
    # Hybrid CNN+RNN
    "lstm_fcn": ("LSTM_FCN", True),
    "gru_fcn": ("GRU_FCN", True),
    "mlstm_fcn": ("MLSTM_FCN", True),
    "lstm_fcn_plus": ("LSTM_FCNPlus", True),
    "gru_fcn_plus": ("GRU_FCNPlus", True),
    "mlstm_fcn_plus": ("MLSTM_FCNPlus", True),
    # Others
    "mwdn": ("mWDN", True),
    "omniscale_cnn": ("OmniScaleCNN", True),
    "rescnn": ("ResCNN", False),
}


def get_available_models() -> list:
    """
    Retorna lista de modelos disponiveis.

    Returns:
        Lista de nomes de modelos
    """
    return list(TSAI_MODELS.keys())


def build_model(
    model_name: str,
    c_in: int,
    c_out: int,
    seq_len: int,
    **kwargs,
) -> nn.Module:
    """
    Constroi modelo do registry tsai.

    Args:
        model_name: Nome do modelo (ex: "inception_time")
        c_in: Numero de canais de entrada (features)
        c_out: Numero de classes
        seq_len: Comprimento da sequencia temporal
        **kwargs: Argumentos extras para o modelo

    Returns:
        Modelo PyTorch

    Raises:
        ValueError: Se modelo nao encontrado
    """
    if model_name not in TSAI_MODELS:
        available = ", ".join(get_available_models())
        raise ValueError(
            f"Modelo '{model_name}' nao encontrado. "
            f"Disponiveis: {available}"
        )

    try:
        from tsai.models.all import (
            InceptionTime, InceptionTimePlus,
            ResNet, ResNetPlus,
            LSTM, LSTMPlus, LSTMAttention,
            GRU, GRUPlus, GRUAttention,
            RNN, RNNPlus, RNNAttention,
            FCN, FCNPlus,
            TCN, TST, TSTPlus,
            TSiTPlus, ConvTranPlus,
            TSPerceiver,
            TransformerGRUPlus, TransformerLSTMPlus, TransformerRNNPlus,
            XCM, XCMPlus,
            XceptionTime, XceptionTimePlus,
            LSTM_FCN, GRU_FCN, MLSTM_FCN,
            LSTM_FCNPlus, GRU_FCNPlus, MLSTM_FCNPlus,
            mWDN, OmniScaleCNN, ResCNN,
        )

        model_registry = {
            "InceptionTime": InceptionTime,
            "InceptionTimePlus": InceptionTimePlus,
            "ResNet": ResNet,
            "ResNetPlus": ResNetPlus,
            "LSTM": LSTM,
            "LSTMPlus": LSTMPlus,
            "LSTMAttention": LSTMAttention,
            "GRU": GRU,
            "GRUPlus": GRUPlus,
            "GRUAttention": GRUAttention,
            "RNN": RNN,
            "RNNPlus": RNNPlus,
            "RNNAttention": RNNAttention,
            "FCN": FCN,
            "FCNPlus": FCNPlus,
            "TCN": TCN,
            "TST": TST,
            "TSTPlus": TSTPlus,
            "TSiTPlus": TSiTPlus,
            "ConvTranPlus": ConvTranPlus,
            "TSPerceiver": TSPerceiver,
            "TransformerGRUPlus": TransformerGRUPlus,
            "TransformerLSTMPlus": TransformerLSTMPlus,
            "TransformerRNNPlus": TransformerRNNPlus,
            "XCM": XCM,
            "XCMPlus": XCMPlus,
            "XceptionTime": XceptionTime,
            "XceptionTimePlus": XceptionTimePlus,
            "LSTM_FCN": LSTM_FCN,
            "GRU_FCN": GRU_FCN,
            "MLSTM_FCN": MLSTM_FCN,
            "LSTM_FCNPlus": LSTM_FCNPlus,
            "GRU_FCNPlus": GRU_FCNPlus,
            "MLSTM_FCNPlus": MLSTM_FCNPlus,
            "mWDN": mWDN,
            "OmniScaleCNN": OmniScaleCNN,
            "ResCNN": ResCNN,
        }

        class_name, uses_seq_len = TSAI_MODELS[model_name]
        model_class = model_registry[class_name]

        # Build kwargs based on whether model accepts seq_len
        if uses_seq_len:
            model = model_class(c_in=c_in, c_out=c_out, seq_len=seq_len, **kwargs)
        else:
            model = model_class(c_in=c_in, c_out=c_out, **kwargs)

        logger.info(
            f"Modelo criado: {model_name} "
            f"(c_in={c_in}, c_out={c_out}, seq_len={seq_len})"
        )

        return model

    except ImportError as e:
        raise ImportError(
            f"tsai nao instalado. Instale com: pip install tsai\n"
            f"Erro: {e}"
        )


def load_trained_model(
    model_dir: str,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Carrega modelo treinado de um diretorio.

    Args:
        model_dir: Diretorio com model.pt ou best_model.pth e config
        device: Dispositivo (cuda/cpu)

    Returns:
        Tupla (modelo, config)
    """
    import json

    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_dir = Path(model_dir)

    # Try different model file names
    model_path = None
    for fname in ["best_model.pth", "model.pt", "model.pth"]:
        if (model_dir / fname).exists():
            model_path = model_dir / fname
            break

    if model_path is None:
        raise FileNotFoundError(f"Modelo nao encontrado em: {model_dir}")

    # Try different config file names
    config = {}
    for cfg_name in ["model_config.json", "config.json"]:
        cfg_path = model_dir / cfg_name
        if cfg_path.exists():
            with open(cfg_path) as f:
                config = json.load(f)
            break

    if not config:
        # Try to extract from checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            config = checkpoint.get("config", {})

    # Create model
    model = build_model(
        model_name=config.get("model_name", "inception_time"),
        c_in=config["c_in"],
        c_out=config["c_out"],
        seq_len=config["seq_len"],
    )

    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    logger.info(f"Modelo carregado: {model_dir}")

    return model, config


def save_model(
    model: nn.Module,
    config: Dict[str, Any],
    save_dir: str,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Salva modelo treinado.

    Args:
        model: Modelo PyTorch
        config: Configuracao do modelo
        save_dir: Diretorio de destino
        metrics: Metricas opcionais
    """
    import json

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Salvar checkpoint
    checkpoint = {
        "model_state": model.state_dict(),
        "config": config,
        "metrics": metrics or {},
    }
    torch.save(checkpoint, save_dir / "model.pt")

    # Salvar config separado para facilitar leitura
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Salvar metricas se houver
    if metrics:
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    logger.info(f"Modelo salvo: {save_dir}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Conta parametros do modelo.

    Args:
        model: Modelo PyTorch

    Returns:
        Dict com contagem de parametros
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }

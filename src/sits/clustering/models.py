"""
Arquiteturas de redes neurais para clustering de series temporais.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class LSTMEncoder(nn.Module):
    """
    Encoder LSTM bidirecional.

    Args:
        input_dim: Dimensao de entrada (features por timestep)
        hidden_dim: Dimensao das camadas ocultas
        latent_dim: Dimensao do espaco latente
        n_layers: Numero de camadas LSTM
        dropout: Taxa de dropout
        bidirectional: Se True, usa LSTM bidirecional
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        latent_dim: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Projecao para espaco latente
        self.fc = nn.Linear(hidden_dim * self.num_directions, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor (batch, seq_len, input_dim)

        Returns:
            Embedding (batch, latent_dim)
        """
        # LSTM output
        _, (h_n, _) = self.lstm(x)

        # Concatenar hidden states das duas direcoes
        if self.bidirectional:
            h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_n = h_n[-1]

        # Projetar para espaco latente
        embedding = self.fc(h_n)

        return embedding


class LSTMDecoder(nn.Module):
    """
    Decoder LSTM para reconstrucao da serie temporal.

    Args:
        latent_dim: Dimensao do espaco latente
        hidden_dim: Dimensao das camadas ocultas
        output_dim: Dimensao de saida (features por timestep)
        seq_len: Comprimento da sequencia
        n_layers: Numero de camadas LSTM
        dropout: Taxa de dropout
    """

    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        output_dim: int = 1,
        seq_len: int = 12,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.n_layers = n_layers

        # Projecao do espaco latente
        self.fc_in = nn.Linear(latent_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        # Projecao para saida
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z: Embedding (batch, latent_dim)

        Returns:
            Reconstrucao (batch, seq_len, output_dim)
        """
        batch_size = z.size(0)

        # Projetar embedding
        h = self.fc_in(z)

        # Repetir para cada timestep
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1)

        # Decodificar
        output, _ = self.lstm(h)

        # Projetar para dimensao de saida
        reconstruction = self.fc_out(output)

        return reconstruction


class DTCAutoencoder(nn.Module):
    """
    Deep Temporal Clustering Autoencoder.

    Combina encoder e decoder LSTM para aprendizado de representacoes
    de series temporais.

    Args:
        input_dim: Dimensao de entrada (features por timestep)
        hidden_dim: Dimensao das camadas ocultas
        latent_dim: Dimensao do espaco latente
        seq_len: Comprimento da sequencia
        n_layers: Numero de camadas LSTM
        dropout: Taxa de dropout
        bidirectional: Se True, usa LSTM bidirecional no encoder
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        latent_dim: int = 8,
        seq_len: int = 12,
        n_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.decoder = LSTMDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            seq_len=seq_len,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.latent_dim = latent_dim
        self.seq_len = seq_len

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Tensor (batch, seq_len, input_dim)

        Returns:
            Tupla (reconstruction, embedding)
        """
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apenas codifica a entrada.

        Args:
            x: Tensor (batch, seq_len, input_dim)

        Returns:
            Embedding (batch, latent_dim)
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apenas decodifica do espaco latente.

        Args:
            z: Embedding (batch, latent_dim)

        Returns:
            Reconstrucao (batch, seq_len, output_dim)
        """
        return self.decoder(z)


class ClusteringLayer(nn.Module):
    """
    Camada de clustering com distribuicao t-Student.

    Calcula a probabilidade de cada amostra pertencer a cada cluster
    usando a distribuicao t-Student (como no DEC/DTC).

    Q_ij = (1 + ||z_i - mu_j||^2 / alpha)^(-(alpha+1)/2)
           / sum_j'(1 + ||z_i - mu_j'||^2 / alpha)^(-(alpha+1)/2)

    Args:
        n_clusters: Numero de clusters
        latent_dim: Dimensao do espaco latente
        alpha: Graus de liberdade da t-Student (default=1.0)
    """

    def __init__(
        self,
        n_clusters: int,
        latent_dim: int,
        alpha: float = 1.0,
    ):
        super().__init__()

        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.alpha = alpha

        # Centroides dos clusters (parametros aprendiveis)
        self.clusters = nn.Parameter(
            torch.randn(n_clusters, latent_dim) * 0.01
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Calcula probabilidades de cluster (distribuicao Q).

        Args:
            z: Embeddings (batch, latent_dim)

        Returns:
            Probabilidades (batch, n_clusters)
        """
        # Distancia quadrada aos centroides
        # (batch, 1, latent_dim) - (1, n_clusters, latent_dim)
        diff = z.unsqueeze(1) - self.clusters.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)

        # Distribuicao t-Student
        q = (1 + dist_sq / self.alpha) ** (-(self.alpha + 1) / 2)
        q = q / q.sum(dim=1, keepdim=True)

        return q

    def init_centroids(self, centroids: torch.Tensor) -> None:
        """
        Inicializa centroides com valores pre-computados (ex: KMeans).

        Args:
            centroids: Tensor (n_clusters, latent_dim)
        """
        with torch.no_grad():
            self.clusters.copy_(centroids)

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        """
        Calcula distribuicao alvo P para treinamento.

        P_ij = Q_ij^2 / sum_i(Q_ij)
               / sum_j'(Q_ij'^2 / sum_i(Q_ij'))

        Args:
            q: Probabilidades Q (batch, n_clusters)

        Returns:
            Distribuicao alvo P (batch, n_clusters)
        """
        weight = q ** 2 / q.sum(dim=0, keepdim=True)
        p = weight / weight.sum(dim=1, keepdim=True)
        return p


class TemporalAttention(nn.Module):
    """
    Mecanismo de atencao temporal.

    Calcula pesos de atencao sobre a sequencia temporal para
    interpretabilidade.

    Args:
        hidden_dim: Dimensao do hidden state
        attention_dim: Dimensao intermediaria da atencao
    """

    def __init__(
        self,
        hidden_dim: int,
        attention_dim: int = 32,
    ):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(
        self, lstm_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aplica atencao sobre a sequencia.

        Args:
            lstm_output: Output do LSTM (batch, seq_len, hidden_dim)

        Returns:
            Tupla (context_vector, attention_weights)
            - context_vector: (batch, hidden_dim)
            - attention_weights: (batch, seq_len)
        """
        # Calcular scores de atencao
        scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)

        # Softmax para pesos
        weights = torch.softmax(scores, dim=1)

        # Vetor de contexto (soma ponderada)
        context = torch.bmm(
            weights.unsqueeze(1), lstm_output
        ).squeeze(1)  # (batch, hidden_dim)

        return context, weights


class DTCAutoencoderWithAttention(nn.Module):
    """
    DTC Autoencoder com mecanismo de atencao temporal.

    Adiciona atencao para interpretabilidade - identifica quais
    timesteps sao mais importantes para o clustering.

    Args:
        input_dim: Dimensao de entrada
        hidden_dim: Dimensao das camadas ocultas
        latent_dim: Dimensao do espaco latente
        seq_len: Comprimento da sequencia
        n_layers: Numero de camadas LSTM
        dropout: Taxa de dropout
        attention_dim: Dimensao da camada de atencao
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        latent_dim: int = 8,
        seq_len: int = 12,
        n_layers: int = 2,
        dropout: float = 0.1,
        attention_dim: int = 32,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True,
        )

        # Atencao temporal
        self.attention = TemporalAttention(
            hidden_dim * 2,  # bidirectional
            attention_dim,
        )

        # Projecao para espaco latente
        self.fc_latent = nn.Linear(hidden_dim * 2, latent_dim)

        # Decoder
        self.decoder = LSTMDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            seq_len=seq_len,
            n_layers=n_layers,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Tensor (batch, seq_len, input_dim)

        Returns:
            Tupla (reconstruction, embedding, attention_weights)
        """
        # Encoder
        lstm_out, _ = self.encoder_lstm(x)

        # Atencao
        context, attention_weights = self.attention(lstm_out)

        # Espaco latente
        embedding = self.fc_latent(context)

        # Decoder
        reconstruction = self.decoder(embedding)

        return reconstruction, embedding, attention_weights

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apenas codifica a entrada.

        Args:
            x: Tensor (batch, seq_len, input_dim)

        Returns:
            Embedding (batch, latent_dim)
        """
        lstm_out, _ = self.encoder_lstm(x)
        context, _ = self.attention(lstm_out)
        embedding = self.fc_latent(context)
        return embedding

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Obtem pesos de atencao para interpretabilidade.

        Args:
            x: Tensor (batch, seq_len, input_dim)

        Returns:
            Pesos de atencao (batch, seq_len)
        """
        lstm_out, _ = self.encoder_lstm(x)
        _, attention_weights = self.attention(lstm_out)
        return attention_weights


# =============================================================================
# MODELOS ADICIONAIS
# =============================================================================


class ConvAutoencoder(nn.Module):
    """
    Autoencoder convolucional 1D para series temporais.

    Arquitetura leve usando convolucoes para capturar
    padroes temporais locais.

    Args:
        input_dim: Dimensao de entrada (features por timestep)
        seq_len: Comprimento da sequencia
        latent_dim: Dimensao do espaco latente
        hidden_channels: Lista de canais das camadas conv
    """

    def __init__(
        self,
        input_dim: int = 1,
        seq_len: int = 12,
        latent_dim: int = 8,
        hidden_channels: list = None,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [32, 64, 128]

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        in_ch = input_dim
        for out_ch in hidden_channels:
            encoder_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2, ceil_mode=True),
            ])
            in_ch = out_ch

        self.encoder_conv = nn.Sequential(*encoder_layers)

        # Calcular tamanho apos convs
        with torch.no_grad():
            dummy = torch.zeros(1, input_dim, seq_len)
            conv_out = self.encoder_conv(dummy)
            self.conv_out_size = conv_out.shape[1] * conv_out.shape[2]

        self.fc_latent = nn.Linear(self.conv_out_size, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.conv_out_size)
        self.decoder_conv_shape = conv_out.shape[1:]

        decoder_layers = []
        hidden_channels_rev = hidden_channels[::-1]
        for i, out_ch in enumerate(hidden_channels_rev[1:] + [input_dim]):
            in_ch = hidden_channels_rev[i]
            decoder_layers.extend([
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.BatchNorm1d(out_ch) if out_ch != input_dim else nn.Identity(),
                nn.ReLU() if out_ch != input_dim else nn.Sigmoid(),
            ])

        self.decoder_conv = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Codifica entrada para espaco latente."""
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        h = self.encoder_conv(x)
        h = h.flatten(start_dim=1)
        z = self.fc_latent(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodifica do espaco latente."""
        h = self.fc_decode(z)
        h = h.view(-1, *self.decoder_conv_shape)
        x = self.decoder_conv(h)
        # Ajustar tamanho
        x = x[:, :, :self.seq_len]
        x = x.transpose(1, 2)
        return x

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z


class InceptionModule(nn.Module):
    """
    Modulo Inception para series temporais.

    Combina convolucoes com diferentes tamanhos de kernel
    para capturar padroes em diferentes escalas temporais.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list = None,
        bottleneck: int = 32,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [10, 20, 40]

        self.bottleneck = nn.Conv1d(in_channels, bottleneck, kernel_size=1)

        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck, out_channels, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])

        self.maxpool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

        self.bn = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bottleneck
        h = self.bottleneck(x)

        # Multi-scale convolutions
        conv_outs = [conv(h) for conv in self.convs]
        conv_outs.append(self.maxpool(x))

        # Concatenar
        out = torch.cat(conv_outs, dim=1)
        out = self.bn(out)
        out = self.relu(out)

        return out


class InceptionTimeEncoder(nn.Module):
    """
    Encoder baseado em InceptionTime para series temporais.

    Args:
        input_dim: Dimensao de entrada
        seq_len: Comprimento da sequencia
        latent_dim: Dimensao do espaco latente
        n_modules: Numero de modulos Inception
        hidden_channels: Canais por modulo
    """

    def __init__(
        self,
        input_dim: int = 1,
        seq_len: int = 12,
        latent_dim: int = 16,
        n_modules: int = 3,
        hidden_channels: int = 32,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # Modulos Inception
        modules = []
        in_ch = input_dim
        out_ch = hidden_channels * 4  # 4 branches no InceptionModule

        for i in range(n_modules):
            modules.append(InceptionModule(in_ch, hidden_channels))
            in_ch = out_ch

        self.inception = nn.Sequential(*modules)

        # Global average pooling + FC
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_ch, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.transpose(1, 2)

        h = self.inception(x)
        h = self.gap(h).squeeze(-1)
        z = self.fc(h)

        return z


class InceptionTimeDecoder(nn.Module):
    """
    Decoder para InceptionTime Autoencoder.

    Args:
        latent_dim: Dimensao do espaco latente
        output_dim: Dimensao de saida
        seq_len: Comprimento da sequencia
        hidden_channels: Canais intermediarios
    """

    def __init__(
        self,
        latent_dim: int = 16,
        output_dim: int = 1,
        seq_len: int = 12,
        hidden_channels: int = 64,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.output_dim = output_dim

        self.fc = nn.Linear(latent_dim, hidden_channels * seq_len)

        self.conv = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels // 2, output_dim, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)

        h = self.fc(z)
        h = h.view(batch_size, -1, self.seq_len)
        x = self.conv(h)
        x = x.transpose(1, 2)

        return x


class InceptionTimeAutoencoder(nn.Module):
    """
    Autoencoder completo baseado em InceptionTime.

    Args:
        input_dim: Dimensao de entrada
        seq_len: Comprimento da sequencia
        latent_dim: Dimensao do espaco latente
        n_modules: Numero de modulos Inception
        hidden_channels: Canais por modulo
    """

    def __init__(
        self,
        input_dim: int = 1,
        seq_len: int = 12,
        latent_dim: int = 16,
        n_modules: int = 3,
        hidden_channels: int = 32,
    ):
        super().__init__()

        self.encoder = InceptionTimeEncoder(
            input_dim=input_dim,
            seq_len=seq_len,
            latent_dim=latent_dim,
            n_modules=n_modules,
            hidden_channels=hidden_channels,
        )

        self.decoder = InceptionTimeDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            seq_len=seq_len,
            hidden_channels=hidden_channels * 4,
        )

        self.latent_dim = latent_dim
        self.seq_len = seq_len

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class InceptionTimeAutoencoderWithAttention(nn.Module):
    """
    InceptionTime Autoencoder com atencao temporal.

    Args:
        input_dim: Dimensao de entrada
        seq_len: Comprimento da sequencia
        latent_dim: Dimensao do espaco latente
        n_modules: Numero de modulos Inception
        hidden_channels: Canais por modulo
        attention_dim: Dimensao da camada de atencao
    """

    def __init__(
        self,
        input_dim: int = 1,
        seq_len: int = 12,
        latent_dim: int = 16,
        n_modules: int = 3,
        hidden_channels: int = 32,
        attention_dim: int = 32,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # Modulos Inception
        modules = []
        in_ch = input_dim
        out_ch = hidden_channels * 4

        for i in range(n_modules):
            modules.append(InceptionModule(in_ch, hidden_channels))
            in_ch = out_ch

        self.inception = nn.Sequential(*modules)

        # Atencao temporal
        self.attention = TemporalAttention(out_ch, attention_dim)

        # Projecao para latente
        self.fc_latent = nn.Linear(out_ch, latent_dim)

        # Decoder
        self.decoder = InceptionTimeDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            seq_len=seq_len,
            hidden_channels=hidden_channels * 4,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encoder
        x_t = x.transpose(1, 2)
        h = self.inception(x_t)

        # Atencao
        h_t = h.transpose(1, 2)
        context, attention_weights = self.attention(h_t)

        # Latente
        z = self.fc_latent(context)

        # Decoder
        reconstruction = self.decoder(z)

        return reconstruction, z, attention_weights

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        h = self.inception(x_t)
        h_t = h.transpose(1, 2)
        context, _ = self.attention(h_t)
        z = self.fc_latent(context)
        return z

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        h = self.inception(x_t)
        h_t = h.transpose(1, 2)
        _, attention_weights = self.attention(h_t)
        return attention_weights


class TS2VecEncoder(nn.Module):
    """
    Encoder inspirado no TS2Vec para aprendizado contrastivo.

    Arquitetura baseada em dilated convolutions para capturar
    dependencias de longo alcance.

    Args:
        input_dim: Dimensao de entrada
        hidden_dim: Dimensao das camadas ocultas
        latent_dim: Dimensao do espaco latente
        n_layers: Numero de camadas conv
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        n_layers: int = 4,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Camada de entrada
        self.input_fc = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        # Dilated convolutions
        layers = []
        for i in range(n_layers):
            dilation = 2 ** i
            layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3,
                              padding=dilation, dilation=dilation),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                )
            )
        self.conv_layers = nn.ModuleList(layers)

        # Projecao para latente
        self.fc_out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.transpose(1, 2)

        h = self.input_fc(x)

        # Residual connections com dilated convs
        for conv in self.conv_layers:
            h = h + conv(h)

        # Global average pooling
        h = h.mean(dim=2)

        # Projecao
        z = self.fc_out(h)

        return z

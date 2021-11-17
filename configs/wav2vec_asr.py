from dataclasses import dataclass
from data_utils import ASRCollate
from models.wav2vec_asr import Wav2VecASR, Wav2VecASRLoss

@dataclass
class TrainingParams:
  epochs: int = 30
  learning_rate: float = 1e-4
  weight_decay: float = 0.005
  grad_clip_thresh: float = 1.0
  batch_size: float = 2
  report_interval: int = 5
  save_interval: int = 100
  random_seed: int = 42
  val_size: float = 0.1
  model_path: str = "wav2vec_asr.pth"
  wandb: str = "accent_embeddings"
  model_cls = Wav2VecASR
  loss_cls = Wav2VecASRLoss

@dataclass
class DataParams:
  filter_length: int = 2048
  hop_length: int = 256
  win_length: int = 1024
  n_mel_channels: int = 80
  fmin: float = 0.0
  fmax: float = 8000.0
  sample_rate: float = 48000
  speaker: str = None
  silence_thresh: float = 35
  collate_cls = ASRCollate

@dataclass
class ModelParams:
  model_name: str = "facebook/wav2vec2-base"
  ctc_loss_reduction: str = "mean"
  pad_token_id: int = 0
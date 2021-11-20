from dataclasses import dataclass
from text import symbols
import hashlib

@dataclass
class TrainingParams:
  epochs:int = 5
  learning_rate: float = 1e-3
  weight_decay: float = 1e-6
  grad_clip_thresh: float = 1.0
  batch_size:float = 4
  report_interval: int = 5
  save_interval: int = 100
  model_path: str = "tacotron.pth"
  random_seed: int = 42
  val_size: float = 0.1

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

  def wav_hash(self):
    return hashlib.md5(f"{self.sample_rate}-{self.silence_thresh}".encode('utf-8')).hexdigest()

  def mfcc_hash(self):
    m = hashlib.md5()
    m.update(str(self.filter_length).encode('utf-8'))
    m.update(str(self.hop_length).encode('utf-8'))
    m.update(str(self.win_length).encode('utf-8'))
    m.update(str(self.n_mel_channels).encode('utf-8'))
    m.update(str(self.fmax).encode('utf-8'))
    m.update(str(self.fmin).encode('utf-8'))
    m.update(str(self.sample_rate).encode('utf-8'))
    m.update(str(self.silence_thresh).encode('utf-8'))
    return m.hexdigest()

@dataclass
class TacotronParams:
  mask_padding:bool = True
  fp16_run:bool = False

  # Input/Output Sizes
  n_mel_channels: int = 80
  n_symbols: int = len(symbols)
  symbols_embedding_dim: int = 512
  
  # Encoder Params
  encoder_kernel_size: int = 5
  encoder_n_convolutions: int = 3
  encoder_embedding_dim: int = 512

  # Decoder Params
  n_frames_per_step: int = 1
  decoder_rnn_dim: int = 1024
  prenet_dim: int = 256
  max_decoder_steps: int =1000
  gate_threshold: float = 0.5
  p_attention_dropout: float = 0.1
  p_decoder_dropout: float = 0.1

  # Attention params
  attention_rnn_dim:int = 1024
  attention_dim:int = 128

  # Location Layer Parameters
  attention_location_n_filters:int = 32
  attention_location_kernel_size: int = 31

  # Mel Post-Net Params
  postnet_embedding_dim:int = 512
  postnet_kernel_size:int = 5
  postnet_n_convolutions:int = 5

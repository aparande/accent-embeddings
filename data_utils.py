import os
import json
from typing import Tuple

from dotenv import load_dotenv
import torch
from torch.utils.data import Dataset
from torchaudio.datasets import VCTK_092
import torchaudio.transforms as transforms

from text import text_to_sequence
from hyper_params import DataParams

# Loads environment variables
load_dotenv()

DATASET_PATH = os.environ['DATASET_PATH']

class VCTK(Dataset):
  """
  A wrapper around the VCTK_092 PyTorch dataset
  It enriches the original dataset with accent/gender information and returns MFCC instead of waveforms.
  Args:
    mfcc_num: How many MFCC coeffients to use for the waveforms
    path: From where to load the dataset (defaults to DATASET_PATH from the environment)
  """
  def __init__(self, params: DataParams, path: str = DATASET_PATH):
    self.mfcc_transform = transforms.MelSpectrogram(sample_rate=params.sample_rate, n_mels=params.n_mel_channels, f_min=params.fmin, f_max=params.fmax, win_length=params.win_length, hop_length=params.hop_length, n_fft=params.filter_length)
    self.vctk = VCTK_092(path, download=True)
    self._load_speaker_metadata()

  def _load_speaker_metadata(self):
    speaker_info_path = f"{DATASET_PATH}/VCTK-Corpus-0.92"
    gender_path = f"{speaker_info_path}/speaker_gender.json"
    accent_path = f"{speaker_info_path}/speaker_accent.json"
    if os.path.exists(gender_path) and os.path.exists(accent_path):
      with open(accent_path, 'r') as f:
        self.accent_map = json.load(f)
      with open(gender_path, 'r') as f:
        self.gender_map = json.load(f)
    else:
      with open(f"{speaker_info_path}/speaker-info.txt", 'r') as f:
        self.accent_map = dict()
        self.gender_map = dict()
        for line in f.readlines()[1:]:
          speaker, _, gender, accent = line.split()[:4]
          self.gender_map[speaker] = gender
          self.accent_map[speaker] = accent
      with open(accent_path, 'w') as f:
        json.dump(self.accent_map, f)
      with open(gender_path, 'w') as f:
        json.dump(self.gender_map, f)

  def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str, str, str, str, str]:
    """
    Load the nth sample from the dataset.
    Returns:
      tuple: (MFCC, utterance, speaker_id, utterance_id, accent, gender)
    """
    sample = self.vctk[n]
    accent = self.accent_map[sample[3]]
    gender = self.gender_map[sample[3]]
    
    mfcc = torch.log(torch.clamp(self.mfcc_transform(sample[0]), min=1e-5))
    text = torch.IntTensor(text_to_sequence(sample[2], ["english_cleaners"]))

    return mfcc.squeeze(0), text, *sample[3:], accent, gender

  def __len__(self) -> int:
    return len(self.vctk)

class TTSCollate():
  """
  Based on TextMelCollate from https://github.com/NVIDIA/tacotron2/blob/master/data_utils.py
  """
  def __init__(self, n_frames_per_step):
    self.n_frames_per_step = n_frames_per_step

  def __call__(self, batch):
    """
    batch: [mfcc, text, speaker_id, utterance_id, accent, gender]
    """

    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(
      torch.LongTensor([len(x[1]) for x in batch]),
      dim=0, descending=True)
    max_input_len = input_lengths[0]

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
      text = batch[ids_sorted_decreasing[i]][1]
      text_padded[i, :text.size(0)] = text

    # Right zero-pad mel-spec
    num_mels = batch[0][0].size(0)
    max_target_len = max([x[0].size(1) for x in batch])
    if max_target_len % self.n_frames_per_step != 0:
      max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
      assert max_target_len % self.n_frames_per_step == 0

    # include mel padded and gate padded
    mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
    # print(mel_padded.shape)
    mel_padded.zero_()
    gate_padded = torch.FloatTensor(len(batch), max_target_len)
    gate_padded.zero_()
    output_lengths = torch.LongTensor(len(batch))
    for i in range(len(ids_sorted_decreasing)):
      mel = batch[ids_sorted_decreasing[i]][0]
      # print(mel.shape)
      mel_padded[i, :, :mel.size(1)] = mel
      gate_padded[i, mel.size(1)-1:] = 1
      output_lengths[i] = mel.size(1)

    return text_padded, input_lengths, mel_padded, gate_padded, output_lengths

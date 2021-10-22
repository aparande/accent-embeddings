import os
import json
from typing import Tuple

from dotenv import load_dotenv
import torch
from torch.utils.data import Dataset
from torchaudio.datasets import VCTK_092
import torchaudio.transforms as transforms

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
  def __init__(self, mfcc_num: int, path: str = DATASET_PATH):
    self.mfcc_num = mfcc_num
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
      tuple: (waveform, utterance, speaker_id, utterance_id, accent, gender)
    """
    sample = self.vctk[n]
    accent = self.accent_map[sample[3]]
    gender = self.gender_map[sample[3]]
    mfcc = transforms.MFCC(sample_rate=sample[1], n_mfcc=self.mfcc_num)(sample[0])

    return mfcc, *sample[2:], accent, gender

  def __len__(self) -> int:
    return len(self.vctk)


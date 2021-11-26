"""
Helper classes for manipulating and loading
"""

import os
import json
import pickle
from typing import Tuple, Any, Dict

from dotenv import load_dotenv
import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio import transforms
import librosa
from tqdm import tqdm

from text import text_to_sequence
from hyper_params import DataParams

# Loads environment variables
load_dotenv()

DATASET_PATH = os.environ['DATASET_PATH']

class VCTK(Dataset):
  """
  A base class for VCTK based datasets.
  Based on VCTK_092 from https://pytorch.org/audio/stable/_modules/torchaudio/datasets/vctk.html
  """
  def __init__(self, params: DataParams, precompute_features: bool = True, root: str = DATASET_PATH, mic_id: str = "mic2", audio_ext: str = ".flac"):
    self.mfcc_transform = transforms.MelSpectrogram(sample_rate=params.sample_rate, n_mels=params.n_mel_channels,
                                                    f_min=params.fmin, f_max=params.fmax, win_length=params.win_length,
                                                    hop_length=params.hop_length, n_fft=params.filter_length)

    self.resample_transform = transforms.Resample(orig_freq=48000, new_freq=params.sample_rate)
    self.precompute_features = precompute_features

    self.params = params

    if mic_id not in ["mic1", "mic2"]:
      raise RuntimeError(f'`mic_id` has to be either "mic1" or "mic2". Found: {mic_id}')

    self._path = os.path.join(root, "VCTK-Corpus-0.92")
    self._txt_dir = os.path.join(self._path, "txt")

    self._orig_audio_dir = os.path.join(self._path, "wav48_silence_trimmed")

    sr_int = int(self.params.sample_rate / 1000)
    self._audio_dir = os.path.join(self._path, f"wav{sr_int}")
    self._mfcc_dir = os.path.join(self._path, "mfcc")
    self._mic_id = mic_id
    self._audio_ext = audio_ext

    if not os.path.isdir(self._path):
      raise RuntimeError("Dataset not found. Please download it from https://datashare.ed.ac.uk/handle/10283/2950")

    # Extracting speaker IDs from the folder structure
    self._speaker_ids = sorted(os.listdir(self._txt_dir)) if self.params.speaker is None else [self.params.speaker]
    self._sample_ids = []

    for speaker_id in self._speaker_ids:
      if speaker_id == "p280" and mic_id == "mic2":
        continue
      utterance_dir = os.path.join(self._txt_dir, speaker_id)
      for utterance_file in sorted(f for f in os.listdir(utterance_dir) if f.endswith(".txt")):
        utterance_id = os.path.splitext(utterance_file)[0]
        audio_path_mic = os.path.join(self._orig_audio_dir, speaker_id, f"{utterance_id}_{mic_id}{self._audio_ext}")
        if speaker_id == "p362" and not os.path.isfile(audio_path_mic):
          continue
        self._sample_ids.append(utterance_id.split("_"))

    self._load_speaker_metadata()

    if precompute_features:
      self._precompute()

  def _precompute(self) -> None:
    if not os.path.exists(self._audio_dir):
      os.mkdir(self._audio_dir)
    if not os.path.exists(self._mfcc_dir):
      os.mkdir(self._mfcc_dir)

    audio_hash_path = f"{self._audio_dir}/hash.txt"
    mfcc_hash_path = f"{self._mfcc_dir}/hash.txt"
    save_audio = True
    save_mfcc = True
    if os.path.isfile(audio_hash_path):
      with open(audio_hash_path, 'r') as f:
        save_audio = f.readline() != self.params.wav_hash()
      if save_audio:
        print("WARNING: OVERWRITING OLD PRECOMPUTED WAVEFORMS")

    if os.path.exists(mfcc_hash_path):
      with open(mfcc_hash_path, 'r') as f:
        save_mfcc = f.readline() != self.params.mfcc_hash()
      if save_mfcc:
        print("WARNING: OVERWRITING OLD PRECOMPUTED MFCCS")

    if not save_mfcc and not save_audio:
      return

    for speaker_id, utterance_id in tqdm(self._sample_ids):
      wav_path = os.path.join(self._audio_dir, speaker_id)
      mfcc_path = os.path.join(self._mfcc_dir, speaker_id)

      waveform, _ = self._load_original_sample(speaker_id, utterance_id)
      if save_audio:
        waveform = self._process_waveform(waveform)
      if save_mfcc:
        mfcc = self._process_mfcc(waveform)

      if not os.path.exists(wav_path):
        os.mkdir(wav_path)
      if not os.path.exists(mfcc_path): 
        os.mkdir(mfcc_path)

      if save_audio:
        torch.save(waveform, f"{wav_path}/{speaker_id}_{utterance_id}.pt")
      if save_mfcc:
        torch.save(mfcc, f"{mfcc_path}/{speaker_id}_{utterance_id}.pt")

    print("Done precomputing data")
    with open(audio_hash_path, 'w') as f:
      f.write(self.params.wav_hash())
    with open(mfcc_hash_path, 'w') as f:
      f.write(self.params.mfcc_hash())

  def _process_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
    if self.params.sample_rate != 48000:
      waveform = self.resample_transform(waveform)

    _, idx = librosa.effects.trim(waveform.numpy()[0], top_db=self.params.silence_thresh)
    return waveform[:, idx[0]:]

  def _process_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(self.mfcc_transform(waveform), min=1e-5))

  def _load_text(self, file_path: str) -> str:
    with open(file_path) as f:
      return f.readlines()[0]

  def _load_audio(self, file_path: str) -> Tuple[torch.Tensor, int]:
    return torchaudio.load(file_path)

  def _load_speaker_metadata(self) -> None:
    gender_path = f"{self._path}/speaker_gender.json"
    accent_path = f"{self._path}/speaker_accent.json"
    if os.path.exists(gender_path) and os.path.exists(accent_path):
      with open(accent_path, 'r') as f:
        self.accent_map = json.load(f)
      with open(gender_path, 'r') as f:
        self.gender_map = json.load(f)
    else:
      with open(f"{self._path}/speaker-info.txt", 'r') as f:
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

  def _load_original_sample(self, speaker_id: str, utterance_id: str) -> Tuple[torch.Tensor, str]:
    transcript_path = os.path.join(self._txt_dir, speaker_id, f"{speaker_id}_{utterance_id}.txt")
    audio_path = os.path.join(self._orig_audio_dir, speaker_id, f"{speaker_id}_{utterance_id}_{self._mic_id}{self._audio_ext}")

    transcript = self._load_text(transcript_path)
    waveform, _ = self._load_audio(audio_path)

    return waveform, transcript

  def _load_sample(self, speaker_id: str, utterance_id: str) -> Tuple[torch.Tensor, torch.Tensor, str]:
    if self.precompute_features:
      transcript_path = os.path.join(self._txt_dir, speaker_id, f"{speaker_id}_{utterance_id}.txt")
      transcript = self._load_text(transcript_path)

      wav_path = os.path.join(self._audio_dir, speaker_id, f"{speaker_id}_{utterance_id}.pt")
      waveform = torch.load(wav_path)

      mfcc_path = os.path.join(self._mfcc_dir, speaker_id, f"{speaker_id}_{utterance_id}.pt")
      mfcc = torch.load(mfcc_path)
    else:
      waveform, transcript = self._load_original_sample(speaker_id, utterance_id)
      _, idx = librosa.effects.trim(waveform.numpy()[0], top_db=self.params.silence_thresh)
      waveform = waveform[:, idx[0]:]
      mfcc = torch.log(torch.clamp(self.mfcc_transform(waveform), min=1e-5))

    return waveform, mfcc, transcript


  def __getitem__(self, n: int) -> Dict[str, Any]:
    speaker_id, utterance_id = self._sample_ids[n]
    waveform, mfcc, transcript = self._load_sample(speaker_id, utterance_id)

    accent = self.accent_map[speaker_id]
    gender = self.gender_map[speaker_id]

    text = torch.IntTensor(text_to_sequence(transcript, ["english_cleaners"]))

    return {
      "mfcc": mfcc.squeeze(0),
      "text_tensor": text,
      "text": transcript,
      "waveform": waveform.squeeze(0),
      "speaker_id": speaker_id,
      "utterance_id": utterance_id,
      "accent": accent,
      "gender": gender
    }

  def __len__(self) -> int:
    return len(self._sample_ids)

class TTSCollate():
  """
  Based on TextMelCollate from https://github.com/NVIDIA/tacotron2/blob/master/data_utils.py
  """
  def __init__(self, n_frames_per_step):
    self.n_frames_per_step = n_frames_per_step

  def __call__(self, batch):
    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(
      torch.LongTensor([len(x["text_tensor"]) for x in batch]),
      dim=0, descending=True)
    max_input_len = input_lengths[0]

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i, idx in enumerate(ids_sorted_decreasing):
      text = batch[idx]["text_tensor"]
      text_padded[i, :text.size(0)] = text

    # Right zero-pad mel-spec
    num_mels = batch[0]["mfcc"].size(0)
    max_target_len = max([x["mfcc"].size(1) for x in batch])
    if max_target_len % self.n_frames_per_step != 0:
      max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
      assert max_target_len % self.n_frames_per_step == 0

    # include mel padded and gate padded
    mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
    mel_padded.zero_()
    gate_padded = torch.FloatTensor(len(batch), max_target_len)
    gate_padded.zero_()
    output_lengths = torch.LongTensor(len(batch))
    for i, idx in enumerate(ids_sorted_decreasing):
      mel = batch[idx]["mfcc"]
      mel_padded[i, :, :mel.size(1)] = mel
      gate_padded[i, mel.size(1)-1:] = 1
      output_lengths[i] = mel.size(1)

    return {
      "text_tensor": text_padded,
      "text_lens": input_lengths,
      "mfcc": mel_padded,
      "gates": gate_padded,
      "mfcc_lens": output_lengths
    }


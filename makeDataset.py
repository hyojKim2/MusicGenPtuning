import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict
import numpy as np
from typing import Dict, List, Tuple

# 최대 오디오 길이 (10초)
max_audio_length = 320000  # 10초 * 32,000Hz

class MusicGenDataset(Dataset):
    def __init__(self, data_dir: str, sample_rate: int = 32000):
        """
        데이터셋 초기화
            data_dir: WAV와 JSON 파일이 있는 디렉토리 경로
            sample_rate: 오디오 샘플링 레이트
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        
        # WAV 파일과 JSON 파일 쌍 찾기
        self.file_pairs = []
        for file in os.listdir(data_dir):
            if file.endswith('.wav'):
                wav_path = os.path.join(data_dir, file)
                json_path = os.path.join(data_dir, file.replace('.wav', '.json'))
                
                if os.path.exists(json_path):
                    self.file_pairs.append((wav_path, json_path))
        
        print(f"Found {len(self.file_pairs)} WAV-JSON pairs")

    def __len__(self) -> int:
        return len(self.file_pairs)

    def _load_audio(self, wav_path: str) -> torch.Tensor:
        """
        WAV 파일을 로드하고 전처리
        """
        waveform, sr = torchaudio.load(wav_path)
        
        # 스테레오를 모노로 변환
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 샘플링 레이트 변환
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # 오디오 길이 조정
        if waveform.shape[1] > max_audio_length:
            waveform = waveform[:, :max_audio_length]  # 10초까지만 자름
        elif waveform.shape[1] < max_audio_length:
            pad_size = max_audio_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))  # 부족하면 패딩
        
        return waveform

    def _load_description(self, json_path: str) -> str:
        """
        JSON 파일에서 description 텍스트 로드
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['description']

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        데이터셋에서 하나의 샘플 반환
        """
        wav_path, json_path = self.file_pairs[idx]
        
        # 오디오 로드 및 전처리
        waveform = self._load_audio(wav_path)
        
        # 설명 텍스트 로드
        description = self._load_description(json_path)
        
        return {
            'audio': waveform,
            'text': description,
            'sampling_rate': self.sample_rate
        }

def create_dataset(data_dir: str) -> DatasetDict:
    """
    학습/검증 데이터셋 생성
    Args:
        data_dir: 데이터 디렉토리 경로
    Returns:
        DatasetDict containing train and validation splits
    """
    dataset = MusicGenDataset(data_dir)
    
    # 데이터셋을 학습/검증 세트로 분할 (8:2 비율)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

if __name__ == '__main__':
    # 사용 예시
    data_dir = "C:/Users/a/Desktop/MusicGen/musicgen-model/dataset_wav"
    dataset = create_dataset(data_dir)
    
    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Validation dataset size: {len(dataset['validation'])}")
    
    # 첫 번째 샘플 확인
    sample = dataset['train'][0]
    print(f"Audio shape: {sample['audio'].shape}")
    print(f"Text description: {sample['text']}")
    print(f"Sampling rate: {sample['sampling_rate']}")

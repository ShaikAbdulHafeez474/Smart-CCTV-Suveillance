# dataset.py
import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class VideoDataset(Dataset):
    def __init__(self, root_dir, sequence_length=16, transform=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples = []
        self.class_map = {'normal': 0, 'suspicious': 1}  # Explicit class mapping

        for cls_name in os.listdir(root_dir):
            cls_folder = os.path.join(root_dir, cls_name)
            for video_file in os.listdir(cls_folder):
                self.samples.append({
                    "path": os.path.join(cls_folder, video_file),
                    "label": self.class_map[cls_name]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = self._load_video_frames(sample['path'])
        label = sample['label']

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        frames = torch.stack(frames)  # Shape: [T, C, H, W]
        return frames, label

    def _load_video_frames(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_indices = self._get_frame_indices(total_frames)
        frames = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = cap.read()
            if not success:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)

        cap.release()

        # Pad or trim to sequence length
        if len(frames) < self.sequence_length:
            while len(frames) < self.sequence_length:
                frames.append(frames[-1])  # repeat last frame
        else:
            frames = frames[:self.sequence_length]

        return [transforms.ToTensor()(f) for f in frames]

    def _get_frame_indices(self, total_frames):
        if total_frames < self.sequence_length:
            return list(range(total_frames))
        interval = total_frames // self.sequence_length
        return [i * interval for i in range(self.sequence_length)]

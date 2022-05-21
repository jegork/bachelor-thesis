import torch
from torchvision import transforms
import torchvision

INPUT_SHAPE = (32, 224, 224, 3)

VIDEOSWIN_NORM = {
    "mean": torch.tensor([123.675, 116.28, 103.53]) / 255.0,
    "std": torch.tensor([58.395, 57.12, 57.375]) / 255.0,
}


class UCF101(torch.utils.data.Dataset):
    def __init__(self, videos_path, split_path, split_n, train, frame_sampler=None):
        self.videos_path = videos_path
        self.split_path = split_path
        self.split_n = split_n
        self.train = train
        self.frame_sampler = frame_sampler
        self.n_frames = INPUT_SHAPE[0]

        self.files = self.get_files()
        self.labels_map = self.get_labels_map()

    def get_labels_map(self):
        with open(f"{self.split_path}/classInd.txt", "r") as f:
            labels_map = {
                l.strip().split(" ")[1]: l.strip().split(" ")[0] for l in f.readlines()
            }

        return labels_map

    def get_files(self):
        if self.train:
            file_path = f"{self.split_path}/trainlist0{self.split_n}.txt"
        else:
            file_path = f"{self.split_path}/testlist0{self.split_n}.txt"

        with open(file_path, "r") as f:
            files = [f.split(" ")[0].strip() for f in f.readlines()]

        return files

    def __len__(self):
        return len(self.files)

    def sample_frames(self, video, idx):
        # sample every 3rd frame by default
        if self.frame_sampler is None or isinstance(self.frame_sampler, int):
            n = self.frame_sampler if isinstance(self.frame_sampler, int) else 3
            sampling = torch.arange(0, video.shape[0], n, dtype=torch.int64)

            out = video[sampling]

            # pad to the desired frames number
            if out.shape[0] < self.n_frames:
                pad_n = self.n_frames - out.shape[0]
                pad = torch.zeros((pad_n,) + out.shape[1:])

                out = torch.cat((out, pad), 0)

            # trim to the desired frames number
            elif out.shape[0] > self.n_frames:
                return out[: self.n_frames]

            return out

        # if self.frame_sampler is a list, then it is a list of frame indices with keyframes
        elif isinstance(self.frame_sampler, list):
            out = video[self.frame_sampler[idx]]
            if out.shape[0] < self.n_frames:
                pad_n = self.n_frames - out.shape[0]
                pad = torch.zeros((pad_n,) + out.shape[1:])

                out = torch.cat((out, pad), 0)

            return out

    def augment(self, video):
        t = [transforms.ConvertImageDtype(torch.float)]

        t = transforms.Compose(t)
        return t(video)

    def post_process(self, video):
        return video

    def format_output(self, video, label):
        return video, label

    def __getitem__(self, idx):
        video_path = self.files[idx]
        label = int(self.labels_map[video_path.split("/")[0]]) - 1

        video_path = self.videos_path + "/" + video_path

        video = torchvision.io.read_video(video_path)[0]

        video = torch.transpose(video, 3, 2)
        video = torch.transpose(video, 1, 2)

        video = self.sample_frames(video, idx)

        video = self.augment(video)

        return self.format_output(video, label)


class BaselineUCF101(UCF101):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    frame_sampler = 3

    def augment(self, video):
        if self.train:
            t = transforms.Compose(
                [
                    transforms.ConvertImageDtype(torch.float),
                    transforms.RandomCrop(
                        size=INPUT_SHAPE[1:-1],
                        pad_if_needed=True,
                        fill=0,
                        padding_mode="constant",
                    ),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.15, saturation=0.15
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
                    transforms.Normalize(mean=self.MEAN, std=self.STD),
                ]
            )
        else:
            t = transforms.Compose(
                [
                    transforms.ConvertImageDtype(torch.float),
                    transforms.CenterCrop(size=INPUT_SHAPE[1:-1]),
                    transforms.Normalize(mean=self.MEAN, std=self.STD),
                ]
            )

        return t(video).transpose(0, 1)


class VIVIT_UCF101(UCF101):
    def __init__(
        self,
        videos_path,
        split_path,
        split_n,
        train,
        feature_extractor,
        frame_sampler=None,
    ):
        super(VIVIT_UCF101, self).__init__(
            videos_path, split_path, split_n, train, frame_sampler=frame_sampler
        )
        self.feature_extractor = feature_extractor

    def augment(self, video):
        if self.train:
            t = transforms.Compose(
                [
                    transforms.ConvertImageDtype(torch.float),
                    transforms.RandomCrop(
                        size=INPUT_SHAPE[1:-1],
                        pad_if_needed=True,
                        fill=0,
                        padding_mode="constant",
                    ),
                    transforms.ColorJitter(
                        brightness=0.3, contrast=0.25, saturation=0.25
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomErasing(p=0.25),
                ]
            )
        else:
            t = transforms.Compose(
                [
                    transforms.ConvertImageDtype(torch.float),
                    transforms.CenterCrop(size=INPUT_SHAPE[1:-1]),
                ]
            )

        return t(video).transpose(0, 1)

    def format_output(self, video, label):
        video = video.transpose(0, 1)
        video = [v for v in video] # convert video to a list of frames
        video = self.feature_extractor(images=video, return_tensors="pt")["pixel_values"]

        video = torch.transpose(video, 0, 1)
        return {"pixel_values": video, "label": label}


class SWIN_UCF101(UCF101):
    def __init__(
        self, videos_path, split_path, split_n, train, normalize, frame_sampler=None
    ):
        super(SWIN_UCF101, self).__init__(
            videos_path, split_path, split_n, train, frame_sampler=frame_sampler
        )
        self.normalize = normalize

    def augment(self, video):
        if self.train:
            t = [
                transforms.ConvertImageDtype(torch.float),
                transforms.RandomCrop(
                    size=INPUT_SHAPE[1:-1],
                    pad_if_needed=True,
                    fill=0,
                    padding_mode="constant",
                ),
                transforms.ColorJitter(brightness=0.3, contrast=0.25, saturation=0.25),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomErasing(p=0.25),
            ]
        else:
            t = [
                transforms.ConvertImageDtype(torch.float),
                transforms.CenterCrop(size=INPUT_SHAPE[1:-1]),
            ]

        if self.normalize:
            mean, std = self.normalize["mean"], self.normalize["std"]
            t.append(transforms.Normalize(mean, std))

        t = transforms.Compose(t)
        return t(video).transpose(0, 1)

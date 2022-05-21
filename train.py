import pytorch_lightning as pl
import joblib
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


def train_vivit(
    base_model_name,
    tokenizer_name,
    training_config,
    train_kfs=None,
    test_kfs=None,
):
    from transformers import ViTFeatureExtractor, TrainingArguments, Trainer
    from data import VIVIT_UCF101
    from vivit_transformers import ViViTForImageClassification

    def compute_metrics(data):
        return {"accuracy": accuracy_score(data.label_ids, data.predictions.argmax(-1))}

    feature_extractor = ViTFeatureExtractor.from_pretrained(tokenizer_name)

    dataset_train = VIVIT_UCF101(
        "UCF-101",
        "ucfTrainTestlist",
        1,
        True,
        feature_extractor=feature_extractor,
        frame_sampler=train_kfs,
    )
    dataset_test = VIVIT_UCF101(
        "UCF-101",
        "ucfTrainTestlist",
        1,
        False,
        feature_extractor=feature_extractor,
        frame_sampler=test_kfs,
    )

    model = ViViTForImageClassification.from_pretrained(base_model_name, num_labels=101)

    for p in model.parameters():
        p.requires_grad = True

    training_args = TrainingArguments(
        output_dir="vivit",
        num_train_epochs=training_config["epochs"],
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config["batch_size"],
        warmup_steps=10000 / training_config["accumulate_grad_batches"],
        weight_decay=0.01,
        logging_dir="./logs",
        logging_strategy="steps",
        save_strategy="epoch",
        fp16=True,
        dataloader_pin_memory=True,
        learning_rate=training_config["learning_rate"],
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=500,
        label_smoothing_factor=0.15,
        save_total_limit=1,
        gradient_accumulation_steps=training_config["accumulate_grad_batches"],
        dataloader_num_workers=training_config["num_workers"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Train baseline models.")

    parser.add_argument(
        "model", type=str, help="Model to train. One of: r2plus1d, cnn_gru, swin"
    )
    parser.add_argument(
        "--train_keyframes_path",
        type=str,
        help="Path to train keyframes. If none, then samples every 3rd frame.",
    )
    parser.add_argument(
        "--test_keyframes_path",
        type=str,
        help="Path to train keyframes. If none, then samples every 3rd frame.",
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for training and testing."
    )
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument(
        "--num_workers", type=int, help="Number of workers for dataloader"
    )
    parser.add_argument(
        "--state_dict_name", type=str, help="Name to be used for the state_dict file"
    )
    parser.add_argument(
        "--vivit_base_model_path",
        type=str,
        help="Path to the base VIVIT model that will be finetuned",
    )
    # parser.add_argument("--use_checkpoint", type=bool, help="Whether to use last best checkpoint for model", default=True)

    args = parser.parse_args()

    if args.model not in ["r2plus1d", "cnn_gru", "swin", "vivit"]:
        raise Exception("Model is not one of: r2plus1d, cnn_gru, swin, vivit")

    train_kfs = None
    test_kfs = None
    epochs = None
    accumulate_grad_batches = None
    num_workers = None
    dataset_train = None
    dataset_test = None
    batch_size = None

    if args.train_keyframes_path:
        train_kfs = joblib.load(args.train_keyframes_path)

    if args.test_keyframes_path:
        test_kfs = joblib.load(args.test_keyframes_path)

    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    if args.model == "vivit":
        training_cfg = {
            "epochs": 10,
            "learning_rate": 4e-5,
            "batch_size": 6,
            "accumulate_grad_batches": 2,
            "num_workers": 16,
        }

        train_vivit(
            args.vivit_base_model_path,
            "facebook/dino-vitb16",
            training_cfg,
            train_kfs=train_kfs,
            test_kfs=test_kfs,
        )

    else:
        ckpt_folder_path = f"checkpoints/{args.model}"

        # get available checkpoints
        checkpoints = (
            os.listdir(ckpt_folder_path) if os.path.exists(ckpt_folder_path) else []
        )

        if len(checkpoints) == 1:
            ckpt_path = os.path.join(ckpt_folder_path, checkpoints[0])
        elif len(checkpoints) == 0:
            ckpt_path = None
        elif len(checkpoints) > 1:
            raise Exception("Too many checkpoints in folder")

        if args.model == "swin":
            from data import SWIN_UCF101, VIDEOSWIN_NORM
            from models import VideoSwin

            model = VideoSwin(
                "./swin/checkpoints/swin_base_patch244_window1677_sthv2.pth",
                101,
                0.15,
                embed_dim=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                patch_size=(2, 4, 4),
                window_size=(16, 7, 7),
                drop_path_rate=0.4,
                patch_norm=True,
            )

            epochs = 10
            accumulate_grad_batches = 2
            num_workers = 8
            batch_size = 6

            dataset_train = SWIN_UCF101(
                "UCF-101",
                "ucfTrainTestlist",
                1,
                True,
                normalize=VIDEOSWIN_NORM,
                frame_sampler=train_kfs,
            )
            dataset_test = SWIN_UCF101(
                "UCF-101",
                "ucfTrainTestlist",
                1,
                False,
                normalize=VIDEOSWIN_NORM,
                frame_sampler=test_kfs,
            )

        elif args.model == "r2plus1d":
            from data import BaselineUCF101
            from models import R2Plus1D

            model = R2Plus1D(101, 0.15)

            epochs = 10
            batch_size = 24
            num_workers = 16
            accumulate_grad_batches = 1

            dataset_train = BaselineUCF101("UCF-101", "ucfTrainTestlist", 1, True)
            dataset_test = BaselineUCF101("UCF-101", "ucfTrainTestlist", 1, False)

        elif args.model == "cnn_gru":
            from data import BaselineUCF101
            from models import CnnGru

            model = CnnGru(101, 0.15, 512)

            epochs = 25
            batch_size = 64
            num_workers = 8
            accumulate_grad_batches = 1

            dataset_train = BaselineUCF101("UCF-101", "ucfTrainTestlist", 1, True)
            dataset_test = BaselineUCF101("UCF-101", "ucfTrainTestlist", 1, False)

        if args.epochs:
            epochs = args.epochs
        if args.num_workers:
            num_workers = args.num_workers
        if args.batch_size:
            batch_size = args.batch_size

        callbacks = [
            pl.callbacks.ModelCheckpoint(
                ckpt_folder_path, monitor="test_acc_epoch", save_top_k=1, mode="max"
            ),
            pl.callbacks.EarlyStopping(
                monitor="test_acc_epoch", patience=5, min_delta=0.01, mode="max"
            ),
        ]

        trainer = pl.Trainer(
            gpus=1,
            precision=16,
            max_epochs=epochs,
            check_val_every_n_epoch=1,
            accumulate_grad_batches=accumulate_grad_batches,
            callbacks=callbacks,
        )

        train_dataloader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )
        test_dataloader = DataLoader(
            dataset_test,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers,
        )

        trainer.fit(model, train_dataloader, test_dataloader, ckpt_path=ckpt_path)

        model = model.load_from_checkpoint(ckpt_path)
        torch.save(
            model.state_dict(),
            f"{args.model if args.state_dict_name is None else args.state_dict_name}_state_dict.pt",
        )

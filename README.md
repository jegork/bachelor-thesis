# Thesis paper

[Paper is available here](paper.pdf)

# Structure

- models.py: implementation of vivit, video swin, (2+1)D model, MobileNet+GRU model.
- data.py: datasets for different models
- keyframes.py: keyframe extractor algorithms
- plotting.ipynb: different plots and visualizations
- results_analysis.ipynb: info and plots, evaluation to compare and analyse models
- folder vivit_transformers: implementation of ViViT inside of transformers library
- vivit.ipynb training of different vivit models
- feature_extraction.ipynb: extracts mobilenet and swin features and retrieves keyframes

# Convert DINO model to ViViT 32-frame model

!python3 vivit_transformers/convert_vit_to_vivit.py --vit_model_path facebook/dino-vitb16 --tubelet_n 2 --video_length 32 --output_path vivit_dino_32frames_untrained

# Training

Example of training of ViViT models is also in vivit.ipynb because ViViT was implemented as an extension to the transformers library.

To train the other models use the CLI:

To train the model without keyframes use:
```sh
python3 train.py [MODEL]
```

To train the model with keyframes use:
```sh
python3 train.py [MODEL] --train_keyframes_path [PATH] --test_keyframes_path [PATH]
```

Possible models are:
- swin
- r2plus1d
- cnn_gru
- vivit

If keyframes are not provided, then uses normal frame sampling.

Possible train keyframes:
- train_01_10frames.pkl (MobileNet features, KNN, 10 frames)
- train_01_32frames.pkl (MobileNet features, KNN, 32 frames)
- train_01_32frames_agglomerative.pkl (MobileNet features, Agglomerative Clustering, 32 frames)
- train_01_32frames_agglomerative_swin.pkl (Swin features, Agglomerative Clustering, 32 frames)


Possible test keyframes:
- test_01_10frames.pkl (MobileNet features, KNN, 10 frames)
- test_01_32frames.pkl (MobileNet features, KNN, 32 frames)
- test_01_32frames_agglomerative.pkl (MobileNet features, Agglomerative Clustering, 32 frames)
- test_01_32frames_agglomerative_swin.pkl (Swin features, Agglomerative Clustering, 32 frames)


## Example

Train videoswin with keyframes:

```sh
python3 train_videoswin.py swin --train_keyframes_path keyframes/train_01_32frames_agglomerative_swin.pkl --test_keyframes_path keyframes/test_01_32frames_agglomerative_swin.pkl
```

Train r2plus1d without keyframes (samples every 3rd frame):

```sh
python3 train_videoswin.py r2plus1d
```


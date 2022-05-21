import sklearn
import numpy as np
from scipy.spatial.distance import cosine
from tqdm.auto import tqdm

try:
    import cupy as cp
except:
    pass


def cosine_similarity(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return 1 - cosine(a, b)
    else:  # if the inputs are not numpy arrays, use the cupy arrays (GPU-accelerated)
        try:
            import cupy as cp

            if isinstance(a, cp.ndarray) and isinstance(b, cp.ndarray):
                return cp.dot(a, b) / cp.linalg.norm(a) * cp.linalg.norm(b)
        except ImportError: 
            raise Exception("cupy is not installed")


class KeyFrameExtractor:
    def __init__(self, n_keyframes, method, cpu=True, output_clusters=False):
        if method not in ["agglomerative", "kmeans"]:
            raise ValueError(
                'Only extraction methods "agglomerative" and "kmeans" are supported'
            )

        self.n_keyframes = n_keyframes
        self.method = method
        self.cpu = cpu
        self.output_clusters = output_clusters

    def get_most_represenative_sample_idx(
        self, points, method="highest_similarity", centroid=None, videos=None
    ):
        if method == "highest_similarity":
            len_points = points.shape[0]

            sums_of_similarities = []

            for i in range(len_points):
                i_sums = sum(
                    [
                        cosine_similarity(points[i], points[j])
                        for j in range(len_points)
                        if i != j
                    ]
                )
                sums_of_similarities.append((i, i_sums))

            return sorted(sums_of_similarities, key=lambda x: x[1])[-1][0]

        elif method == "centroid":
            if centroid is None:
                centroid = points.mean(0)

            similarities = [
                (i, cosine_similarity(i, centroid)) for i in range(points.shape[0])
            ]

            return sorted(similarities, key=lambda x: x[1])[-1][0]
        else:
            raise ValueError(
                'Currently only methods "centroid" and "highest_similarity" are supported'
            )

    def get_keyframes_idx(self, video):
        if video.shape[0] <= self.n_keyframes:
            return list(range(video.shape[0]))

        if (
            not self.cpu and video.shape[0] > 180
        ):  # trim video, > 220 frames produces error @ 80GB A100
            video = video[:180]

        if self.method == "kmeans":
            if self.cpu:
                from sklearn.cluster import KMeans

                clustering = KMeans(n_clusters=self.n_keyframes)
            else:
                from cuml import KMeans

                clustering = KMeans(n_clusters=self.n_keyframes, output_type="numpy")

        elif self.method == "agglomerative":
            if self.cpu:
                from sklearn.cluster import AgglomerativeClustering

                clustering = AgglomerativeClustering(n_clusters=self.n_keyframes)
            else:
                from cuml import AgglomerativeClustering

                clustering = AgglomerativeClustering(
                    n_clusters=self.n_keyframes, output_type="numpy"
                )

        clusters = clustering.fit_predict(video)
        keyframes = []

        for c in np.unique(clusters):
            c_idx = np.argwhere(clusters == c).flatten()

            if self.method == "kmeans":
                most_representative_idx = self.get_most_represenative_sample_idx(
                    video[c_idx], "centroid", clustering.cluster_centers_[c]
                )
            else:
                most_representative_idx = self.get_most_represenative_sample_idx(
                    video[c_idx]
                )

            keyframes.append(c_idx[most_representative_idx])

        keyframes.sort()

        if self.output_clusters:
            return {"keyframes": keyframes, "clusters": clusters}
        else:
            return keyframes

    def predict(self, videos):
        return [self.get_keyframes_idx(video) for video in tqdm(videos)]

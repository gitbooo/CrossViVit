import numpy as np
import matplotlib


def get_numpy_video(x, cmap="RdBu_r"):
    """
    Args:
        x (np.array): A tensor of shape [T, H, W] that will be turned into a video.
        cmap (str): Color map to use for plotting. See https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    if len(x.shape) == 3:
        frames = []
        for i in range(x.shape[0]):
            cm = matplotlib.cm.get_cmap(cmap)
            normed_data = ((x[i] - x[i].min()) / (x[i].max() - x[i].min())).numpy()
            img = cm(normed_data)
            img = (255 * img).astype("uint8")
            frames.append(img)
        frames = np.stack(frames, axis=0)  # T, H, W, 4
        frames = frames.transpose(0, 3, 1, 2)

        return frames
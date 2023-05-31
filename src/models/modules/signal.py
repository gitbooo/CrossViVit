import numpy as np
import torch


def dct1d(x, norm=None):

    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))  # add this line

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct1d(X, norm=None):

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = (
        torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :]
        * np.pi
        / (2 * N)
    )
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)  # comment this line
    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)  # add this line

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


def dct2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct1d(x, norm=norm)
    X2 = dct1d(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct1d(X, norm=norm)
    x2 = idct1d(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_separation(x: torch.Tensor, low_cutoff_mode: int):
    """
    Takes an input signal of shape [B, C, T], applies DCT and splits its low and high frequency components according
    to ``low_cutoff_mode``.
    Args:
        x (torch.Tensor): Input tensor of shape [B, C, T]
        low_cutoff_mode (int): The mode where to split frequency components.
    """
    k = dct1d(x, norm="ortho")
    low = k[:, :, :low_cutoff_mode].reshape(k.shape[0], -1)
    high = k[:, :, low_cutoff_mode:].reshape(k.shape[0], -1)
    return low, high


def fft_separation(x: torch.Tensor, low_cutoff_mode: int, padding: int = 0):
    """
    Takes an input signal of shape [B, C, T], applies FFT and splits its low and high frequency components according
    to ``low_cutoff_mode``. Since the frequency components are complex, the real and imaginary parts are concatenated.
    Args:
        x (torch.Tensor): Input tensor of shape [B, C, T]
        low_cutoff_mode (int): The mode where to split frequency components.
    """
    if padding > 0:
        x = torch.nn.functional.pad(x, (padding, padding), mode="constant")
    k = torch.fft.rfft(x, dim=-1, norm="ortho")
    low = k[:, :, :low_cutoff_mode]
    low = torch.cat([low.real, low.imag], dim=1).reshape(k.shape[0], -1)
    high = k[:, :, low_cutoff_mode:]
    high = torch.cat([high.real, high.imag], dim=1).reshape(k.shape[0], -1)
    return low, high

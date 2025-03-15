# 绘制扩散示例，用于大创演示
import numpy as np
from matplotlib import pyplot as plt

from aberration import ZERNIKE_NUMS, PhaseScreen


def plot_coeff(coeff_, i):
    fig = plt.figure()
    ps.set_zernike_coeffients(list(coeff_[0]))
    ps.add_pupil()
    image_ = ps.get_screen()
    plt.imshow(coeff_, aspect="auto", cmap=plt.get_cmap("viridis"))
    plt.yticks([])
    plt.xticks([])
    plt.colorbar()
    # plt.title("Zernike coefficient")
    plt.savefig("Image/coeff_{}.png".format(i))
    plt.imsave("Image/screen_{}.png".format(i), image_, cmap=plt.get_cmap("viridis"))


if __name__ == "__main__":
    HEIGHT = WIDTH = 512
    ps = PhaseScreen(HEIGHT, ZERNIKE_NUMS, cache_dir="TURBULENCE/cache")
    ps.simulate_turbulence(10, "zernike")
    ps.add_pupil()
    mode = 'coeff'
    if mode == 'coeff':
        coeff = ps.get_coeffients()
        coeff = (coeff - np.mean(coeff)) / np.std(coeff)
        coeff = coeff[None, :]
        np.save("Image/coeff.npy", coeff)
        ############
        coeff = np.load("Image/coeff.npy")
        plot_coeff(coeff, 0)

        eps = np.random.normal(0, 0.5, coeff.shape)
        noise_coeff = coeff + eps
        plot_coeff(noise_coeff, 1)

        noise_coeff = coeff + np.random.normal(0, 0.5, coeff.shape)
        plot_coeff(noise_coeff, 2)

        noise_coeff = coeff + np.random.normal(0, 0.5, coeff.shape)
        plot_coeff(noise_coeff, 3)

        noise = np.random.normal(0, 1, noise_coeff.shape)
        plot_coeff(noise, 4)

    elif mode == 'Image':
        Image = ps.get_screen()
        Image = (Image - np.mean(Image)) / np.var(Image)
        np.save("Image/phase_screen.npy", Image)
        #################
        Image = np.load("Image/phase_screen.npy")
        plt.imsave("Image/phase_screen.png", Image)
        eps = np.random.normal(0, 0.2, (HEIGHT, WIDTH))
        Noise_Image = Image + eps
        plt.imsave("Image/noisy1.png", Noise_Image)
        noise = Image + np.random.normal(0, 3, (HEIGHT, WIDTH))
        plt.imsave("Image/noise.png", noise)

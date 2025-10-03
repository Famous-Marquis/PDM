import numpy as np
import matplotlib.pyplot as plt

from aberration import PhaseScreen
from generate_data import beam_list, process_coeff_beam

if __name__ == '__main__':
    ps=PhaseScreen(N=224)
    ps.simulate_turbulence(r0=0.05, l0= 5e-3, L0=10, R= 0.5,alpha= 10 / 3)
    plt.imshow(abs(ps.get_psf()))
    plt.colorbar()
    plt.show()
    coeff=ps.get_coeffients()
    for beam in beam_list:
        plt.imshow(beam)
        plt.colorbar()
        plt.show()
        img_real,img_im,img=process_coeff_beam(beam=beam, coeff=coeff,ps=ps,test_mode=True)
        # plt.imshow(abs(img_real))
        # plt.colorbar()
        # plt.show()

        plt.imshow(img)
        plt.colorbar()
        plt.show()

        # plt.imshow(abs(img_im))
        # plt.colorbar()
        # plt.show()


    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # # 创建示例数据
    # x, y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    # z = np.sin(x) * np.cos(y)  # 包含正负值的函数
    #
    # plt.figure(figsize=(10, 8))
    #
    # # 最佳实践显示
    # vmax = max(abs(z.min()), abs(z.max()))
    # im = plt.imshow(z,
    #                 cmap='RdBu_r',  # 红蓝对称颜色映射
    #                 vmin=-vmax,  # 最小值
    #                 vmax=vmax,  # 最大值
    #                 extent=[-3, 3, -3, 3],  # 坐标范围
    #                 origin='lower')  # 原点在左下角
    #
    # plt.colorbar(im, label='数值')
    # plt.title('包含负数的热图显示 (最佳实践)')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    #
    # plt.show()
    # data = np.load("./Datasets/VGG-param1-500.npz")
    # imgs = data["imgs"]
    # for _ in range(1000):
    #     plt.imshow(imgs[_])
    #     plt.show()
    #     plt.close()
    #     input("Press Enter to continue...")



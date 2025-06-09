import tensorflow as tf

def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print('Hi, {}'.format(name))# 按 F9 切换断点。
    # print("Hi, name")
    return name

# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    x = 123
    print_hi(x)
    gpu_available = tf.config.list_physical_devices('GPU')
    print("tensorflow version: ", tf.__version__)
    print("cuDNN version:", tf.sysconfig.get_build_info()['cudnn_version'])
    print("GPU available: ", gpu_available)
    # a = tf.constant([1.0,2.0], name="a")
    # b = tf.constant([3.0,4.0], name="b")
    a = tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]], name="a")
    b = tf.constant([[2.0,3.0,4.0],[5.0,6.0,7.0]], name="b")
    c = tf.constant([  [[1.0,2.0,3.0],[4.0,5.0,6.0]]
                      ,[[2.0,3.0,4.0],[5.0,6.0,7.0]]  ])
    result = tf.add(a, b)
    print(result)
    print(c)
    print('\"\"')



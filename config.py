# @Time : 2021-03-29 19:24
# @Author : Xuanhan Liu
# @Site : 
# @File : config.py
# @Software: PyCharm

class Config(object):
    image_size = 256
    batch_size = 8
    data_root = 'data/'
    num_workers = 2
    use_gpu = True
    style_path = 'mosaic.jpg'
    lr = 1e-3
    env = 'neural-style'  # visdom env
    epoches = 2
    content_weight = 1e5
    style_weight = 1e10

    model_path = None
    debug_file = None

    content_path = 'input3.jpg'
    result_path = 'result3.jpg'


config = Config

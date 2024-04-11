from ..model.segment import Dann, DenseMobileNet2D, DenseNet2D, DenseMobileNet2D_2, Ellseg

# Dictionary of models
MODEL_DICT = {}
MODEL_DICT['simple_unet'] = DenseNet2D
MODEL_DICT['u_mobilenet'] = DenseMobileNet2D
MODEL_DICT['u_mobilenet_2'] = DenseMobileNet2D_2
MODEL_DICT['ellseg'] = Ellseg
MODEL_DICT['dann'] = Dann
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset

# @DATASETS.register_module()
# class DDSMDataset(CustomDataset):  
#   CLASSES = ('normal', 'calc-beli', 'mass-beli', 'calc-mali', 'mass-mali')
#   PALETTE = [[[0],[1],[2],[3],[4]]] 
#   def __init__(self, **kwargs):
#     super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)
    # assert osp.exists(self.img_dir) and self.split is not None

@DATASETS.register_module()
class DDSMDataset(CustomDataset):  
  CLASSES = ('normal', 'calc', 'mass')
  PALETTE = [[[0],[1,3],[2,4]]] 
  def __init__(self, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)
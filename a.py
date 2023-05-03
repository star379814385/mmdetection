from mmcv import Config
from pathlib import Path
from mmdet.utils import replace_cfg_vals, update_data_root

if __name__ == "__main__":
    cfg_path = r"D:\code\mmdetection\extension\KNet_main\configs\det\knet\knet_s3_r50_fpn_1x_coco.py"
    save_dir = r"D:\code\mmdetection\full_configs"
    save_path = str(Path(save_dir) / Path(cfg_path).name)
    cfg = Config.fromfile(cfg_path)
    cfg = replace_cfg_vals(cfg)
    # update data root according to MMDET_DATASETS
    update_data_root(cfg)
    cfg.dump(save_path)
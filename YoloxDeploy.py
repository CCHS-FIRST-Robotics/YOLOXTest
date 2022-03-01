import torch
import torch.backends.cudnn as cudnn
import cv2

from yolox.core import launch
from yolox.exp import get_exp
from yolox.data import ValTransform
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger, postprocess


# -n  yolox-s -c {MODEL_PATH} -b 64 -d 1 --conf 0.001 -f

class YoloxDeploy:
    def __init__(self):
        self.rank = 0
        self.exp = get_exp(None, "yolox-s")
        self.model = self.exp.get_model()
        torch.cuda.set_device(self.rank)
        self.model.cuda(self.rank)
        self.model.eval()
        loc = "cuda:{}".format(self.rank)
        ckpt = torch.load("best_ckpt.pth.tar", map_location=loc)
        self.model.load_state_dict(ckpt["model"])
        self.preproc = ValTransform(
            rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def deploy(self, cv2img):
        imgs = self.preproc(cv2img)
        imgs = cv2.resize(imgs, (416, 416), interpolation = cv2.INTER_AREA)
        with torch.no_grad():
            imgs = imgs.type(torch.cuda.FloatTensor)
            outputs = self.model(imgs)
            outputs = postprocess(
                outputs, 2
            )
            print(outputs.shape)
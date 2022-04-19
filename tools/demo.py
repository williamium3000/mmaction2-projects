import re
from mmcv.runner import checkpoint
from mmcv.utils import config
from mmaction.apis.inference import init_detector, inference_recognizer
import os

root = "data/test"
config = "configs/washing_hands/timesformer_divST_8x32x1_15e_rgb_all.py"
checkpoint = "work_dirs/latest.pth"

device = "cuda:0"

videos = [os.path.join(root, img) for img in os.listdir(root)]
print("to inference: {}".format(videos))
model = init_detector(config=config, checkpoint=checkpoint, device=device)

res = inference_recognizer(model, videos)
print(res)


# ans = {}
# for p, r in zip(videos, res):
#     ans_i = []
#     for cls_id, bbox in enumerate(r):
#         if bbox.shape[0] > 0:
#             ans_i.append({"class":cls_id, "box":bbox})
#     ans[p] = ans_i
# print(ans)
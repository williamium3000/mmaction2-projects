import re
from mmcv.runner import checkpoint
from mmcv.utils import config
from mmaction.apis.inference import init_recognizer, inference_recognizer
import os

root = "data/wash_hands/2/correct_2"
config = "configs/washing_hands/timesformer_divST_8x32x1_15e_rgb_all.py"
checkpoint = "work_dirs/washing_hands/timesformer_divST_8x32x1_15e_rgb_all/latest.pth"

device = "cuda:0"

videos = [os.path.join(root, img) for img in os.listdir(root)]
print("to inference: {}".format(videos))
model = init_recognizer(config=config, checkpoint=checkpoint, device=device)

res = [inference_recognizer(model, video) for video in videos]

ans = {}
for p, r in zip(videos, res):
    ans[p] = {"index":r[0][0], "confidence":r[0][1]}    
print(ans)
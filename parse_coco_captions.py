import torch
import skimage.io as io
import clip
from clip.clip import tokenize
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse


def main(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/coco/oscar_split_{clip_model_name}_captions_train.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open('/mnt/SSD/datasets/COCO_4_ClipClap/train_caption.json', 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        d = data[i]
        img_id = d["image_id"]
        capt = d["caption"]
        capt_tensor = tokenize(capt).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_text(capt_tensor).cpu()
        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        all_captions.append(d)
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))

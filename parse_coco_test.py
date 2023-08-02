import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse


def main(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/coco/oscar_split_{clip_model_name}_test.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    all_embeddings = []
    all_imid = []
    for filena in os.listdir("/mnt/SSD/datasets/coco/data/test2014/"):
        img_id = filena[-16:-4]
        print(img_id)
        filename = "/mnt/SSD/datasets/coco/data/test2014/" + filena
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        all_embeddings.append(prefix)
        all_imid.append(img_id)
    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "img_id": all_imid}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))

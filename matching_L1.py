data_path = './data/coco/oscar_split_ViT-B_32_images_captions_onlytrain.pkl'
import pickle
import sys
import torch
with open(data_path, 'rb') as f:
    all_data = pickle.load(f)
print("Data size is %0d" % len(all_data["clip_im_embedding"]))
sys.stdout.flush()
prefixes_im = all_data["clip_im_embedding"]
prefixes_text = all_data["clip_text_embedding"]
from Cycle_consistency_training import Generator
from torch.nn.functional import cosine_similarity
L1 = torch.nn.L1Loss()

text_prefixes_translated = []
im_prefixes_translated = []
device = torch.device('cuda:0')
generator_text = Generator(512,512).to(device)
generator_text.load_state_dict(torch.load("./coco_train/generator_text-019_id12_onlytrain.pt"))
generator_im = Generator(512,512).to(device)
generator_im.load_state_dict(torch.load("./coco_train/generator_im-019_id12_onlytrain.pt"))

print(len(prefixes_im))
print(yolo)

for i in range(len(prefixes_im)):
    prefix_im = prefixes_im[i].to(device)
    prefix_text = prefixes_text[i].to(device)
    prefix_im = prefix_im.float()
    prefix_im = prefix_im / prefix_im.norm(2, -1)
    prefix_text = prefix_text.float()
    prefix_text = prefix_text / prefix_text.norm(2, -1)
    text_prefixes_translated.append(generator_im(prefix_text).cpu())
    im_prefixes_translated.append(generator_text(prefix_im).cpu())
    
score_matching_im_clip = 0
score_matching_cap_clip = 0
score_matching_im_cycle = 0
score_matching_cap_cycle = 0
for i in range(len(prefixes_im)):
    prefix_im = prefixes_im[i].unsqueeze(dim=0)
    prefix_text = prefixes_text[i].unsqueeze(dim=0)
    prefix_im = prefix_im.float()
    prefix_im = prefix_im / prefix_im.norm(2, -1)
    prefix_text = prefix_text.float()
    prefix_text = prefix_text / prefix_text.norm(2, -1)
    text_prefix_translated = text_prefixes_translated[i].unsqueeze(dim=0)
    im_prefix_translated = im_prefixes_translated[i].unsqueeze(dim=0)
    
    matching_caption_clip = prefix_text
    matching_image_clip = prefix_im
    matching_caption_cycle = text_prefix_translated
    matching_image_cycle = im_prefix_translated
        
        
    for j in range(len(prefixes_im)):
        prefix_im_bis = prefixes_im[j].unsqueeze(dim=0)
        prefix_text_bis = prefixes_text[j].unsqueeze(dim=0)
        prefix_im_bis = prefix_im_bis.float()
        prefix_im_bis = prefix_im_bis / prefix_im_bis.norm(2, -1)
        prefix_text_bis = prefix_text_bis.float()
        prefix_text_bis = prefix_text_bis / prefix_text_bis.norm(2, -1)
        text_prefix_translated_bis = text_prefixes_translated[j].unsqueeze(dim=0)
        im_prefix_translated_bis = im_prefixes_translated[j].unsqueeze(dim=0)
        a=False
        b=False
        c=False
        d=False
        
        if L1(prefix_text,prefix_im_bis)<L1(prefix_text,matching_image_clip) :
            matching_image_clip = prefix_im_bis
            a=True
                
        if L1(prefix_im,prefix_text_bis)<L1(prefix_im,matching_caption_clip) :
            matching_caption_clip = prefix_text_bis
            b=True
            
        if L1(prefix_text,im_prefix_translated_bis)<L1(prefix_text,matching_image_cycle) :
            matching_image_cycle = im_prefix_translated_bis
            c=True
            
        if L1(prefix_im,text_prefix_translated_bis)<L1(prefix_im,matching_caption_cycle) :
            matching_caption_cycle = text_prefix_translated_bis
            d=True
            
        if a and b and c and d:
            break
                
                
    if L1(matching_image_clip,prefix_im)==0:
        score_matching_im_clip +=1
    if L1(matching_caption_clip,prefix_text)==0:
        score_matching_cap_clip += 1
    if L1(matching_image_cycle,im_prefix_translated_bis)==0:
        score_matching_im_cycle +=1
    if L1(matching_caption_cycle,text_prefix_translated_bis)==0:
        score_matching_cap_cycle += 1
    print(i)
    print(score_matching_cap_cycle)
    print(score_matching_cap_clip)
    print(score_matching_im_cycle)
    print(score_matching_im_clip)
    print("\n")
        
score_matching_im_clip = score_matching_im_clip/len(prefixes_im)
score_matching_text_clip = score_matching_text_clip/len(prefixes_im)
score_matching_im_cycle = score_matching_im_cycle/len(prefixes_im)
score_matching_text_cycle = score_matching_text_cycle/len(prefixes_im)
print("score_matching_im_clip", score_matching_im_clip)
print("score_matching_text_clip", score_matching_text_clip)
print("score_matching_im_cycle", score_matching_im_cycle)
print("score_matching_text_cycle",score_matching_text_cycle)
        
    
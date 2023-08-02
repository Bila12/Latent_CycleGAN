import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
import statistics


from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

ex = Experiment(name="Cycle_CLIP_COCO")  # Create a sacred experiment
ex.captured_out_filter = apply_backspaces_and_linefeeds  # if tqdm is used
#ex.add_config("./cfg/bird_attn2.yaml")

ex.captured_out_filter = lambda captured_output: "Output capturing turned off."

DEBUG = 0
if not DEBUG:
    observer = MongoObserver(url="127.0.0.1:27017",
                             db_name="COCO_CLIP_experiment")
    ex.observers.append(observer)

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.caption2imembedding)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        prefix_im = self.prefixes_im[self.caption2imembedding[item]]
        prefix_text = self.prefixes_text[self.caption2textembedding[item]]
        if self.normalize_prefix:
            prefix_im = prefix_im.float()
            prefix_im = prefix_im / prefix_im.norm(2, -1)
            prefix_text = prefix_text.float()
            prefix_text = prefix_text / prefix_text.norm(2, -1)
        return tokens, mask, prefix_im, prefix_text

    def __init__(self, data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_im_embedding"]))
        sys.stdout.flush()
        self.prefixes_im = all_data["clip_im_embedding"]
        self.prefixes_text = all_data["clip_text_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2imembedding, self.caption2textembedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2imembedding = []
            self.caption2textembedding = []
            max_seq_len = 0
            for caption in captions_raw:
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                self.caption2imembedding.append(caption["clip_im_embedding"])
                self.caption2textembedding.append(caption["clip_text_embedding"])
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2imembedding, self.caption2textembedding, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     clip_length, num_layers)


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


class Generator(nn.Module):

    def __init__(self, input_length: int, output_length: int):
        super(Generator, self).__init__()
        self.dense_layer_1 = nn.Linear(int(input_length), 512)
        self.dense_layer_2 = nn.Linear(512, 512)
        self.dense_layer_3 = nn.Linear(512, 512)
        self.dense_layer_4 = nn.Linear(512, output_length)
        self.activation = nn.Tanh()
        self.model = nn.Sequential(self.dense_layer_1, self.activation, self.dense_layer_2, self.activation, self.dense_layer_3, self.activation, self.dense_layer_4)#nn.Sequential(self.dense_layer_1, self.activation, self.dense_layer_2, self.activation, self.dense_layer_3)
        #self.identity_init()
        

    def forward(self, x):
        return self.model(x)
        
    def identity_init(self):
            '''self.dense_layer_1.weight.data.copy_(torch.eye(200))
            self.dense_layer_2.weight.data.copy_(torch.eye(200))'''
            nn.init.eye_(self.dense_layer_1.weight)
            nn.init.eye_(self.dense_layer_2.weight)
            
        
        
class Discriminator(nn.Module):
    def __init__(self, input_length: int):
        super(Discriminator, self).__init__()
        self.dense_1 = nn.Linear(int(input_length), 256)
        #self.dense_2 = nn.Linear(256,128)
        self.dense_2 = nn.Linear(256, 1)
        #self.dense_3 = nn.Linear(int(input_length/3), 1)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.end = nn.Sigmoid()
        self.model = nn.Sequential(self.dense_1, self.activation, self.dense_2)#nn.Sequential(self.dense_1, self.activation, nn.Dropout(0.3), self.dense_2)#

    def forward(self, x):
        return self.model(x)
        

def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model, parser


def train(_run, dataset: ClipCocoDataset, model: ClipCaptionModel, args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):

    device = torch.device('cuda:0')
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    generator_text = Generator(512,512).to(device)
    discriminator_text = Discriminator(512).to(device)
    generator_im = Generator(512,512).to(device)
    discriminator_im = Discriminator(512).to(device)
    
    # Optimizers
    generator_parameters = list(generator_text.model.parameters()) + list(generator_im.model.parameters())
    discriminator_parameters  = list(discriminator_text.model.parameters()) + list(discriminator_im.model.parameters())
    generator_optimizer = AdamW(generator_parameters, lr=lr)
    discriminator_optimizer = AdamW(discriminator_parameters, lr=lr) #torch.optim.RMSprop(discriminator_parameters, lr=0.0003)
    #optimizer = AdamW(model.parameters(), lr=lr)
    
    
    lambda_idt = 0.5
    lambda_text = 10
    lambda_im = lambda_text*10
    
    
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler_gen = get_linear_schedule_with_warmup(
        generator_optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    scheduler_disc = get_linear_schedule_with_warmup(
        discriminator_optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    
    criterionGAN = nn.BCEWithLogitsLoss()
    criterionId = torch.nn.L1Loss()
    criterionCycle = torch.nn.L1Loss()
    # save_config(args)
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        accuracies_disc = []
        for idx, (tokens, mask, prefix_im, prefix_text) in enumerate(train_dataloader):
            
            tokens, mask, prefix_im, prefix_text = tokens.to(device), mask.to(device), prefix_im.to(device, dtype=torch.float32), prefix_text.to(device, dtype=torch.float32)
            
            
            
            #####################################################################################################################################
            
            
            #####LOSS GENERATOR######
            generator_optimizer.zero_grad()
            #if idx%100000000==0:
            fake_text = generator_text(prefix_im)  # G_A(A)
            rec_im = generator_im(fake_text)   # G_B(G_A(A))
            fake_im = generator_im(prefix_text)  # G_B(B)
            rec_text = generator_text(fake_im)   # G_A(G_B(B))
            for net in [generator_im, generator_text]:
                if net is not None:
                    for param in net.parameters():
                        param.requires_grad = True
            for net in [discriminator_im, discriminator_text]:
                if net is not None:
                    for param in net.parameters():
                        param.requires_grad = False
                    
            
            # Identity loss
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            idt_text = generator_text(prefix_text)
            loss_idt_text = criterionId(idt_text, prefix_text) 
            _run.log_scalar("Text_generator_identity_loss", loss_idt_text.cpu().detach().numpy().tolist(), idx)
            #loss_idt_text = loss_idt_text * lambda_text * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            idt_im = generator_im(prefix_im)
            loss_idt_im = criterionId(idt_im, prefix_im) 
            _run.log_scalar("Im_generator_identity_loss", loss_idt_im.cpu().detach().numpy().tolist(), idx)
            #loss_idt_im = loss_idt_im * lambda_im * lambda_idt
            
            # GAN loss D_A(G_A(A))
            loss_G_text = criterionGAN(discriminator_text(fake_text), torch.transpose(torch.Tensor([[True]*batch_size]),0,1).to(device))
            # GAN loss D_B(G_B(B))
            loss_G_im = criterionGAN(discriminator_im(fake_im), torch.transpose(torch.Tensor([[True]*batch_size]),0,1).to(device))
            # Forward cycle loss || G_B(G_A(A)) - A||
            loss_cycle_text = criterionCycle(rec_text, prefix_text) 
            _run.log_scalar("Text_cycle_loss", loss_cycle_text.cpu().detach().numpy().tolist(), idx)
            #loss_cycle_text = loss_cycle_text * lambda_text
            # Backward cycle loss || G_A(G_B(B)) - B||
            loss_cycle_im = criterionCycle(rec_im, prefix_im)
            _run.log_scalar("Im_cycle_loss", loss_cycle_im.cpu().detach().numpy().tolist(), idx)
            #loss_cycle_im = loss_cycle_im * lambda_im
            # combined loss and calculate gradients
            loss_G = loss_G_text + loss_G_im*2 + loss_cycle_text * lambda_text + loss_cycle_im * lambda_im + loss_idt_text * lambda_text * lambda_idt + loss_idt_im* lambda_im * lambda_idt
            loss_G.backward()
            generator_optimizer.step()
            #generator_optimizer.zero_grad()
            #_run.log_scalar("Text_generator_loss", loss_G_text.cpu().detach().numpy().tolist(), idx)
            
            
            
            
            
            #####################################################################################################################################
            
            
            ###LOSS DISCRIMINATOR####
            if idx%12==0:
                continue
            else:
                for net in [discriminator_im, discriminator_text]:
                    if net is not None:
                        for param in net.parameters():
                            param.requires_grad = True
                for net in [generator_im, generator_text]:
                    if net is not None:
                        for param in net.parameters():
                            param.requires_grad = False
                
                fake_text = generator_text(prefix_im)  # G_A(A)
                rec_im = generator_im(fake_text)   # G_B(G_A(A))
                fake_im = generator_im(prefix_text)  # G_B(B)
                rec_text = generator_text(fake_im)
                            
                discriminator_optimizer.zero_grad()
                pred_real_im = discriminator_im(prefix_im)
                loss_D_real_im = criterionGAN(pred_real_im, torch.transpose(torch.Tensor([[True]*batch_size]),0,1).to(device))
                # Fake
                pred_fake_im = discriminator_im(fake_im)
                loss_D_fake_im = criterionGAN(pred_fake_im, torch.transpose(torch.Tensor([[False]*batch_size]),0,1).to(device))
                
                # Combined loss and calculate gradients
                loss_D_im = (loss_D_real_im + loss_D_fake_im) * 0.5
                
                
                pred_real_text = discriminator_text(prefix_text)
                loss_D_real_text = criterionGAN(pred_real_text, torch.transpose(torch.Tensor([[True]*batch_size]),0,1).to(device))
                # Fake
                pred_fake_text = discriminator_text(fake_text)
                loss_D_fake_text = criterionGAN(pred_fake_text, torch.transpose(torch.Tensor([[False]*batch_size]),0,1).to(device))
                # Combined loss and calculate gradients
                loss_D_text = (loss_D_real_text + loss_D_fake_text) * 0.5
                loss_D = loss_D_text + loss_D_im
                loss_D.backward()
                discriminator_optimizer.step()
                #####################################################################################################################################
                scheduler_gen.step()
                scheduler_disc.step()
                #print(pred_fake_text)
                _run.log_scalar("true_discriminator_text_loss", loss_D_real_text.cpu().detach().numpy().tolist(), idx)
                _run.log_scalar("true_discriminator_im_loss", loss_D_real_im.cpu().detach().numpy().tolist(), idx)
                _run.log_scalar("fake_discriminator_text_loss", loss_D_fake_text.cpu().detach().numpy().tolist(), idx)
                _run.log_scalar("fake_discriminator_im_loss", loss_D_fake_im.cpu().detach().numpy().tolist(), idx)
                
                pred = []
                
                for t in nnf.sigmoid(pred_fake_text.cpu().detach()):
                    #print(t)
                    #print(round(t[0].item()))
                    #print(round(t[0].item())==False)
                    pred.append(round(t[0].item())==False)
                _run.log_scalar("fake_discriminator_text_acc", statistics.mean(pred), idx)
                #print(statistics.mean(pred))
                accuracies_disc.append(statistics.mean(pred))
                
                pred = []
                
                for t in nnf.sigmoid(pred_real_text.cpu().detach()):
                    #print(t)
                    pred.append(round(t[0].item())==True)
                accuracies_disc.append(statistics.mean(pred))
                _run.log_scalar("true_discriminator_text_acc", statistics.mean(pred), idx)
                
                    
                for t in nnf.sigmoid(pred_fake_im.cpu().detach()):
                    #print(t)
                    #print(round(t[0].item()))
                    #print(round(t[0].item())==False)
                    pred.append(round(t[0].item())==False)
                _run.log_scalar("fake_discriminator_im_acc", statistics.mean(pred), idx)
                accuracies_disc.append(statistics.mean(pred))
                #print(statistics.mean(pred))
                
                
                pred = []
                
                for t in nnf.sigmoid(pred_real_im.cpu().detach()):
                    #print(t)
                    pred.append(round(t[0].item())==True)
                _run.log_scalar("true_discriminator_im_acc", statistics.mean(pred), idx)
                accuracies_disc.append(statistics.mean(pred))
                
                if len(accuracies_disc) > 100:
                    accuracies_disc = [0.5]
                
                
            
            #print(accuracies_disc)
            
            
            progress.set_postfix({"acc": statistics.mean(accuracies_disc)})
            
           
            progress.update()
            if (idx + 1) % 10000 == 0:
                torch.save(
                    generator_im.state_dict(),
                    os.path.join(output_dir, f"generator_im_latest.pt"),
                )
                torch.save(
                    generator_text.state_dict(),
                    os.path.join(output_dir, f"generator_text_latest.pt"),
                )
                torch.save(
                    discriminator_im.state_dict(),
                    os.path.join(output_dir, f"discriminator_im_latest.pt"),
                )
                torch.save(
                    discriminator_text.state_dict(),
                    os.path.join(output_dir, f"discriminator_text_latest.pt"),
                )
        progress.close()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                    generator_im.state_dict(),
                    os.path.join(output_dir, f"generator_im-{epoch:03d}_id12_onlytrain.pt"),
                )
            torch.save(
                    generator_text.state_dict(),
                    os.path.join(output_dir, f"generator_text-{epoch:03d}_id12_onlytrain.pt"),
                )
            torch.save(
                    discriminator_im.state_dict(),
                    os.path.join(output_dir, f"discriminator_im-{epoch:03d}_id12_onlytrain.pt"),
                )
            torch.save(
                    discriminator_text.state_dict(),
                    os.path.join(output_dir, f"discriminator_text-{epoch:03d}_id12_onlytrain.pt"),
                )
            
    return model

@ex.automain
def main(_run):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/coco/oscar_split_ViT-B_32_images_captions_onlytrain.pkl')
    parser.add_argument('--out_dir', default='./coco_train/')
    parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=40)
    parser.add_argument('--prefix_length_clip', type=int, default=40)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--only_prefix', type=bool,default=True)
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    args = parser.parse_args()
    prefix_length = args.prefix_length
    dataset = ClipCocoDataset(args.data, prefix_length, normalize_prefix=args.normalize_prefix)
    prefix_dim = 640 if args.is_rn else 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train only prefix")
    else:
        model = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train both prefix and GPT")
        sys.stdout.flush()
    train( _run, dataset, model, args, output_dir=args.out_dir, output_prefix=args.prefix)




#if __name__ == '__main__':
 #   main()

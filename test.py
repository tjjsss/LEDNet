import os
import clip
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, average_precision_score
import random
from PIL import Image, UnidentifiedImageError
from PIL.Image import BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import argparse
from tqdm import tqdm  # 导入 tqdm
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None
    images, labels, paths = zip(*batch)
    return torch.stack(images), torch.tensor(labels)

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

preprocess_ = _transform(224)

def load_text_dataset(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                image_path = parts[0]
                label = int(parts[1])
                data.append((image_path, label))
    random.shuffle(data)
    return data

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        try:
            image = Image.open(image_path)
            image = preprocess_(image)
            return image, label, image_path
        except (IOError, UnidentifiedImageError, SyntaxError) as e:
            print(f"Error processing image {image_path}: {e}")
            return None
class PROMPTLEARNER(nn.Module):
    def __init__(self, prompt_dim=768, num_heads=8, num_layers=2):
        super(PROMPTLEARNER, self).__init__()
        self.clip_model, preprocess = clip.load("ViT-L/14", device='cpu', jit=False)
        
        self.clip_model.eval()
        self.text_input = clip.tokenize(['Real Photo or painting', 'Fake Photo or painting'])
        
        self.fc = nn.Linear(prompt_dim, 1)  
        self.prompt_vector = nn.Parameter(torch.randn(1, prompt_dim).float())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.requires_grad_(False)
        
       
        encoder_layer = nn.TransformerEncoderLayer(d_model=prompt_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, image_input):
        image_input = image_input.to(self.device)
        
      
        image_features = self.clip_model.encode_image(image_input)  
        #print(image_features.shape)
        text_features = self.clip_model.encode_text(self.text_input.to(self.device))  
        
        
        text_features = text_features + self.prompt_vector.to(self.device)
        text_features = text_features.unsqueeze(0)
        text_features= text_features.repeat(image_features.shape[0], 1, 1)
        #print(text_features)

        #image_features = image_features.unsqueeze(1).expand(-1, text_features.size(0), -1)

        
       
        combined_features = torch.cat((image_features,text_features),dim=1)
        #print(combined_features.shape)
        
       
        attn_output = self.transformer_encoder(combined_features.permute(1, 0, 2)) 
        attn_output = attn_output.permute(1, 0, 2)  
        
        
        final_features = attn_output[:, 0, :]
        
       
        output = torch.sigmoid(self.fc(final_features))
        return output


def load_test_data(test_dir):
    test_sets = {'table_1': {}, 'table_2': {}, 'table_3': {}, 'table_4': {}}
    table_1_datasets = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
    table_2_datasets = ['AttGAN', 'BEGAN', 'CramerGAN', 'InfoMaxGAN', 'MMDGAN', 'RelGAN', 'S3GAN', 'SNGAN', 'STGAN']
    
    table_3_datasets = ['dalle', 'glide_100_10', 'glide_100_27', 'glide_50_27', 'adm', 'ldm_100', 'ldm_200', 'ldm_200_cfg']

    for entry in os.scandir(test_dir):
        if entry.is_dir():
            subdirs = os.listdir(entry.path)
            if "0_real" in subdirs and "1_fake" in subdirs:
                test_data = []
                for label_dir in ["0_real", "1_fake"]:
                    label = 0 if label_dir == "0_real" else 1
                    label_dir_path = os.path.join(entry.path, label_dir)
                    for img_file in os.listdir(label_dir_path):
                        img_path = os.path.join(label_dir_path, img_file)
                        test_data.append((img_path, label))
                if entry.name in table_1_datasets:
                    test_sets['table_1'][entry.name] = test_data
                elif entry.name in table_2_datasets:
                    test_sets['table_2'][entry.name] = test_data
                elif entry.name in table_3_datasets:
                    test_sets['table_3'][entry.name] = test_data
               
            else:
                test_data = []
                for sub_entry in os.scandir(entry.path):
                    if sub_entry.is_dir():
                        subdirs = os.listdir(sub_entry.path)
                        if "0_real" in subdirs and "1_fake" in subdirs:
                            for label_dir in ["0_real", "1_fake"]:
                                label = 0 if label_dir == "0_real" else 1
                                label_dir_path = os.path.join(sub_entry.path, label_dir)
                                for img_file in os.listdir(label_dir_path):
                                    img_path = os.path.join(label_dir_path, img_file)
                                    test_data.append((img_path, label))
                if test_data:
                    if entry.name in table_1_datasets:
                        test_sets['table_1'][entry.name] = test_data
                    elif entry.name in table_2_datasets:
                        test_sets['table_2'][entry.name] = test_data
                    elif entry.name in table_3_datasets:
                        test_sets['table_3'][entry.name] = test_data
                    
    return test_sets

def evaluate_model(model, test_set,agrs):
    
    model.eval()
    results = {}
    with torch.no_grad():
        for table, datasets in test_sets.items():
            for dataset_name, test_data in datasets.items():
                test_loader = DataLoader(CustomDataset(test_data), batch_size=args.batch_size, collate_fn=collate_fn)
                all_labels = []
                all_preds = []
                all_outputs=[]
                for batch_images, batch_labels in tqdm(test_loader, desc=f"Evaluating {dataset_name}"):
                    if batch_images is None or batch_labels is None:
                        continue
                    output = model(batch_images)
                    preds = (output > 0.5).cpu().numpy().astype(int)
                    all_labels.extend(batch_labels.cpu().numpy())
                    all_preds.extend(preds)
                    all_outputs.extend(output.cpu().numpy())
                accuracy = accuracy_score(all_labels, all_preds)
                ap = average_precision_score(all_labels, all_outputs)
                results[dataset_name] = (accuracy, ap)
               
    return results

def test(data_set,args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model =  PROMPTLEARNER()
    model.load_state_dict(torch.load(f'{args.model_path}'))
    model.to(device)
    model.eval()  
    with open("table_output/test.txt", "w") as f:
            

                    results = evaluate_model(model, test_sets,args)
                    f.write(f" Results: {results}\n")
                    f.flush()
                   
                    

                    table_means = []
                    for table in ['table_1', 'table_2',  'table_3']:
                        table_data = [acc for name, (acc, ap) in results.items() if name in test_sets[table]]
                        table_mean = sum(table_data) / len(table_data)
                        table_means.append(table_mean)
                        f.write(f"Table {table}, Mean Accuracy: {table_mean:.4f}\n")
                        f.flush()
                        table_data_ap = [ap for name, (acc, ap) in results.items() if name in test_sets[table]]
                        table_mean_ap = sum(table_data_ap) / len(table_data_ap)
                        f.write(f"Table {table}, Mean AP: {table_mean_ap:.4f}\n")
                        f.flush()
            

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for testing")
    parser.add_argument("--test_data_path", type=str, default="/home/rstao/Tangshijie/data/68_test/test", help="Path to test data directory")
    parser.add_argument("--model_path", type=str, default="/home/rstao/Tangshijie/lab/deepfake/image_text/ap/models/best_model.pth", help="Path to model")

    args = parser.parse_args()
    test_sets = load_test_data(args.test_data_path)
    test(test_sets,args)

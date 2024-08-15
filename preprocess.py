import os
import random

def generate_dataset_txt(base_path, categories):
    data = []
    total_images = 0

    for category in categories:
        category_path = os.path.join(base_path, category)
        
        fake_path = os.path.join(category_path, "1_fake")
        real_path = os.path.join(category_path, "0_real")
        
        if os.path.exists(fake_path):
            for img_name in os.listdir(fake_path):
                if img_name.endswith('.png'):
                    img_path = os.path.join(fake_path, img_name)
                    data.append(f"{img_path} 1")
                    total_images += 1

        if os.path.exists(real_path):
            for img_name in os.listdir(real_path):
                if img_name.endswith('.png'):
                    img_path = os.path.join(real_path, img_name)
                    data.append(f"{img_path} 0")
                    total_images += 1

    random.shuffle(data)

    output_file = f'annotation/dataset_4train_{total_images}.txt'
    with open(output_file, 'w') as f:
        for line in data:
            f.write(f"{line}\n")
    
    return output_file

base_path = '/home/rstao/Tangshijie/data/68_test/train'
categories = ['car', 'cat', 'chair', 'horse']

output_file = generate_dataset_txt(base_path, categories)
print(f"output file: {output_file}")

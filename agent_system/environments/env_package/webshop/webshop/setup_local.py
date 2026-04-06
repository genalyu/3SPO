import gdown
import os

# 目标保存目录
save_dir = "/Users/a1-6/webshop_data"
os.makedirs(save_dir, exist_ok=True)

# WebShop 核心数据文件列表 (来自 setup.sh)
files = {
    "items_shuffle_1000.json": "1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib",
    "items_ins_v2_1000.json": "1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu",
    "items_shuffle.json": "1A2whVgOO0euk5O13n2iYDM0bQRkkRduB",
    "items_ins_v2.json": "1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi",
    "items_human_ins.json": "14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O"
}

for filename, file_id in files.items():
    url = f'https://drive.google.com/uc?id={file_id}'
    output = os.path.join(save_dir, filename)
    print(f"Downloading {filename}...")
    gdown.download(url, output, quiet=False)

print("Download complete. Please upload the contents of './webshop_data' to your server's 'agent_system/environments/env_package/webshop/webshop/data/' directory.")
from torch_geometric.data import Dataset
from parse_dot import parse_dot_file_enhanced, CodeGraphEncoder
import os
import torch
import re
import pydot
from typing import List, Set
from tqdm.auto import tqdm

class HCPGDataset(Dataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.dot_dir = os.path.join("dots-cpg")
        
        # 检查目录是否存在
        if not os.path.exists(self.dot_dir):
            raise ValueError(f"Dataset directory not found: {self.dot_dir}")
        
        dot_files = os.listdir(self.dot_dir)
        self.dot_files = sorted(dot_files, key=lambda x: int(re.search(r'(\d+)', x).group()))

        self.encoder = CodeGraphEncoder()

        # 解析标签
        self.labels = []
        for file_path in self.dot_files:
            # 获取文件名
            file_name = os.path.basename(file_path)
            # 从文件名中提取标签, 文件名格式为: "序号-cpg-标签.dot"
            try:
                # 去掉扩展名
                name_without_ext = os.path.splitext(file_name)[0]
                
                # 按"-"分割并获取最后一部分作为标签
                parts = name_without_ext.split("-")
                if len(parts) >= 3:  # 确保格式正确
                    label = int(parts[-1])  # 转换为整数
                    self.labels.append(label)
                else:
                    print(f"警告: 文件名格式不符合预期 - {file_name}")
            except ValueError:
                print(f"警告: 无法从文件名中提取标签 - {file_name}")
                
        # 初始化编码器（只初始化一次）
        # print("Loading CodeBERT model...")
        # print("CodeBERT model loaded successfully!")


    def encode_dot_to_feature(self):
        """
        解析DOT文件并转换为图数据
        :param dot_path: DOT文件路径
        :param label: 标签
        :return: 图数据对象
        """
        dataset = []  # 用于存储解析后的图数据
        # 遍历所有DOT文件
        for dot_file_path, dot_file_label in zip(self.dot_files, self.labels):
            path = os.path.join(self.dot_dir, dot_file_path)
            label = dot_file_label
            print("--------------------------------------------------")
            print(f"Processing file: {path} - Label: {label}")
            
            # 读取DOT文件为图数据对象
            graphs = pydot.graph_from_dot_file(path)
            if not graphs:
                print(f"警告: 无法读取DOT文件 {path}，跳过该文件。")
                continue
            graph = graphs[0]  # 取第一个图（通常只有一个图）

            # 解析DOT文件
            data = parse_dot_file_enhanced(graph, label, self.encoder)
            dataset.append(data)

        return dataset


if __name__ == "__main__":
    # 加载 dot 文件数据集
    print("Loading HCPG dataset...")
    HCPGDataset_obj = HCPGDataset()
    dataset = HCPGDataset_obj.encode_dot_to_feature()
    torch.save(dataset, 'hcpg_dataset.pkl')
    print("HCPG dataset saved to 'hcpg_dataset.pkl'")
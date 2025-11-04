import os
import re
import pydot
import torch
import html
from torch_geometric.data import Data
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

NODE_TYPE_MAP = {
    "METHOD": 0, "BLOCK": 1, "IDENTIFIER": 2, "LITERAL": 3,
    "CALL": 4, "METHOD_RETURN": 5, "CLASS": 6, "FILE": 7,
    "PARAM": 8, "LOCAL": 9, "FIELD_IDENTIFIER": 10, "MODIFIER": 11,
    "<operator>.assignment": 12, "<operator>.indirectFieldAccess": 13
}
EDGE_TYPE_MAP = {
    "AST": 0, "CFG": 1, "DDG": 2, "CDG": 3
}

class CodeGraphEncoder(nn.Module):
    """
    结合CodeBERT和图神经网络的编码器
    """
    def __init__(self, 
                 codebert_model_name='microsoft/codebert-base',
                 node_type_embed_dim=64,
                 edge_type_embed_dim=32,
                 hidden_dim=256):
        super().__init__()
        
        # CodeBERT用于编码代码文本
        self.tokenizer = AutoTokenizer.from_pretrained(codebert_model_name)
        self.codebert = AutoModel.from_pretrained(codebert_model_name)
        
        # 节点类型嵌入
        self.node_type_embedding = nn.Embedding(
            len(NODE_TYPE_MAP) + 1, node_type_embed_dim
        )
        
        # 边/超边类型嵌入
        self.edge_type_embedding = nn.Embedding(
            len(EDGE_TYPE_MAP) + 1, edge_type_embed_dim
        )
        
        # 小投影用于把 edge-type embedding 与流数值结合
        self.edge_attr_proj = nn.Sequential(
            nn.Linear(edge_type_embed_dim + 1, edge_type_embed_dim),
            nn.ReLU(),
            nn.Linear(edge_type_embed_dim, edge_type_embed_dim)
        )
        
        # 特征融合层
        codebert_dim = self.codebert.config.hidden_size  # 通常是768
        self.node_fusion = nn.Linear(
            codebert_dim + node_type_embed_dim, hidden_dim
        )
        
        self.dropout = nn.Dropout(0.1)
        
    def encode_code_text(self, code_texts, max_length=128):
        """
        使用CodeBERT编码代码文本
        """
        if not code_texts:
            return torch.zeros(1, self.codebert.config.hidden_size)
            
        # 处理批量文本
        inputs = self.tokenizer(
            code_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = self.codebert(**inputs)
            # 使用[CLS]token的表示
            embeddings = outputs.last_hidden_state[:, 0, :]
            
        return embeddings
    
    def forward(self, data):
        """
        前向：返回节点特征与超边初始嵌入（基于类型+流）
        """
        node_type_embeds = self.node_type_embedding(data.node_types.to(next(self.parameters()).device))
        code_embeds = data.x.to(next(self.parameters()).device)  # x: codebert embeddings
        
        # 融合节点类型和代码嵌入
        node_features = torch.cat([node_type_embeds, code_embeds], dim=1)
        node_features = self.node_fusion(node_features)
        node_features = F.relu(node_features)
        
        # 构造超边嵌入（如果存在）
        if hasattr(data, 'edge_attr') and data.edge_attr.numel() > 0:
            # data.edge_attr: [num_hyperedges, attr_dim], 这里 attr_dim = 3 (edge_type, has_flow, flow_count)
            edge_types = data.edge_attr[:, 0].long().to(next(self.parameters()).device)
            flow_counts = data.edge_attr[:, 2].float().unsqueeze(1).to(next(self.parameters()).device)
            edge_type_emb = self.edge_type_embedding(edge_types)
            edge_emb_input = torch.cat([edge_type_emb, flow_counts], dim=1)
            edge_embeds = self.edge_attr_proj(edge_emb_input)
        else:
            edge_embeds = torch.empty(0, self.edge_type_embedding.embedding_dim, device=next(self.parameters()).device)
        
        return node_features, edge_embeds

def parse_node_label(raw_label):
    """
    解析节点标签，提取节点类型和节点值
    """
    if not raw_label or len(raw_label) < 2:
        return "UNKNOWN", ""
        
    try:
        # 去掉最外层的引号和尖括号
        inner_content = raw_label.strip('"<>')
        
        # 按<BR/>分割
        parts = inner_content.split('<BR/>')
        if len(parts) >= 2:
            # 处理第一部分获取节点类型
            first_part = parts[0]
            node_type = first_part.split(',')[0].strip()
            
            # 处理第二部分获取的节点类型和节点值并还原转义字符
            node_type = html.unescape(node_type)
            node_value = html.unescape(parts[1])
            return node_type, node_value
        else:
            # 如果没有<BR/>，尝试其他格式
            if ',' in inner_content:
                node_type = inner_content.split(',')[0].strip()
                return node_type, inner_content
            else:
                return "UNKNOWN", inner_content
                
    except Exception as e:
        print(f"Error parsing label {raw_label}: {e}")
        return "UNKNOWN", ""

def _parse_ddg_flow(raw_label):
    """
    从 DDG 标签中提取花括号内的映射/值，返回原始字符串及计数
    """
    m = re.search(r'\{(.*)\}', raw_label)
    if not m:
        return "", 0
    inner = m.group(1).strip()
    if inner == "":
        return "", 0
    # 用逗号分割条目，忽略空项
    parts = [p.strip() for p in inner.split(',') if p.strip()]
    return inner, len(parts)

def parse_dot_file_enhanced(graph, label, encoder_model):
    """
    解析DOT文件并生成图数据（超图：incidence 表示）
    - hyperedge incidence: tensor([node_idx_list, hyperedge_id_list])
    - edge_attr: per-hyperedge attributes: [edge_type_id, has_flow, flow_count]
    - incidence_role: 对应每个 incidence 的角色 0=source 1=target
    - hyperedge_flow: 原始 flow 字符串列表（Python list）
    """
    try:
        nodes = graph.get_nodes()
        edges = graph.get_edges()

        node_types = []
        code_texts = []
        node_id_map = {}

        # 解析节点
        for idx, node in enumerate(nodes):
            node_id = node.get_name().strip('"')
            attrs = node.get_attributes()
            raw_label = attrs.get("label", "")
            
            if not raw_label:
                node_type, code_text = "UNKNOWN", ""
            else:
                node_type, code_text = parse_node_label(raw_label)
            
            type_id = NODE_TYPE_MAP.get(node_type, len(NODE_TYPE_MAP))
            node_types.append(type_id)
            code_texts.append(code_text if code_text else "")
            node_id_map[node_id] = idx

        # 如果没有节点，创建空图
        if not node_types:
            return Data(
                x=torch.zeros(1, 768),  # 空图的默认特征
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 3, dtype=torch.long),
                y=torch.tensor([label], dtype=torch.long)
            )

        # 使用CodeBERT编码代码文本
        code_embeddings = encoder_model.encode_code_text(code_texts)

        # 解析边 -> 构造超边 incidence
        incidence_node_idx = []
        incidence_hyperedge_idx = []
        incidence_roles = []  # 0: source, 1: target
        hyperedge_attrs = []  # [edge_type_id, has_flow(0/1), flow_count]
        hyperedge_flow_raw = []

        hyperedge_counter = 0

        for edge in edges:
            src_raw = edge.get_source().strip('"')
            dst_raw = edge.get_destination().strip('"')
            raw_edge_label = edge.get_attributes().get("label", "AST")
            # 提取边类型字符串（如 "DDG", "AST" 等）
            edge_type_str = raw_edge_label.split(":")[0].strip().strip('"')
            edge_type_id = EDGE_TYPE_MAP.get(edge_type_str, len(EDGE_TYPE_MAP))

            # 解析 DDG 花括号内的流信息
            flow_raw, flow_count = _parse_ddg_flow(raw_edge_label)

            # 把源/目标按逗号分开（支持超边）
            src_nodes = [s.strip() for s in src_raw.split(',') if s.strip()]
            dst_nodes = [d.strip() for d in dst_raw.split(',') if d.strip()]

            # 如果任意一端节点在 node_id_map 中，则创建一个超边（directed）
            # 把所有 source 标记为 role=0，所有 target 标记为 role=1
            valid = False
            for s in src_nodes:
                if s in node_id_map:
                    valid = True
            for d in dst_nodes:
                if d in node_id_map:
                    valid = True
            if not valid:
                continue

            # 记录 incidence
            for s in src_nodes:
                if s in node_id_map:
                    incidence_node_idx.append(node_id_map[s])
                    incidence_hyperedge_idx.append(hyperedge_counter)
                    incidence_roles.append(0)
            for d in dst_nodes:
                if d in node_id_map:
                    incidence_node_idx.append(node_id_map[d])
                    incidence_hyperedge_idx.append(hyperedge_counter)
                    incidence_roles.append(1)

            # hyperedge 属性：类型、有无流、流计数
            has_flow = 1 if flow_count > 0 else 0
            hyperedge_attrs.append([edge_type_id, has_flow, flow_count])
            hyperedge_flow_raw.append(flow_raw)
            hyperedge_counter += 1

        # 转换为张量
        if incidence_node_idx:
            hyperedge_index = torch.tensor([incidence_node_idx, incidence_hyperedge_idx], dtype=torch.long)
            incidence_roles = torch.tensor(incidence_roles, dtype=torch.long)
            edge_attr = torch.tensor(hyperedge_attrs, dtype=torch.long)  # shape: [num_hyperedges, 3]
        else:
            hyperedge_index = torch.empty(2, 0, dtype=torch.long)
            incidence_roles = torch.empty(0, dtype=torch.long)
            edge_attr = torch.empty(0, 3, dtype=torch.long)
            hyperedge_flow_raw = []

        # 创建Data对象
        data = Data(
            x=code_embeddings,
            edge_index=hyperedge_index,            # incidence 表示: [node_idx_list, hyperedge_id_list]
            edge_attr=edge_attr,                   # per-hyperedge attrs
            y=torch.tensor([label], dtype=torch.long)
        )
        # 附加额外信息
        data.node_types = torch.tensor(node_types, dtype=torch.long)
        data.incidence_role = incidence_roles     # 每个 incidence 的 role：0=source,1=target
        data.hyperedge_flow = hyperedge_flow_raw  # 原始字符串（python list）
        data.num_hyperedges = edge_attr.size(0) if edge_attr.numel() > 0 else 0

        return data
        
    except Exception as e:
        print(f"Error parsing DOT file: {e}")
        # 返回空图
        return Data(
            x=torch.zeros(1, 768),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_attr=torch.empty(0, 3, dtype=torch.long),
            y=torch.tensor([label], dtype=torch.long)
        )

if __name__ == "__main__":
    # 初始化编码器
    print("Loading CodeBERT model...")
    encoder = CodeGraphEncoder()
    
    # 示例用法
    dot_file_path = "./dots-hcpg/2-hcpg-0.dot"
    label = 0
    
    graphs = pydot.graph_from_dot_file(dot_file_path)
    if graphs is None:
        print(f"Failed to load graph from {dot_file_path}")
        exit(1)
    graph = graphs[0]
    
    print("Parsing DOT file...")
    data = parse_dot_file_enhanced(graph, label, encoder)
    print("Data structure:")
    print(f"Number of nodes: {data.x.size(0)}")
    print(f"Node features shape: {data.x.shape}")
    print(f"Hyperedge incidence shape: {data.edge_index.shape}")
    print(f"Edge attributes shape (num_hyperedges x attr_dim): {data.edge_attr.shape}")
    print(f"Incidence roles length: {len(data.incidence_role)}")
    print(f"Label: {data.y}")
    print("Data parsed successfully!")
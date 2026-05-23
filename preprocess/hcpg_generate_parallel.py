from __future__ import annotations
from parse_dot import parse_dot_file_enhanced, CodeGraphEncoder
import os
import torch
import re
import pydot

# --------------------------
# pydot DOT helper utilities
# --------------------------

def _canon_node_id(x) -> str:
    """Canonicalize a node id across different DOT quoting styles."""
    if x is None:
        return ""
    return str(x).strip().strip('"')


def get_node_safe(graph: pydot.Dot, node_id):
    """Safely fetch a node object by id.

    pydot may store node ids with/without quotes, and nodes can also
    live inside subgraphs (clusters). This helper tries multiple forms
    and searches recursively.
    """
    if node_id is None:
        return None

    raw = str(node_id).strip()
    canon = _canon_node_id(raw)
    candidates = [raw, canon, f'"{canon}"']

    def _search(g: pydot.Dot):
        for key in candidates:
            try:
                ns = g.get_node(key)
            except Exception:
                ns = []
            if ns:
                return ns[0]
        for sg in getattr(g, 'get_subgraph_list', lambda: [])() or []:
            found = _search(sg)
            if found is not None:
                return found
        return None

    return _search(graph)


def _same_node(a, b) -> bool:
    return _canon_node_id(a) == _canon_node_id(b)

class HCPGDot():
    def __init__(self):
        self.dot_dir = os.path.join("dots-cpg")
        self.output_dir = os.path.join("dots-hcpg")
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


    def semantic_preserving_graph_pruning(self):
        """
        解析DOT文件并转换为图数据
        :param dot_path: DOT文件路径
        :param label: 标签
        :return: HCPG
        """
        # 遍历所有DOT文件
        for dot_file_path, dot_file_label in zip(self.dot_files, self.labels):
            path = os.path.join(self.dot_dir, dot_file_path)
            idx = dot_file_path.split("-")[0]
            label = dot_file_label
            print("--------------------------------------------------")
            print(f"Processing file: {path} - Label: {label}")
            
            # 读取DOT文件为图数据对象
            graphs = pydot.graph_from_dot_file(path)
            if not graphs:
                print(f"警告: 无法读取DOT文件 {path}，跳过该文件。")
                continue
            graph = graphs[0]  # 取第一个图（通常只有一个图）
            
            # 1.中间节点裁剪
            graph_inter_prun = remove_immediate_assignment_and_print_nodes(graph)
            # 2.叶子节点裁剪
            graph_leaf_prun = remove_and_merge_ast_leaf_nodes(graph_inter_prun)
            # 3.构建控制超边
            graph_ctl_hyperedges = build_control_hyperedges(graph_leaf_prun)
            # 4.构建数据超边
            HCPG = build_data_hyperedges(graph_ctl_hyperedges)
            # 保存 HCPG
            output_path = os.path.join(self.output_dir, f"{idx}-hcpg-{label}.dot")
            HCPG.write_raw(output_path)
            print(f"HCPG saved to {output_path}")
        return HCPG

## 判断叶子节点 ##
def is_ast_leaf_node(graph: pydot.Dot, node_name: str, node) -> bool:
    """
    判断一个节点是否是叶子节点, 且与其父节点之间仅存在 AST 边
    
    Args:
        graph: pydot.Dot 对象
        node_name: 节点名（字符串）
    
    Returns:
        bool: 满足条件返回 True, 否则 False
    """
    
    # 检查是否为叶子节点（没有出边）
    out_edges = []
    for edge in graph.get_edges():
        if _same_node(edge.get_source(), node_name):
            out_edges.append(edge)
    
    if len(out_edges) > 0:
        return False
    
    # 获取所有入边
    in_edges = []
    for edge in graph.get_edges():
        if _same_node(edge.get_destination(), node_name):
            in_edges.append(edge)
    
    # 如果没有入边，也不是AST叶子节点
    if len(in_edges) == 0:
        return False
    
    # 检查所有入边是否都是 AST 类型
    for edge in in_edges:
        label = edge.get_label()
        if not label or "AST: " not in str(label):
            return False
    
    return True

## 获取父节点 ##
def get_parent_nodes(graph: pydot.Dot, node_name: str):
        """获取节点的所有父节点"""
        parents = []
        for edge in graph.get_edges():
            if _same_node(edge.get_destination(), node_name):
                parents.append(edge.get_source())
        return parents

## 获取子节点 ##
def get_child_nodes(graph: pydot.Dot, node_name: str):
        """获取节点的所有子节点"""
        children = []
        for edge in graph.get_edges():
            if _same_node(edge.get_source(), node_name):
                children.append(edge.get_destination())
        return children

## 中间节点裁剪 ##
def remove_immediate_assignment_and_print_nodes(graph: pydot.Dot) -> pydot.Dot:
    """
    1.删除节点内容同时在其父子节点中出现的中间赋值节点（规则1）
    2.删除中间节点中的常量 print 节点（规则2）
    
    Args:
        graph: pydot.Dot 对象（将被修改）
    
    Returns:
        pydot.Dot: 修改后的图对象
    """
    # 收集要删除的节点
    nodes_to_remove = []

    for node in graph.get_nodes():  # node: "111669149696" [label = <METHOD, 1<BR/>&lt;global&gt;> ]
        node_name = node.get_name() # 获取节点序号名: node_name: "111669149696"
        label = node.get_label() # 获取节点label中的值 label: <METHOD, 1<BR/>&lt;global&gt;>
        node_type = str(label).split(",")[0].strip('<') # 获取节点类型 node_type: METHOD
        if node_type.startswith("&lt;operator&gt;.assignment"):
            #print(f"Processing node: {node.get_name()}")
            parent_nodes = get_parent_nodes(graph, node_name) # 获取父节点 列表
            # print(f"Parent nodes: {parent_nodes}")
            child_nodes = get_child_nodes(graph, node_name) # 获取子节点 列表
            # print(f"Child nodes: {child_nodes}")
            # 如果父子节点中都包含该节点，则删除该节点
            if len(parent_nodes) > 0 and len(child_nodes) > 0:
                for parent in parent_nodes:
                    parent_node = get_node_safe(graph, parent)  # 获取父节点对象（兼容引号/子图）
                    if parent_node is None:
                        continue
                    # print(f"Parent node: {parent_node}")
                    if str(parent_node.get_label()).split(",")[0].strip('<').startswith("BLOCK"):   # 判断父节点类型是否为 BLOCK
                        for child in child_nodes:
                            child_node = get_node_safe(graph, child)  # 获取子节点对象（兼容引号/子图）
                            if child_node is None:
                                continue
                            if str(child_node.get_label()).split(",")[0].strip('<').startswith("&lt;operator&gt;.alloc"): # 判断子节点类型是否为 alloc
                                nodes_to_remove.append(node)
        elif node_type.startswith("printf") or node_type.startswith("sprintf") or node_type.startswith("fprintf") or node_type.startswith("snprintf") or node_type.startswith("vprintf") or node_type.startswith("vsnprintf"):
            nodes_to_remove.append(node)  # 收集 printf 节点
                                
    # 执行节点删除操作
    for node in nodes_to_remove:
        node_name = node.get_name()
        node_type = str(node.get_label()).split(",")[0].strip('<')  # 获取节点类型
        # print(f"Removing node: {node_name}")
        parent_nodes = get_parent_nodes(graph, node_name) # 获取目标节点的 父节点 列表
        child_nodes = get_child_nodes(graph, node_name) # 获取目标节点的 子节点 列表

        # 删除该节点
        graph.del_node(node_name)
        # print(f"Removing node: {node_name}, {node}")
        
        # 移除涉及该节点的所有边
        edges_to_remove = []
        for edge in graph.get_edges():
            if (_same_node(edge.get_source(), node_name) or 
                _same_node(edge.get_destination(), node_name)):
                edges_to_remove.append(edge)
        for edge in edges_to_remove:
            graph.del_edge(edge.get_source(), edge.get_destination())
            # print(f"Removing edge: {edge.get_source()} -> {edge.get_destination()}")
        
        # 创建新边连接父节点和子节点
        if node_type.startswith("&lt;operator&gt;.assignment"):
            for parent in parent_nodes:
                parent_node = get_node_safe(graph, parent)  # 获取父节点对象（兼容引号/子图）
                if parent_node is None:
                    continue
                for child in child_nodes:
                    child_node = get_node_safe(graph, child)  # 获取子节点对象（兼容引号/子图）
                    if child_node is None:
                        continue
                    if str(child_node.get_label()).split(",")[0].strip('<').startswith("&lt;operator&gt;.alloc"): # 判断子节点类型是否为 alloc
                        graph.add_edge(pydot.Edge(parent, child, label="AST: "))
                        # print(f"Adding new edge: {parent} -> {child}")
                    else:
                        if str(parent_node.get_label()).split(",")[0].strip('<').startswith("BLOCK"):
                            graph.add_edge(pydot.Edge(parent, child, label="AST: "))
                            # print(f"Adding new edge: {parent} -> {child}")

        # printf类型的节点和相连的边直接删除，无需重建边

    return graph

## 叶子节点裁剪 ##
def remove_and_merge_ast_leaf_nodes(graph: pydot.Dot) -> pydot.Dot:
    """
    1.删除图中所有"与父节点仅存在 AST 边, 节点类型为 LITERAL 或 IDENTIFIER, 且自身为叶子节点"的节点和相关边 (规则3)
    2.(1)合并所有节点类型为 LOCAL/PARAM, 节点内容为同数据类型定义(如 char *ptr: char* 和 char *leak: char*)的节点, 并将这些节点的入边指向新合并的节点 (规则4) 
      (2)如果存在多个相同类型(如 char* )的 TYPE_REF/METHOD_RETURN 节点, 则仅保留一个节点, 删除的节点的入边指向保留的节点 (规则4)
    Args:
        graph: pydot.Dot 对象（将被修改）
    
    Returns:
        pydot.Dot: 修改后的图对象
    """
    
    # 收集要删除的节点
    nodes_to_delete = []
    
    # 收集要合并的节点集
    local_nodes_to_merge = []
    param_nodes_to_merge = []
    typeref_nodes_to_merge = []
    mreturn_nodes_to_merge = []
    
    for node in graph.get_nodes():
        node_name = node.get_name()
        if node_name == 'node':  # 跳过默认属性节点
            continue

        if is_ast_leaf_node(graph, node_name, node):
            # 检查节点类型是否为 LITERAL 或 IDENTIFIER
            label = node.get_label()
            node_type = str(label).split(",")[0].strip('<') # 获取节点类型
            if (node_type.startswith("LITERAL") or node_type.startswith("IDENTIFIER")):
            # print(f"Checking node_type: {node_type}")
                nodes_to_delete.append(node_name)
            elif node_type.startswith("LOCAL"):
                local_nodes_to_merge.append(node_name)
            elif node_type.startswith("PARAM"):
                param_nodes_to_merge.append(node_name)
            elif node_type.startswith("TYPE_REF"):
                typeref_nodes_to_merge.append(node_name)
            elif node_type.startswith("METHOD_RETURN"):
                mreturn_nodes_to_merge.append(node_name)
    
    # print(f"找到 {len(nodes_to_delete)} 个满足裁剪要求的AST叶子节点")
    # print(f"满足要求的节点: {nodes_to_delete}")
    
    # 删除叶子节点和相关边
    for node_name in nodes_to_delete:
        # print(f"删除节点: {node_name}")
        
        # 删除相关的边
        edges_to_remove = []
        for edge in graph.get_edges():
            if (_same_node(edge.get_source(), node_name) or 
                _same_node(edge.get_destination(), node_name)):
                edges_to_remove.append(edge)
        
        for edge in edges_to_remove:
            graph.del_edge(edge.get_source(), edge.get_destination())
        
        # 删除节点
        for node in graph.get_nodes():
            if node.get_name() == node_name:
                graph.del_node(node.get_name())
                break
    
    # 辅助函数：提取数据类型定义
    def extract_data_type(label_str):
        """从节点标签中提取数据类型定义"""
        # 示例: "LOCAL,char *ptr" -> "char*"
        # 示例: "PARAM,int value" -> "int"
        if ',' in label_str:
            parts = label_str.split(',', 1)
            if len(parts) > 1:
                var_def = parts[1].strip()
                # 提取类型部分 (变量名之前的部分)
                # 例如: "char *ptr" -> "char*", "int value" -> "int"
                tokens = var_def.split()
                if len(tokens) >= 2:
                    # 假设最后一个token是变量名，前面的是类型
                    type_tokens = tokens[:-1]
                    return ''.join(type_tokens).replace(' ', '')
                elif len(tokens) == 1:
                    # 只有一个token，可能是简单类型
                    return tokens[0]
        return None
    
    # 辅助函数：合并同类型节点
    def merge_nodes_by_type(nodes_list, node_type_prefix):
        """合并具有相同数据类型的节点"""
        if not nodes_list:
            return
            
        # 按数据类型分组
        type_groups = {}
        node_info = {}  # 存储节点信息
        
        for node_name in nodes_list:
            # 获取节点信息
            for node in graph.get_nodes():
                if node.get_name() == node_name:
                    label = str(node.get_label()).strip('"')
                    node_info[node_name] = (node, label)
                    
                    data_type = extract_data_type(label)
                    if data_type:
                        if data_type not in type_groups:
                            type_groups[data_type] = []
                        type_groups[data_type].append(node_name)
                    break
        
        # 对每个类型组进行合并
        for data_type, nodes_in_group in type_groups.items():
            if len(nodes_in_group) <= 1:
                continue  # 只有一个节点，无需合并
                
            # 保留第一个节点作为代表节点
            representative_node = nodes_in_group[0]
            nodes_to_merge = nodes_in_group[1:]
            
            # 将被合并节点的所有入边重定向到代表节点
            for node_to_merge in nodes_to_merge:
                # 收集入边
                incoming_edges = []
                for edge in graph.get_edges():
                    if _same_node(edge.get_destination(), node_to_merge):
                        incoming_edges.append(edge)
                
                # 重定向入边到代表节点
                for edge in incoming_edges:
                    source = edge.get_source()
                    edge_label = edge.get_label()
                    edge_attrs = edge.get_attributes()
                    
                    # 删除原边
                    graph.del_edge(edge.get_source(), edge.get_destination())
                    
                    # 创建新边指向代表节点
                    new_edge = pydot.Edge(source, representative_node)
                    if edge_label:
                        new_edge.set_label(edge_label)
                    for attr, value in edge_attrs.items():
                        new_edge.set(attr, value)
                    graph.add_edge(new_edge)
                
                # 删除被合并的节点
                graph.del_node(node_to_merge)
    
    # 辅助函数：合并相同类型的TYPE_REF和METHOD_RETURN节点
    def merge_same_type_nodes(nodes_list):
        """合并相同类型的节点（用于TYPE_REF和METHOD_RETURN）"""
        if not nodes_list:
            return
            
        # 按节点标签内容分组
        label_groups = {}
        
        for node_name in nodes_list:
            for node in graph.get_nodes():
                if node.get_name() == node_name:
                    label = str(node.get_label()).strip('"')
                    if label not in label_groups:
                        label_groups[label] = []
                    label_groups[label].append(node_name)
                    break
        
        # 对每个标签组进行合并
        for label_content, nodes_in_group in label_groups.items():
            if len(nodes_in_group) <= 1:
                continue  # 只有一个节点，无需合并
                
            # 保留第一个节点作为代表节点
            representative_node = nodes_in_group[0]
            nodes_to_merge = nodes_in_group[1:]
            
            # 将被合并节点的所有入边重定向到代表节点
            for node_to_merge in nodes_to_merge:
                # 收集入边
                incoming_edges = []
                for edge in graph.get_edges():
                    if _same_node(edge.get_destination(), node_to_merge):
                        incoming_edges.append(edge)
                
                # 重定向入边到代表节点
                for edge in incoming_edges:
                    source = edge.get_source()
                    edge_label = edge.get_label()
                    edge_attrs = edge.get_attributes()
                    
                    # 删除原边
                    graph.del_edge(edge.get_source(), edge.get_destination())
                    
                    # 创建新边指向代表节点
                    new_edge = pydot.Edge(source, representative_node)
                    if edge_label:
                        new_edge.set_label(edge_label)
                    for attr, value in edge_attrs.items():
                        new_edge.set(attr, value)
                    graph.add_edge(new_edge)
                
                # 删除被合并的节点
                graph.del_node(node_to_merge)
    
    # 合并所有具备相同数据类型定义的 LOCAL 节点
    merge_nodes_by_type(local_nodes_to_merge, "LOCAL")
    
    # 合并所有具备相同数据类型定义的 PARAM 节点
    merge_nodes_by_type(param_nodes_to_merge, "PARAM")
    
    # 存在多个相同类型的 TYPE_REF 节点, 保留一个
    merge_same_type_nodes(typeref_nodes_to_merge)
    
    # 存在多个相同类型的 METHOD_RETURN 节点, 保留一个
    merge_same_type_nodes(mreturn_nodes_to_merge)

    # print(f"删除了 {len(nodes_to_delete)} 个AST叶子节点")
    return graph

## 构建控制超边 ##
def build_control_hyperedges(graph: pydot.Dot) -> pydot.Dot:
    """
    构建控制超边 (Control Hyperedges)
    规则: 如果存在两个及以上节点同时与另一个目标节点存在控制依赖关系(CFG边),
    则合并这些节点到目标节点的控制流边，共同构成控制超边。
    
    在dot中表现形式:
    原始: v_1 -> target [ label = "CFG: " ]
         v_2 -> target [ label = "CFG: " ]
    转换为: v_1,v_2 -> target [ label = "CFG: " ]
    
    Args:
        graph: pydot.Dot 对象（将被修改）
    
    Returns:
        pydot.Dot: 修改后的图对象
    """
    
    # 第一步：建立目标节点 -> 源节点列表的映射（仅考虑 CFG 边）
    target_to_sources = {}
    
    for edge in graph.get_edges():
        edge_label = edge.get_label()
        
        # 检查边是否为 CFG 边
        if edge_label and "CFG: " in str(edge_label):
            source = edge.get_source()
            destination = edge.get_destination()
            
            if destination not in target_to_sources:
                target_to_sources[destination] = []
            
            target_to_sources[destination].append(source)
    
    # 第二步：找出所有需要构建控制超边的目标节点（有两个及以上源节点）
    hyperedges_to_create = {}
    for target, sources in target_to_sources.items():
        if len(sources) >= 2:
            hyperedges_to_create[target] = sorted(sources)  # 排序保证一致性
    
    # 第三步：删除原始的多条 CFG 边，创建超边
    edges_to_remove = []
    
    for target, sources in hyperedges_to_create.items():
        # 收集需要删除的边
        for edge in graph.get_edges():
            edge_label = edge.get_label()
            if (edge_label and "CFG: " in str(edge_label) and 
                edge.get_destination() == target and 
                edge.get_source() in sources):
                edges_to_remove.append(edge)
    
    # 删除这些边
    for edge in edges_to_remove:
        graph.del_edge(edge.get_source(), edge.get_destination())
    
    # 第四步：添加超边（用逗号分隔的源节点ID作为源）
    for target, sources in hyperedges_to_create.items():
        # 创建逗号分隔的源节点标识
        hyperedge_source = ",".join(sources)
        
        # 格式化源与目标为逗号分隔并用双引号包裹的单一标识符
        def mk_quoted(items_list):
            stripped = [x.strip('"') for x in items_list]
            return f'"{",".join(stripped)}"'

        hyperedge_source = mk_quoted(sources)
        
        # 添加超边
        hyperedge = pydot.Edge(hyperedge_source, target, label="CFG: ")
        graph.add_edge(hyperedge)

        print(f'Adding control hyperedge: {hyperedge_source} -> {target} label="CFG: "')

    return graph
    
## 构建数据超边 ##
def build_data_hyperedges(graph: pydot.Dot) -> pydot.Dot:
    """
    构建数据超边 (Data Hyperedges)
    规则：如果存在单个（或多个）节点同时数据依赖于另外两个（或多个）节点，则合并这些节点之间的数据依赖边，形成数据超边。
    额外规则：当一些边的 body 为类似 "var = val"，一些边为单一标识（如 "avbuf"），或一些边 body 为空（"DDG: "），
    合并时按优先级：
      1) 若解析出 var=val，则使用 {var:val,...} 形式；
      2) 否则使用原始 body 的集合，保留空 body（作为空项），并以 {v1,v2,} 形式展示（满足示例要求）。
    """
    import re
    # 收集所有 DDG 边并解析赋值信息，同时保留原始 body 字符串（可能为空）
    # ddg_edges: list of (src, dst, assigns_dict, raw_body, original_edge_obj)
    ddg_edges = []
    for edge in graph.get_edges():
        label = edge.get_label()
        if label and "DDG:" in str(label):
            src = edge.get_source()
            dst = edge.get_destination()
            label_str = str(label).strip().strip('"')
            m = re.search(r"DDG:\s*(.*)", label_str)
            assigns = {}
            raw_body = ""
            if m:
                body = m.group(1).strip()
                raw_body = body  # 可能为 "" 或 "avbuf" 或 "a = 1, b = 2"
                # 解析 var = val 形式
                for var_m in re.finditer(r"([\w\-]+)\s*=\s*([^,}]+)", body):
                    var = var_m.group(1).strip()
                    val = var_m.group(2).strip()
                    assigns[var] = val
            ddg_edges.append((src, dst, assigns, raw_body, edge))

    # 构建 target -> set(sources) 映射以及记录每对边的信息
    target_to_sources = {}
    pair_info = {}  # (src,dst) -> {'assigns':..., 'raw': raw_body}
    for src, dst, assigns, raw_body, edge in ddg_edges:
        target_to_sources.setdefault(dst, set()).add(src)
        pair_info[(src, dst)] = {'assigns': assigns, 'raw': raw_body}

    # 将具有相同 incoming source 集合的目标分组： key = tuple(sorted(sources)) -> list of targets
    targets_by_sources = {}
    for dst, sources in target_to_sources.items():
        key = tuple(sorted(sources))
        targets_by_sources.setdefault(key, []).append(dst)

    # 决定要创建超边的组：当 sources 数量 >=2 或 targets 数量 >=2 时创建超边
    hypergroups = []  # 每项为 (sources_list, targets_list)
    for sources_key, targets in targets_by_sources.items():
        sources = list(sources_key)
        if len(sources) >= 2 or len(targets) >= 2:
            hypergroups.append((sources, sorted(targets)))

    # 收集需要删除的原始边对（src,dst）
    pairs_to_remove = set()
    for sources, targets in hypergroups:
        for src in sources:
            for dst in targets:
                if (src, dst) in pair_info:
                    pairs_to_remove.add((src, dst))

    # 删除这些边（尝试带/不带引号）
    for src, dst in pairs_to_remove:
        try:
            graph.del_edge(src, dst)
        except Exception:
            s = src.strip('"')
            d = dst.strip('"')
            try:
                graph.del_edge(s, d)
            except Exception:
                pass

    # 创建超边并聚合赋值信息和原始标识信息（支持空 body 作为显式空项）
    for sources, targets in hypergroups:
        # 聚合变量赋值：var -> set(values)
        var_to_values = {}
        raw_set = set()
        empty_present = False
        for src in sources:
            for dst in targets:
                info = pair_info.get((src, dst), {})
                assigns = info.get('assigns', {}) or {}
                raw = info.get('raw', None)
                if raw is None:
                    raw = ""
                raw = raw.strip()
                # 聚合 var=val 情形
                for var, val in assigns.items():
                    var_to_values.setdefault(var, set()).add(val)
                # 若没有 var=val，则收集 raw（可能为空）
                if not assigns:
                    if raw == "":
                        empty_present = True
                    else:
                        # raw 可能包含逗号分隔多个项，拆分并收集
                        for part in [p.strip() for p in raw.split(',') if p.strip()]:
                            raw_set.add(part)

        # 构建标签：
        label = "DDG: {}"
        if var_to_values:
            items = []
            for var in sorted(var_to_values.keys()):
                vals = sorted(var_to_values[var])
                val_str = "|".join(vals) if len(vals) > 1 else vals[0]
                items.append(f"{var}:{val_str}")
            label_content = "{" + ",".join(items) + "}"
            label = f"DDG: {label_content}"
        else:
            # 使用 raw_set + empty_present 构建 {v1,v2,} 样式，包括空项（若存在）
            entries = sorted(raw_set)
            if empty_present:
                entries.append("")  # 空项放在最后，生成像 "avbuf," 的效果
            if entries:
                label_content = "{" + ",".join(entries) + "}"
                label = f"DDG: {label_content}"
            else:
                label = "DDG: {}"

        # 格式化源与目标为逗号分隔并用双引号包裹的单一标识符
        def mk_quoted(items_list):
            stripped = [x.strip('"') for x in items_list]
            return f'"{",".join(stripped)}"'

        hyper_source = mk_quoted(sources)
        hyper_target = mk_quoted(targets)
        print(f"Adding data hyperedge: {hyper_source} -> {hyper_target}  label={label}")
        hyperedge = pydot.Edge(hyper_source, hyper_target, label=label)
        graph.add_edge(hyperedge)

    return graph


# --------------------------
# Parallel / sequential runner (all-in-one)
# --------------------------

import argparse
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


def _extract_label_from_filename(file_name: str):
    """Filename format: <idx>-cpg-<label>.dot"""
    name_without_ext = os.path.splitext(os.path.basename(file_name))[0]
    parts = name_without_ext.split("-")
    if len(parts) >= 3:
        try:
            return int(parts[-1])
        except Exception:
            return None
    return None


def _sort_key(name: str):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def process_one(dot_dir: str, output_dir: str, dot_file: str, label: int, quiet: bool = False):
    """Process a single DOT file: pruning + hyperedge construction + write HCPG DOT."""
    path = os.path.join(dot_dir, dot_file)
    idx = dot_file.split("-")[0]
    out_path = os.path.join(output_dir, f"{idx}-hcpg-{label}.dot")

    try:
        graphs = pydot.graph_from_dot_file(path)
        if not graphs:
            return dot_file, False, f"Cannot read DOT: {path}"
        graph = graphs[0]

        graph_inter_prun = remove_immediate_assignment_and_print_nodes(graph)
        graph_leaf_prun = remove_and_merge_ast_leaf_nodes(graph_inter_prun)
        graph_ctl = build_control_hyperedges(graph_leaf_prun)
        hcpg = build_data_hyperedges(graph_ctl)

        os.makedirs(output_dir, exist_ok=True)
        hcpg.write_raw(out_path)

        if not quiet:
            print(f"[OK] {dot_file} -> {out_path}")
        return dot_file, True, out_path
    except Exception:
        return dot_file, False, traceback.format_exc()


def run(dot_dir: str, output_dir: str, executor: str = "process", workers: int | None = None, quiet: bool = False):
    if not os.path.isdir(dot_dir):
        raise ValueError(f"Dataset directory not found: {dot_dir}")

    dot_files = sorted(os.listdir(dot_dir), key=_sort_key)
    tasks = []
    for f in dot_files:
        lab = _extract_label_from_filename(f)
        if lab is None:
            if not quiet:
                print(f"[SKIP] cannot parse label from: {f}")
            continue
        tasks.append((dot_dir, output_dir, f, lab, quiet))

    if not tasks:
        print("No DOT files to process.")
        return 0, 0

    if workers is None:
        workers = os.cpu_count() or 8

    ok = 0
    fail = 0

    if executor == "sequential":
        for t in tasks:
            dot_file, success, info = process_one(*t)
            if success:
                ok += 1
            else:
                fail += 1
                print(f"[FAIL] {dot_file}\n{info}")
        return ok, fail

    ex_cls = ThreadPoolExecutor if executor == "thread" else ProcessPoolExecutor

    # IMPORTANT: ProcessPool requires __main__ guard (we are in it).
    with ex_cls(max_workers=workers) as ex:
        futures = [ex.submit(process_one, *t) for t in tasks]
        for fut in as_completed(futures):
            dot_file, success, info = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                print(f"[FAIL] {dot_file}\n{info}")

    return ok, fail


def main():
    ap = argparse.ArgumentParser(description="HCPG generation (fixed) with optional multi-thread/multi-process runner.")
    ap.add_argument("--dot_dir", default="dots-cpg", help="Input DOT directory")
    ap.add_argument("--output_dir", default="dots-hcpg", help="Output DOT directory")
    ap.add_argument("--executor", choices=["sequential", "thread", "process"], default="process",
                    help="sequential=single process, thread=ThreadPoolExecutor, process=ProcessPoolExecutor")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8, help="Number of workers for thread/process")
    ap.add_argument("--quiet", action="store_true", help="Reduce per-file prints")
    args = ap.parse_args()

    ok, fail = run(args.dot_dir, args.output_dir, args.executor, args.workers, args.quiet)
    print(f"Done. success={ok}, fail={fail}, output_dir={args.output_dir}")


if __name__ == "__main__":
    main()

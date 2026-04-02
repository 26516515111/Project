import json
import os
import argparse
from pathlib import Path

# 先设置HF环境变量
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:
    import networkx as nx
except ImportError:
    print("正在安装 networkx...")
    os.system("pip install networkx")
    import networkx as nx

try:
    from sentence_transformers import SentenceTransformer
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("正在安装 sentence_transformers 和 scikit-learn...")
    os.system("pip install sentence_transformers scikit-learn")
    from sentence_transformers import SentenceTransformer
    import torch
    from sklearn.metrics.pairwise import cosine_similarity

def create_default_expected_list(path: Path):
    if not path.exists():
        default_entities = [
            "主柴油机及电气遥控系统",
            "驾驶室船舶值班报警系统(BWAS)",
            "机舱微机监测报警系统",
            "机舱总线制监测报警系统",
            "柴油机智能延伸检测单元",
            "主电源板",
            "数据采集板",
            "通信模块",
            "显示终端",
            "继电器",
            "控制箱",
            "报警喇叭"
        ]
        path.write_text("\n".join(default_entities), encoding="utf-8")
        print(f"已创建默认基准实体测试文件: {path}")

def evaluate_kg(kg_path: str, expected_list_path: str):
    print(f"开始加载 KG 数据: {kg_path} ...")
    with open(kg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    G = nx.DiGraph()
    # 建立名字到实体属性的映射
    name_to_node = {}
    
    # 导入节点
    entities = data.get("entities", [])
    for e in entities:
        name = e["name"]
        label = e.get("label", "Entity")
        G.add_node(name, label=label, desc=e.get("description", ""))
        name_to_node[name] = e

    # 导入关系，特别是层级关系
    hierarchical_relations = {"HAS_COMPONENT", "BELONGS_TO", "PART_OF"}
    H_G = nx.DiGraph()
    
    relations = data.get("relations", [])
    for r in relations:
        head = r.get("head", "")
        tail = r.get("tail", "")
        rtype = r.get("relation", "")
        
        # 确保两端节点都存在于图中
        if head not in G:
            G.add_node(head, label=r.get("head_label", "Unknown"))
        if tail not in G:
            G.add_node(tail, label=r.get("tail_label", "Unknown"))
            
        G.add_edge(head, tail, type=rtype)
        
        # 将明确的上下级包含关系抽离出来，专门检测是否存在互相包含（成环）的逻辑悖论
        if rtype in hierarchical_relations:
            H_G.add_edge(head, tail)

    print("\n" + "="*40)
    print("         知识图谱结构 & 质量诊断")
    print("="*40)
    print(f"🔗 总实体 (节点) 数量: {G.number_of_nodes()}")
    print(f"🔗 总三元组 (边) 数量: {G.number_of_edges()}")

    # 1. 拓扑与成环检测
    print("\n--- 1. [逻辑自洽度] 层级关系 DAG 验证 ---")
    if H_G.number_of_edges() == 0:
        print("ℹ️ 图谱中未发现 `HAS_COMPONENT` 或 `BELONGS_TO` 类层级关系。")
    else:
        try:
            cycles = list(nx.simple_cycles(H_G))
            if cycles:
                print(f"🚨 警告: 检测到 {len(cycles)} 个层级关系发生互相包含（逻辑悖论）！")
                for i, cycle in enumerate(cycles[:3]):
                    print(f"   ✖ 环 {i+1}: {' -> '.join(cycle + [cycle[0]])}")
            else:
                print("✅ 优秀: 层级关系 ('HAS_COMPONENT' / 'BELONGS_TO') 呈现严格的树状结构，没有发生逻辑成环，不会影响 RAG 推理的准确性。")
        except Exception as e:
            print(f"成环检测出错: {e}")

    # 2. 超级节点检测 (度数过大，可能由于泛化词合并过度，导致毛线球)
    print("\n--- 2. [图谱清晰度] 节点中心性 & 黑洞节点检测 ---")
    degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
    super_nodes_detected = False
    print("📡 拥有最多连接的实体 (Top 5):")
    for node, degree in degrees[:5]:
        label = G.nodes[node].get("label", "Unknown")
        print(f"   - [{label}] {node} : 连接了 {degree} 个关系")
        if degree > 20 and label not in ["System", "Domain"]:
            super_nodes_detected = True
            
    if super_nodes_detected:
        print("❗ 提示: 存在底层节点连接数过大。如果它是诸如'电源'、'主板'这样宽泛的词，说明大模型没有带上子系统前缀进行命名消歧，形成了将不同语料缠绕在一起的'毛线球'或'黑洞孤岛'。")
    else:
        print("✅ 优秀: 没有异常的超级节点聚簇，网状实体分布比较均匀。")

    # 3. 覆盖率与语义相似度检测
    print("\n--- 3. [业务覆盖度] 语义命中与基准清单匹配 ---")
    expected_path = Path(expected_list_path)
    create_default_expected_list(expected_path)
    expected_entities = [line.strip() for line in expected_path.read_text("utf-8").splitlines() if line.strip()]
    print(f"🎯 基准目标: 从 `{expected_path.name}` 读取了 {len(expected_entities)} 个核心设备/组件预期列表。")
    
    # 只提取我们生成的节点中可能表示实体的（排除孤立属性等，但这里全查）
    extracted_names = list(G.nodes)
    
    if not extracted_names:
        print("❌ 当前图谱为空！")
        return
        
    print("🚀 正在加载嵌入模型进行语义匹配测评...")
    model_name = os.environ.get("BGE_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = SentenceTransformer(model_name, device=device)
        print(f"   ✔️ 模型已加载，运行在 {device} 上。")
        
        expected_embeddings = model.encode(expected_entities, show_progress_bar=False)
        extracted_embeddings = model.encode(extracted_names, show_progress_bar=False)
        
        # 计算相似度矩阵
        sim_matrix = cosine_similarity(expected_embeddings, extracted_embeddings)
        
        threshold = 0.85
        hit_count = 0
        
        print(f"📊 语义阈值设定为: {threshold} (相似度大于此值即视为图谱已成功覆盖该知识)")
        for i, expected_name in enumerate(expected_entities):
            max_sim_idx = sim_matrix[i].argmax()
            max_sim_score = sim_matrix[i][max_sim_idx]
            matched_entity = extracted_names[max_sim_idx]
            
            if max_sim_score >= threshold:
                hit_count += 1
                print(f"   [命中] ✓ {expected_name:<20} -> {matched_entity} (置信度: {max_sim_score:.2f})")
            else:
                print(f"   [缺失] ✗ {expected_name:<20} -> 最佳匹配候选: {matched_entity} (置信度仅: {max_sim_score:.2f})")
                
        coverage_rate = hit_count / len(expected_entities)
        print(f"\n💡 终版评估覆盖率: {coverage_rate:.1%} ({hit_count}/{len(expected_entities)})")
        if coverage_rate >= 0.8:
            print("🌟 结论: 覆盖度极佳！核心设备知识网络基本已经全部被该自动流水线吸收。")
        else:
            print("⚠️ 结论: 抽取有漏掉的关键实体，可能需要扩充语料、调整 Chunk 大小或优化大模型 Prompt 提示词的引导力度。")
            
    except Exception as e:
        print(f"语义相似度计算异常: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KG Quality Evaluator")
    parser.add_argument("--kg", type=str, default="Project/rag_app/data/KG/delivery/kg_merged.json")
    parser.add_argument("--expect", type=str, default="Project/rag_app/docs/expected_equipment.txt")
    args = parser.parse_args()
    
    # Ensure dir
    os.makedirs(os.path.dirname(args.expect), exist_ok=True)
    evaluate_kg(args.kg, args.expect)

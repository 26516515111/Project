import json
import networkx as nx

with open("Project/rag_app/data/KG/delivery/kg_merged.json", "r", encoding="utf-8") as f:
    data = json.load(f)

G = nx.Graph()
entities = data.get("entities", [])
for e in entities:
    G.add_node(e["name"])
    
for r in data.get("relations", []):
    G.add_edge(r["head"], r["tail"])

fine_grained_keywords = {"继电器", "保险", "端子", "引脚", "接线", "二极管", "触头", "插座", "电源接口", "COM", "电阻", "电容", "螺丝", "指示灯", "按钮", "开关"}

pruned = []
for node, degree in G.degree():
    if any(kw in str(node) for kw in fine_grained_keywords) or (degree == 1 and "系统" not in node and "箱" not in node):
        # Let's count how many degree 1 nodes we have that aren't system/box
        pass

degree_one = [node for node, degree in G.degree() if degree == 1]
print(f"Total degree 1 nodes: {len(degree_one)}")
for n in degree_one[:20]:
    print(" -", n)
    
kw_matches = [node for node in G.nodes() if any(kw in str(node) for kw in fine_grained_keywords)]
print(f"\nTotal kw matches: {len(kw_matches)}")
for n in kw_matches[:20]:
    print(" -", n)


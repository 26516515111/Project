import sys
import os
sys.path.append("/home/shahanshah/work/ship/Project/rag_app")
from kg_pipeline.steps import _build_sections_from_headings

text = """[H1] 一、概述
测试内容1
[H1] 二、主要功能及特点
[H2] 1、主机软件在WIN2000环境下运行；
[H2] 2、采用双机冗余运行模式，当主机出现故障时备用主机能够自动投入运行；
测试内容2
"""
res = _build_sections_from_headings(text)
for r in res:
    print(r)

# src/analysis/network_analysis/protocol_network.py
"""
协议网络分析模块
分析DeFi协议间的关系和系统性风险
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import json

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ProtocolNetworkAnalyzer:
    """DeFi协议网络分析器"""
    
    def __init__(self):
        self.network = nx.Graph()
        self.protocol_data = {}
        self.analysis_results = {}
        
    def build_network_from_data(self, protocol_metrics: List[Dict[str, Any]], 
                               tvl_threshold: float = 1000000.0) -> nx.Graph:
        """从协议数据构建网络图"""
        logger.info("Building protocol network from data")
        
        # 清空现有网络
        self.network.clear()
        self.protocol_data.clear()
        
        # 处理协议数据
        protocols_by_chain = {}
        
        for metric in protocol_metrics:
            protocol = metric.get('protocol', 'unknown')
            chain = metric.get('chain', 'unknown')
            tvl = metric.get('tvl_usd', 0)
            
            # 只包含TVL超过阈值的协议
            if tvl < tvl_threshold:
                continue
            
            # 存储协议数据
            self.protocol_data[protocol] = {
                'chain': chain,
                'tvl_usd': tvl,
                'volume_24h': metric.get('volume_24h', 0),
                'category': metric.get('metadata', {}).get('category', 'unknown')
            }
            
            # 按链分组
            if chain not in protocols_by_chain:
                protocols_by_chain[chain] = []
            protocols_by_chain[chain].append(protocol)
        
        # 添加节点
        for protocol, data in self.protocol_data.items():
            self.network.add_node(protocol, **data)
        
        # 添加边（基于共同链和TVL相似性）
        self._add_network_edges(protocols_by_chain)
        
        logger.info(f"Network built with {self.network.number_of_nodes()} nodes and {self.network.number_of_edges()} edges")
        return self.network
    
    def _add_network_edges(self, protocols_by_chain: Dict[str, List[str]]):
        """添加网络边"""
        
        # 1. 同链协议间的连接
        for chain, protocols in protocols_by_chain.items():
            if len(protocols) < 2:
                continue
                
            for i, protocol1 in enumerate(protocols):
                for protocol2 in protocols[i+1:]:
                    # 基于TVL相似性添加权重
                    tvl1 = self.protocol_data[protocol1]['tvl_usd']
                    tvl2 = self.protocol_data[protocol2]['tvl_usd']
                    
                    # 计算相似性权重
                    similarity = self._calculate_tvl_similarity(tvl1, tvl2)
                    
                    if similarity > 0.1:  # 只添加相似性较高的边
                        self.network.add_edge(protocol1, protocol2, 
                                            weight=similarity,
                                            relationship='same_chain',
                                            chain=chain)
        
        # 2. 跨链协议连接（如果协议名称相似）
        protocols = list(self.protocol_data.keys())
        for i, protocol1 in enumerate(protocols):
            for protocol2 in protocols[i+1:]:
                if not self.network.has_edge(protocol1, protocol2):
                    # 检查协议名称相似性
                    name_similarity = self._calculate_name_similarity(protocol1, protocol2)
                    if name_similarity > 0.7:  # 高名称相似性
                        self.network.add_edge(protocol1, protocol2,
                                            weight=name_similarity * 0.5,
                                            relationship='cross_chain')
    
    def _calculate_tvl_similarity(self, tvl1: float, tvl2: float) -> float:
        """计算TVL相似性"""
        if tvl1 == 0 or tvl2 == 0:
            return 0
        
        ratio = min(tvl1, tvl2) / max(tvl1, tvl2)
        return ratio
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """计算协议名称相似性"""
        # 简单的字符串相似性计算
        name1, name2 = name1.lower(), name2.lower()
        
        if name1 == name2:
            return 1.0
        
        # 检查是否一个是另一个的子字符串
        if name1 in name2 or name2 in name1:
            return 0.8
        
        # 简单的字符重叠计算
        set1, set2 = set(name1), set(name2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0
    
    def analyze_network_centrality(self) -> Dict[str, Dict[str, float]]:
        """分析网络中心性指标"""
        if self.network.number_of_nodes() == 0:
            return {}
        
        logger.info("Analyzing network centrality metrics")
        
        centrality_metrics = {}
        
        try:
            # 度中心性
            degree_centrality = nx.degree_centrality(self.network)
            
            # 介数中心性
            betweenness_centrality = nx.betweenness_centrality(self.network)
            
            # 特征向量中心性
            try:
                eigenvector_centrality = nx.eigenvector_centrality(self.network, max_iter=1000)
            except:
                eigenvector_centrality = {node: 0 for node in self.network.nodes()}
            
            # 接近中心性
            closeness_centrality = nx.closeness_centrality(self.network)
            
            # 组合所有指标
            for node in self.network.nodes():
                centrality_metrics[node] = {
                    'degree_centrality': degree_centrality.get(node, 0),
                    'betweenness_centrality': betweenness_centrality.get(node, 0),
                    'eigenvector_centrality': eigenvector_centrality.get(node, 0),
                    'closeness_centrality': closeness_centrality.get(node, 0),
                    'tvl_usd': self.protocol_data.get(node, {}).get('tvl_usd', 0)
                }
            
            self.analysis_results['centrality'] = centrality_metrics
            logger.info(f"Centrality analysis completed for {len(centrality_metrics)} protocols")
            
        except Exception as e:
            logger.error(f"Error in centrality analysis: {e}")
            return {}
        
        return centrality_metrics
    
    def identify_critical_protocols(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """识别关键协议"""
        if 'centrality' not in self.analysis_results:
            self.analyze_network_centrality()
        
        centrality_data = self.analysis_results.get('centrality', {})
        
        if not centrality_data:
            return []
        
        # 计算综合重要性分数
        critical_protocols = []
        
        for protocol, metrics in centrality_data.items():
            # 综合分数 = 加权中心性指标
            importance_score = (
                metrics['degree_centrality'] * 0.3 +
                metrics['betweenness_centrality'] * 0.3 +
                metrics['eigenvector_centrality'] * 0.2 +
                metrics['closeness_centrality'] * 0.2
            )
            
            critical_protocols.append({
                'protocol': protocol,
                'importance_score': importance_score,
                'tvl_usd': metrics['tvl_usd'],
                'chain': self.protocol_data.get(protocol, {}).get('chain', 'unknown'),
                'centrality_metrics': metrics
            })
        
        # 按重要性分数排序
        critical_protocols.sort(key=lambda x: x['importance_score'], reverse=True)
        
        return critical_protocols[:top_n]
    
    def analyze_systemic_risk(self) -> Dict[str, Any]:
        """分析系统性风险"""
        logger.info("Analyzing systemic risk")
        
        risk_analysis = {
            'network_density': 0,
            'clustering_coefficient': 0,
            'connected_components': 0,
            'largest_component_size': 0,
            'risk_score': 0,
            'vulnerabilities': []
        }
        
        if self.network.number_of_nodes() == 0:
            return risk_analysis
        
        try:
            # 网络密度
            risk_analysis['network_density'] = nx.density(self.network)
            
            # 聚类系数
            risk_analysis['clustering_coefficient'] = nx.average_clustering(self.network)
            
            # 连通组件
            components = list(nx.connected_components(self.network))
            risk_analysis['connected_components'] = len(components)
            risk_analysis['largest_component_size'] = len(max(components, key=len)) if components else 0
            
            # 计算风险分数
            density_risk = min(risk_analysis['network_density'] * 2, 1.0)  # 高密度 = 高风险
            clustering_risk = risk_analysis['clustering_coefficient']  # 高聚类 = 高风险
            concentration_risk = risk_analysis['largest_component_size'] / self.network.number_of_nodes() if self.network.number_of_nodes() > 0 else 0
            
            risk_analysis['risk_score'] = (density_risk * 0.4 + clustering_risk * 0.3 + concentration_risk * 0.3)
            
            # 识别漏洞
            vulnerabilities = []
            
            if risk_analysis['network_density'] > 0.7:
                vulnerabilities.append("High network density indicates potential contagion risk")
            
            if risk_analysis['clustering_coefficient'] > 0.8:
                vulnerabilities.append("High clustering suggests concentrated risk")
            
            if concentration_risk > 0.8:
                vulnerabilities.append("Most protocols are in a single connected component")
            
            risk_analysis['vulnerabilities'] = vulnerabilities
            
            self.analysis_results['systemic_risk'] = risk_analysis
            logger.info(f"Systemic risk analysis completed with risk score: {risk_analysis['risk_score']:.3f}")
            
        except Exception as e:
            logger.error(f"Error in systemic risk analysis: {e}")
        
        return risk_analysis
    
    def simulate_contagion(self, initial_shock_protocols: List[str], 
                          shock_magnitude: float = 0.5) -> Dict[str, Any]:
        """模拟传染效应"""
        logger.info(f"Simulating contagion from {len(initial_shock_protocols)} initial protocols")
        
        contagion_result = {
            'initial_protocols': initial_shock_protocols,
            'affected_protocols': set(initial_shock_protocols),
            'contagion_rounds': [],
            'total_tvl_affected': 0,
            'contagion_stopped': False
        }
        
        if not initial_shock_protocols or self.network.number_of_nodes() == 0:
            return contagion_result
        
        current_affected = set(initial_shock_protocols)
        round_num = 0
        max_rounds = 10  # 防止无限循环
        
        while round_num < max_rounds:
            round_num += 1
            new_affected = set()
            
            # 检查当前受影响协议的邻居
            for protocol in current_affected:
                if protocol in self.network:
                    neighbors = list(self.network.neighbors(protocol))
                    
                    for neighbor in neighbors:
                        if neighbor not in contagion_result['affected_protocols']:
                            # 计算传染概率
                            edge_weight = self.network[protocol][neighbor].get('weight', 0.1)
                            contagion_prob = edge_weight * shock_magnitude
                            
                            # 简单的传染规则
                            if contagion_prob > 0.3:  # 传染阈值
                                new_affected.add(neighbor)
            
            if not new_affected:
                contagion_result['contagion_stopped'] = True
                break
            
            contagion_result['affected_protocols'].update(new_affected)
            contagion_result['contagion_rounds'].append({
                'round': round_num,
                'new_affected': list(new_affected),
                'cumulative_affected': len(contagion_result['affected_protocols'])
            })
            
            current_affected = new_affected
        
        # 计算受影响的总TVL
        total_tvl = sum(
            self.protocol_data.get(protocol, {}).get('tvl_usd', 0)
            for protocol in contagion_result['affected_protocols']
        )
        contagion_result['total_tvl_affected'] = total_tvl
        
        logger.info(f"Contagion simulation completed: {len(contagion_result['affected_protocols'])} protocols affected")
        
        return contagion_result
    
    def get_network_summary(self) -> Dict[str, Any]:
        """获取网络摘要"""
        summary = {
            'timestamp': datetime.utcnow(),
            'nodes_count': self.network.number_of_nodes(),
            'edges_count': self.network.number_of_edges(),
            'protocols_by_chain': {},
            'total_tvl': 0,
            'analysis_available': list(self.analysis_results.keys())
        }
        
        # 按链统计协议
        for protocol, data in self.protocol_data.items():
            chain = data.get('chain', 'unknown')
            if chain not in summary['protocols_by_chain']:
                summary['protocols_by_chain'][chain] = {'count': 0, 'tvl': 0}
            
            summary['protocols_by_chain'][chain]['count'] += 1
            summary['protocols_by_chain'][chain]['tvl'] += data.get('tvl_usd', 0)
            summary['total_tvl'] += data.get('tvl_usd', 0)
        
        return summary
    
    def export_network_data(self, filepath: str = None) -> Dict[str, Any]:
        """导出网络数据"""
        export_data = {
            'network_summary': self.get_network_summary(),
            'protocol_data': self.protocol_data,
            'analysis_results': self.analysis_results,
            'network_edges': []
        }
        
        # 导出边数据
        for edge in self.network.edges(data=True):
            export_data['network_edges'].append({
                'source': edge[0],
                'target': edge[1],
                'attributes': edge[2]
            })
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
                logger.info(f"Network data exported to {filepath}")
            except Exception as e:
                logger.error(f"Error exporting network data: {e}")
        
        return export_data

# 全局网络分析器实例
protocol_network_analyzer = ProtocolNetworkAnalyzer()
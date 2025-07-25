# src/visualization/network_dashboard.py
"""
DeFi网络分析可视化仪表板
使用Streamlit创建交互式界面
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
from datetime import datetime
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.analysis.network_analysis.protocol_network import protocol_network_analyzer
    from src.data.data_pipeline import data_pipeline
    from src.data.storage.database import db_manager
    from src.utils.logger import setup_logger
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please make sure you're running from the project root directory")
    st.stop()

logger = setup_logger(__name__)

# 页面配置
st.set_page_config(
    page_title="OnChain Macro Analysis - Network Dashboard",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_sample_data():
    """创建示例数据"""
    return [
        {'protocol': 'uniswap', 'chain': 'ethereum', 'tvl_usd': 5000000000, 'volume_24h': 1000000000, 'metadata': {'category': 'dex'}},
        {'protocol': 'aave', 'chain': 'ethereum', 'tvl_usd': 8000000000, 'volume_24h': 500000000, 'metadata': {'category': 'lending'}},
        {'protocol': 'compound', 'chain': 'ethereum', 'tvl_usd': 3000000000, 'volume_24h': 200000000, 'metadata': {'category': 'lending'}},
        {'protocol': 'makerdao', 'chain': 'ethereum', 'tvl_usd': 6000000000, 'volume_24h': 100000000, 'metadata': {'category': 'stablecoin'}},
        {'protocol': 'curve', 'chain': 'ethereum', 'tvl_usd': 4000000000, 'volume_24h': 800000000, 'metadata': {'category': 'dex'}},
        {'protocol': 'lido', 'chain': 'ethereum', 'tvl_usd': 15000000000, 'volume_24h': 50000000, 'metadata': {'category': 'staking'}},
        {'protocol': 'pancakeswap', 'chain': 'bsc', 'tvl_usd': 2000000000, 'volume_24h': 400000000, 'metadata': {'category': 'dex'}},
        {'protocol': 'aave', 'chain': 'polygon', 'tvl_usd': 500000000, 'volume_24h': 50000000, 'metadata': {'category': 'lending'}},
        {'protocol': 'uniswap', 'chain': 'arbitrum', 'tvl_usd': 1000000000, 'volume_24h': 200000000, 'metadata': {'category': 'dex'}},
        {'protocol': 'gmx', 'chain': 'arbitrum', 'tvl_usd': 800000000, 'volume_24h': 300000000, 'metadata': {'category': 'derivatives'}},
    ]

def create_network_visualization(network, protocol_data):
    """创建网络可视化图"""
    if network.number_of_nodes() == 0:
        return None
    
    # 使用春力导向布局
    pos = nx.spring_layout(network, k=3, iterations=50)
    
    # 准备节点数据
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    for node in network.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        tvl = protocol_data.get(node, {}).get('tvl_usd', 0)
        chain = protocol_data.get(node, {}).get('chain', 'unknown')
        
        node_text.append(f"{node}<br>TVL: ${tvl:,.0f}<br>Chain: {chain}")
        node_size.append(max(20, min(60, tvl / 200000000)))  # 根据TVL调整大小
        
        # 根据链设置颜色
        chain_colors = {'ethereum': '#627EEA', 'bsc': '#F3BA2F', 'polygon': '#8247E5', 'arbitrum': '#28A0F0'}
        node_color.append(chain_colors.get(chain, '#888888'))
    
    # 准备边数据
    edge_x = []
    edge_y = []
    
    for edge in network.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # 创建图形
    fig = go.Figure()
    
    # 添加边
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='Connections'
    ))
    
    # 添加节点
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[node.upper() for node in network.nodes()],
        textposition="middle center",
        textfont=dict(size=10, color="white"),
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white')
        ),
        name='Protocols'
    ))
    
    fig.update_layout(
        title="DeFi Protocol Network",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Node size = TVL, Color = Chain",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(size=12, color="#888")
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        height=500
    )
    
    return fig

def create_centrality_chart(centrality_data):
    """创建中心性指标图表"""
    if not centrality_data:
        return None
    
    # 准备数据
    protocols = list(centrality_data.keys())
    degree_cent = [centrality_data[p]['degree_centrality'] for p in protocols]
    between_cent = [centrality_data[p]['betweenness_centrality'] for p in protocols]
    close_cent = [centrality_data[p]['closeness_centrality'] for p in protocols]
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Degree Centrality', 'Betweenness Centrality', 'Closeness Centrality', 'Combined Score'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # 度中心性
    fig.add_trace(go.Bar(
        x=protocols, y=degree_cent,
        name='Degree', marker_color='lightblue'
    ), row=1, col=1)
    
    # 介数中心性
    fig.add_trace(go.Bar(
        x=protocols, y=between_cent,
        name='Betweenness', marker_color='lightgreen'
    ), row=1, col=2)
    
    # 接近中心性
    fig.add_trace(go.Bar(
        x=protocols, y=close_cent,
        name='Closeness', marker_color='lightyellow'
    ), row=2, col=1)
    
    # 综合评分
    combined_scores = [degree_cent[i] * 0.4 + between_cent[i] * 0.3 + close_cent[i] * 0.3 
                      for i in range(len(protocols))]
    
    fig.add_trace(go.Scatter(
        x=protocols, y=combined_scores,
        mode='markers+lines',
        name='Combined Score',
        marker=dict(size=10, color='red')
    ), row=2, col=2)
    
    fig.update_layout(
        title="Protocol Centrality Analysis",
        showlegend=False,
        height=600
    )
    
    return fig

def create_risk_gauge(risk_score):
    """创建风险评分仪表盘"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Systemic Risk Score"},
        delta = {'reference': 0.5},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgreen"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "red"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.8}}))
    
    fig.update_layout(height=400)
    return fig

def main():
    """主仪表板函数"""
    st.title("🕸️ DeFi Protocol Network Analysis")
    st.markdown("Analyze systemic risks and relationships in the DeFi ecosystem")
    
    # 侧边栏控制
    st.sidebar.header("🔧 Controls")
    
    # 数据源选择
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Sample Data", "Live Data", "Database"]
    )
    
    # TVL阈值
    tvl_threshold = st.sidebar.slider(
        "TVL Threshold (Million USD)",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100
    ) * 1000000
    
    # 加载数据
    with st.spinner("Loading network data..."):
        if data_source == "Sample Data":
            protocol_data = create_sample_data()
        else:
            st.info("Live data integration coming soon!")
            protocol_data = create_sample_data()
        
        # 构建网络
        network = protocol_network_analyzer.build_network_from_data(
            protocol_data, 
            tvl_threshold=tvl_threshold
        )
    
    # 主要指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Protocols", network.number_of_nodes())
    
    with col2:
        st.metric("Connections", network.number_of_edges())
    
    with col3:
        total_tvl = sum(p.get('tvl_usd', 0) for p in protocol_data if p.get('tvl_usd', 0) >= tvl_threshold)
        st.metric("Total TVL", f"${total_tvl/1e9:.1f}B")
    
    with col4:
        density = nx.density(network) if network.number_of_nodes() > 0 else 0
        st.metric("Network Density", f"{density:.3f}")
    
    # 网络可视化
    st.header("🔗 Protocol Network")
    
    if network.number_of_nodes() > 0:
        network_fig = create_network_visualization(network, protocol_network_analyzer.protocol_data)
        if network_fig:
            st.plotly_chart(network_fig, use_container_width=True)
    else:
        st.warning("No protocols meet the TVL threshold. Try lowering the threshold.")
    
    # 分析结果
    if network.number_of_nodes() > 0:
        # 中心性分析
        st.header("📊 Centrality Analysis")
        
        if st.button("Run Centrality Analysis"):
            with st.spinner("Analyzing centrality metrics..."):
                centrality_results = protocol_network_analyzer.analyze_network_centrality()
                
                if centrality_results:
                    # 显示图表
                    centrality_fig = create_centrality_chart(centrality_results)
                    if centrality_fig:
                        st.plotly_chart(centrality_fig, use_container_width=True)
                    
                    # 显示数据表
                    st.subheader("Centrality Metrics Table")
                    centrality_df = pd.DataFrame(centrality_results).T
                    centrality_df = centrality_df.round(3)
                    st.dataframe(centrality_df)
        
        # 关键协议识别
        st.header("🎯 Critical Protocols")
        
        if st.button("Identify Critical Protocols"):
            with st.spinner("Identifying critical protocols..."):
                critical_protocols = protocol_network_analyzer.identify_critical_protocols(top_n=10)
                
                if critical_protocols:
                    # 创建表格
                    critical_df = pd.DataFrame([
                        {
                            'Protocol': p['protocol'].upper(),
                            'Chain': p['chain'],
                            'Importance Score': f"{p['importance_score']:.3f}",
                            'TVL (USD)': f"${p['tvl_usd']:,.0f}",
                            'Degree': f"{p['centrality_metrics']['degree_centrality']:.3f}",
                            'Betweenness': f"{p['centrality_metrics']['betweenness_centrality']:.3f}",
                            'Closeness': f"{p['centrality_metrics']['closeness_centrality']:.3f}"
                        }
                        for p in critical_protocols
                    ])
                    
                    st.dataframe(critical_df, use_container_width=True)
                    
                    # 重要性分数图表
                    fig = px.bar(
                        x=[p['protocol'].upper() for p in critical_protocols],
                        y=[p['importance_score'] for p in critical_protocols],
                        title="Protocol Importance Scores",
                        labels={'x': 'Protocol', 'y': 'Importance Score'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # 系统性风险分析
        st.header("⚠️ Systemic Risk Analysis")
        
        if st.button("Analyze Systemic Risk"):
            with st.spinner("Analyzing systemic risk..."):
                risk_analysis = protocol_network_analyzer.analyze_systemic_risk()
                
                if risk_analysis:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 风险仪表盘
                        risk_fig = create_risk_gauge(risk_analysis['risk_score'])
                        st.plotly_chart(risk_fig, use_container_width=True)
                    
                    with col2:
                        # 风险指标
                        st.subheader("Risk Metrics")
                        st.metric("Network Density", f"{risk_analysis['network_density']:.3f}")
                        st.metric("Clustering Coefficient", f"{risk_analysis['clustering_coefficient']:.3f}")
                        st.metric("Connected Components", risk_analysis['connected_components'])
                        st.metric("Largest Component", f"{risk_analysis['largest_component_size']} protocols")
                        
                        # 漏洞列表
                        if risk_analysis['vulnerabilities']:
                            st.subheader("⚠️ Identified Vulnerabilities")
                            for vuln in risk_analysis['vulnerabilities']:
                                st.warning(vuln)
                        else:
                            st.success("No major vulnerabilities identified")
        
        # 传染效应模拟
        st.header("🦠 Contagion Simulation")
        
        # 选择初始冲击协议
        available_protocols = list(protocol_network_analyzer.protocol_data.keys())
        if available_protocols:
            selected_protocol = st.selectbox(
                "Select Protocol for Initial Shock",
                available_protocols
            )
            
            shock_magnitude = st.slider(
                "Shock Magnitude",
                min_value=0.1,
                max_value=1.0,
                value=0.6,
                step=0.1
            )
            
            if st.button("Run Contagion Simulation"):
                with st.spinner("Simulating contagion effects..."):
                    contagion_result = protocol_network_analyzer.simulate_contagion(
                        initial_shock_protocols=[selected_protocol],
                        shock_magnitude=shock_magnitude
                    )
                    
                    # 显示结果
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Protocols Affected", len(contagion_result['affected_protocols']))
                    
                    with col2:
                        impact_ratio = len(contagion_result['affected_protocols']) / len(available_protocols)
                        st.metric("Impact Ratio", f"{impact_ratio:.1%}")
                    
                    with col3:
                        st.metric("TVL Affected", f"${contagion_result['total_tvl_affected']/1e9:.1f}B")
                    
                    # 传染轮次
                    if contagion_result['contagion_rounds']:
                        st.subheader("Contagion Spread")
                        rounds_df = pd.DataFrame(contagion_result['contagion_rounds'])
                        st.dataframe(rounds_df)
                    
                    # 受影响协议列表
                    st.subheader("Affected Protocols")
                    affected_list = list(contagion_result['affected_protocols'])
                    st.write(", ".join([p.upper() for p in affected_list]))

    # 页脚
    st.markdown("---")
    st.markdown("**OnChain Macro Analysis** - Network Analysis Dashboard 🚀")

if __name__ == "__main__":
    main()
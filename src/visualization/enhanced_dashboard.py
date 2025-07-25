# src/visualization/enhanced_dashboard.py
"""
增强版DeFi网络分析仪表板 - 支持真实数据
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import asyncio
from datetime import datetime
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.analysis.network_analysis.protocol_network import protocol_network_analyzer
    from src.data.real_data_fetcher import fetch_real_data
    from src.utils.logger import setup_logger
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

logger = setup_logger(__name__)

# 页面配置
st.set_page_config(
    page_title="OnChain Macro Analysis - Enhanced Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=300)  # 缓存5分钟
def load_real_data():
    """加载真实数据（带缓存）"""
    try:
        return asyncio.run(fetch_real_data())
    except Exception as e:
        st.error(f"Failed to load real data: {e}")
        return None

def create_sample_data():
    """备用示例数据"""
    return [
        {'protocol': 'uniswap', 'chain': 'ethereum', 'tvl_usd': 5000000000, 'volume_24h': 1000000000, 'metadata': {'category': 'dex'}},
        {'protocol': 'aave', 'chain': 'ethereum', 'tvl_usd': 8000000000, 'volume_24h': 500000000, 'metadata': {'category': 'lending'}},
        {'protocol': 'compound', 'chain': 'ethereum', 'tvl_usd': 3000000000, 'volume_24h': 200000000, 'metadata': {'category': 'lending'}},
        {'protocol': 'makerdao', 'chain': 'ethereum', 'tvl_usd': 6000000000, 'volume_24h': 100000000, 'metadata': {'category': 'stablecoin'}},
        {'protocol': 'curve', 'chain': 'ethereum', 'tvl_usd': 4000000000, 'volume_24h': 800000000, 'metadata': {'category': 'dex'}},
        {'protocol': 'lido', 'chain': 'ethereum', 'tvl_usd': 15000000000, 'volume_24h': 50000000, 'metadata': {'category': 'staking'}},
    ]

def create_network_visualization(network, protocol_data):
    """创建增强版网络可视化"""
    if network.number_of_nodes() == 0:
        return None
    
    # 使用spring layout with better parameters
    pos = nx.spring_layout(network, k=2, iterations=100, seed=42)
    
    # 节点数据
    node_trace = []
    
    for node in network.nodes():
        x, y = pos[node]
        tvl = protocol_data.get(node, {}).get('tvl_usd', 0)
        chain = protocol_data.get(node, {}).get('chain', 'unknown')
        category = protocol_data.get(node, {}).get('metadata', {}).get('category', 'unknown')
        
        # 根据链设置颜色
        chain_colors = {
            'ethereum': '#627EEA',
            'bsc': '#F3BA2F', 
            'polygon': '#8247E5',
            'arbitrum': '#28A0F0',
            'avalanche': '#E84142',
            'optimism': '#FF0420',
            'fantom': '#1969FF'
        }
        color = chain_colors.get(chain, '#888888')
        
        # 节点大小基于TVL (log scale)
        size = max(15, min(50, 15 + (tvl / 100000000) ** 0.3))
        
        node_trace.append({
            'x': x, 'y': y,
            'text': node.upper(),
            'hovertext': f"<b>{node.upper()}</b><br>Chain: {chain}<br>Category: {category}<br>TVL: ${tvl:,.0f}",
            'size': size,
            'color': color,
            'chain': chain,
            'category': category
        })
    
    # 边数据
    edge_x, edge_y = [], []
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
        line=dict(width=0.8, color='rgba(125,125,125,0.3)'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    ))
    
    # 按链分组添加节点
    chains = set(trace['chain'] for trace in node_trace)
    for chain in chains:
        chain_nodes = [trace for trace in node_trace if trace['chain'] == chain]
        
        fig.add_trace(go.Scatter(
            x=[trace['x'] for trace in chain_nodes],
            y=[trace['y'] for trace in chain_nodes],
            mode='markers+text',
            text=[trace['text'] for trace in chain_nodes],
            textposition="middle center",
            textfont=dict(size=8, color="white"),
            hovertext=[trace['hovertext'] for trace in chain_nodes],
            hoverinfo='text',
            marker=dict(
                size=[trace['size'] for trace in chain_nodes],
                color=chain_nodes[0]['color'],
                line=dict(width=1, color='white'),
                opacity=0.8
            ),
            name=chain.capitalize(),
            showlegend=True
        ))
    
    fig.update_layout(
        title="DeFi Protocol Network - Real Data",
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        height=600
    )
    
    return fig

def create_chain_tvl_chart(chain_data):
    """创建链TVL分布图"""
    if not chain_data:
        return None
    
    # 准备数据
    chain_df = pd.DataFrame(chain_data)
    chain_df = chain_df.sort_values('tvl_usd', ascending=False).head(10)
    
    # 创建条形图
    fig = px.bar(
        chain_df,
        x='chain',
        y='tvl_usd',
        title='Top 10 Chains by TVL',
        labels={'tvl_usd': 'TVL (USD)', 'chain': 'Chain'},
        color='tvl_usd',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        yaxis_tickformat='$,.0f'
    )
    
    return fig

def create_protocol_categories_chart(protocol_data):
    """创建协议分类分布图"""
    if not protocol_data:
        return None
    
    # 统计分类
    categories = {}
    for protocol in protocol_data:
        category = protocol.get('metadata', {}).get('category', 'unknown')
        tvl = protocol.get('tvl_usd', 0)
        
        if category not in categories:
            categories[category] = {'count': 0, 'tvl': 0}
        
        categories[category]['count'] += 1
        categories[category]['tvl'] += tvl
    
    # 创建数据框
    cat_df = pd.DataFrame([
        {
            'category': cat,
            'protocol_count': data['count'],
            'total_tvl': data['tvl']
        }
        for cat, data in categories.items()
    ]).sort_values('total_tvl', ascending=False)
    
    # 创建子图
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Protocols by Category', 'TVL by Category'),
        specs=[[{"type": "pie"}, {"type": "bar"}]]
    )
    
    # 协议数量饼图
    fig.add_trace(go.Pie(
        labels=cat_df['category'],
        values=cat_df['protocol_count'],
        name="Protocol Count"
    ), row=1, col=1)
    
    # TVL条形图
    fig.add_trace(go.Bar(
        x=cat_df['category'],
        y=cat_df['total_tvl'],
        name="Total TVL",
        marker_color='lightblue'
    ), row=1, col=2)
    
    fig.update_layout(
        title="DeFi Ecosystem Distribution",
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    """主仪表板函数"""
    st.title("🚀 OnChain Macro Analysis - Enhanced Dashboard")
    st.markdown("**Real-time DeFi ecosystem analysis with live data**")
    
    # 侧边栏
    st.sidebar.header("🔧 Dashboard Controls")
    
    # 数据源选择
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["🌐 Live Data (DeFiLlama + CoinGecko)", "📊 Sample Data"]
    )
    
    # TVL过滤器
    min_tvl = st.sidebar.slider(
        "Minimum TVL (Million USD)",
        min_value=1,
        max_value=1000,
        value=50,
        step=10
    ) * 1000000
    
    # 最大协议数
    max_protocols = st.sidebar.slider(
        "Max Protocols to Analyze",
        min_value=10,
        max_value=100,
        value=30,
        step=5
    )
    
    # 加载数据
    with st.spinner("🌐 Loading data..."):
        if "🌐 Live Data" in data_source:
            real_data = load_real_data()
            if real_data and real_data['protocols']:
                # 过滤和排序协议
                protocols = [p for p in real_data['protocols'] if p['tvl_usd'] >= min_tvl]
                protocols = sorted(protocols, key=lambda x: x['tvl_usd'], reverse=True)[:max_protocols]
                
                chain_data = real_data.get('chains', [])
                global_indicators = real_data.get('global_indicators', [])
                
                st.success(f"✅ Loaded {len(protocols)} live protocols")
            else:
                st.warning("Failed to load live data, using sample data")
                protocols = create_sample_data()
                chain_data = []
                global_indicators = []
        else:
            protocols = create_sample_data()
            chain_data = []
            global_indicators = []
    
    # 全局指标显示
    if global_indicators:
        st.header("🌍 Global Crypto Market")
        
        # 提取主要指标
        market_cap = next((ind['value'] for ind in global_indicators if ind['indicator'] == 'crypto_total_market_cap'), 0)
        volume_24h = next((ind['value'] for ind in global_indicators if ind['indicator'] == 'crypto_total_volume_24h'), 0)
        btc_dominance = next((ind['value'] for ind in global_indicators if ind['indicator'] == 'bitcoin_dominance'), 0)
        defi_market_cap = next((ind['value'] for ind in global_indicators if ind['indicator'] == 'defi_market_cap'), 0)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Market Cap", f"${market_cap/1e12:.2f}T" if market_cap > 0 else "N/A")
        
        with col2:
            st.metric("24h Volume", f"${volume_24h/1e9:.1f}B" if volume_24h > 0 else "N/A")
        
        with col3:
            st.metric("BTC Dominance", f"{btc_dominance:.1f}%" if btc_dominance > 0 else "N/A")
        
        with col4:
            st.metric("DeFi Market Cap", f"${defi_market_cap/1e9:.1f}B" if defi_market_cap > 0 else "N/A")
    
    # 链分析
    if chain_data:
        st.header("⛓️ Blockchain Analysis")
        
        # 总TVL
        total_tvl = sum(chain['tvl_usd'] for chain in chain_data)
        st.metric("Total DeFi TVL", f"${total_tvl/1e9:.1f}B")
        
        # 链TVL图表
        chain_chart = create_chain_tvl_chart(chain_data)
        if chain_chart:
            st.plotly_chart(chain_chart, use_container_width=True)
    
    # 协议分析
    st.header("📊 Protocol Analysis")
    
    if protocols:
        # 构建网络
        with st.spinner("🔗 Building protocol network..."):
            network = protocol_network_analyzer.build_network_from_data(
                protocols, 
                tvl_threshold=min_tvl
            )
        
        # 主要指标
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Protocols", network.number_of_nodes())
        
        with col2:
            st.metric("Network Connections", network.number_of_edges())
        
        with col3:
            protocol_tvl = sum(p['tvl_usd'] for p in protocols)
            st.metric("Protocol TVL", f"${protocol_tvl/1e9:.1f}B")
        
        with col4:
            density = nx.density(network) if network.number_of_nodes() > 0 else 0
            st.metric("Network Density", f"{density:.3f}")
        
        # 协议分类分析
        category_chart = create_protocol_categories_chart(protocols)
        if category_chart:
            st.plotly_chart(category_chart, use_container_width=True)
        
        # 网络可视化
        st.header("🕸️ Protocol Network Visualization")
        
        if network.number_of_nodes() > 0:
            network_fig = create_network_visualization(network, protocol_network_analyzer.protocol_data)
            if network_fig:
                st.plotly_chart(network_fig, use_container_width=True)
            
            # 网络分析按钮
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🎯 Analyze Centrality", use_container_width=True):
                    with st.spinner("Analyzing centrality..."):
                        centrality_results = protocol_network_analyzer.analyze_network_centrality()
                        
                        if centrality_results:
                            st.success("✅ Centrality analysis completed!")
                            
                            # 创建中心性数据表
                            centrality_df = pd.DataFrame(centrality_results).T
                            centrality_df = centrality_df.round(3)
                            centrality_df['TVL (B)'] = centrality_df['tvl_usd'] / 1e9
                            centrality_df = centrality_df.sort_values('degree_centrality', ascending=False)
                            
                            st.subheader("🏆 Top Protocols by Centrality")
                            st.dataframe(
                                centrality_df[['degree_centrality', 'betweenness_centrality', 'closeness_centrality', 'TVL (B)']].head(10),
                                use_container_width=True
                            )
            
            with col2:
                if st.button("⚠️ Risk Assessment", use_container_width=True):
                    with st.spinner("Assessing systemic risk..."):
                        risk_analysis = protocol_network_analyzer.analyze_systemic_risk()
                        
                        if risk_analysis:
                            st.success("✅ Risk assessment completed!")
                            
                            # 风险指标
                            risk_col1, risk_col2 = st.columns(2)
                            
                            with risk_col1:
                                st.metric("Risk Score", f"{risk_analysis['risk_score']:.3f}")
                                st.metric("Network Density", f"{risk_analysis['network_density']:.3f}")
                            
                            with risk_col2:
                                st.metric("Clustering Coeff.", f"{risk_analysis['clustering_coefficient']:.3f}")
                                st.metric("Connected Components", risk_analysis['connected_components'])
                            
                            # 风险警告
                            if risk_analysis['vulnerabilities']:
                                st.subheader("⚠️ Risk Vulnerabilities")
                                for vuln in risk_analysis['vulnerabilities']:
                                    st.warning(vuln)
                            
                            # 风险等级
                            risk_score = risk_analysis['risk_score']
                            if risk_score < 0.3:
                                st.success("🟢 Low systemic risk")
                            elif risk_score < 0.7:
                                st.warning("🟡 Medium systemic risk")
                            else:
                                st.error("🔴 High systemic risk")
            
            with col3:
                if st.button("🦠 Contagion Simulation", use_container_width=True):
                    # 选择要模拟的协议
                    available_protocols = list(protocol_network_analyzer.protocol_data.keys())
                    if available_protocols:
                        # 选择TVL最大的协议作为默认
                        largest_protocol = max(
                            available_protocols,
                            key=lambda p: protocol_network_analyzer.protocol_data.get(p, {}).get('tvl_usd', 0)
                        )
                        
                        with st.spinner(f"Simulating contagion from {largest_protocol}..."):
                            contagion_result = protocol_network_analyzer.simulate_contagion(
                                initial_shock_protocols=[largest_protocol],
                                shock_magnitude=0.7
                            )
                            
                            st.success("✅ Contagion simulation completed!")
                            
                            # 影响统计
                            sim_col1, sim_col2, sim_col3 = st.columns(3)
                            
                            with sim_col1:
                                st.metric("Affected Protocols", len(contagion_result['affected_protocols']))
                            
                            with sim_col2:
                                impact_ratio = len(contagion_result['affected_protocols']) / len(available_protocols)
                                st.metric("Impact Ratio", f"{impact_ratio:.1%}")
                            
                            with sim_col3:
                                st.metric("Affected TVL", f"${contagion_result['total_tvl_affected']/1e9:.1f}B")
                            
                            # 受影响的协议
                            if len(contagion_result['affected_protocols']) > 1:
                                st.subheader("🔗 Contagion Spread")
                                affected_list = list(contagion_result['affected_protocols'])
                                st.write(f"**Initial shock:** {largest_protocol.upper()}")
                                if len(affected_list) > 1:
                                    other_affected = [p for p in affected_list if p != largest_protocol]
                                    st.write(f"**Also affected:** {', '.join([p.upper() for p in other_affected])}")
                            else:
                                st.info("🛡️ Contagion contained - no spread to other protocols")
        
        # 协议详细数据表
        st.header("📋 Protocol Details")
        
        if protocols:
            # 创建详细数据表
            protocol_df = pd.DataFrame([
                {
                    'Protocol': p['protocol'].upper(),
                    'Chain': p['chain'].upper(),
                    'Category': p.get('metadata', {}).get('category', 'unknown').upper(),
                    'TVL (USD)': f"${p['tvl_usd']:,.0f}",
                    'TVL (B)': f"{p['tvl_usd']/1e9:.2f}",
                }
                for p in protocols
            ])
            
            st.dataframe(
                protocol_df,
                use_container_width=True,
                hide_index=True
            )
            
            # 下载数据
            csv = protocol_df.to_csv(index=False)
            st.download_button(
                label="💾 Download Protocol Data",
                data=csv,
                file_name=f"defi_protocols_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    else:
        st.warning("No protocol data available")
    
    # 页脚信息
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("**OnChain Macro Analysis** - Enhanced Dashboard 🚀")
    
    with col2:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    with col3:
        st.markdown(f"*Last updated: {datetime.now().strftime('%H:%M:%S')}*")

if __name__ == "__main__":
    main()
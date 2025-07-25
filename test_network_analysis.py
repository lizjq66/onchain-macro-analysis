# test_network_analysis.py
#!/usr/bin/env python3
"""
网络分析功能测试脚本
演示DeFi协议网络分析的完整流程
"""

import asyncio
import sys
import os
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.analysis.network_analysis.protocol_network import protocol_network_analyzer
from src.data.data_pipeline import data_pipeline
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def create_sample_protocol_data():
    """创建样本协议数据用于测试"""
    sample_data = [
        {
            'protocol': 'uniswap',
            'chain': 'ethereum',
            'tvl_usd': 5000000000,  # 50亿
            'volume_24h': 1000000000,  # 10亿
            'metadata': {'category': 'dex'}
        },
        {
            'protocol': 'aave',
            'chain': 'ethereum', 
            'tvl_usd': 8000000000,  # 80亿
            'volume_24h': 500000000,  # 5亿
            'metadata': {'category': 'lending'}
        },
        {
            'protocol': 'compound',
            'chain': 'ethereum',
            'tvl_usd': 3000000000,  # 30亿
            'volume_24h': 200000000,  # 2亿
            'metadata': {'category': 'lending'}
        },
        {
            'protocol': 'makerdao',
            'chain': 'ethereum',
            'tvl_usd': 6000000000,  # 60亿
            'volume_24h': 100000000,  # 1亿
            'metadata': {'category': 'stablecoin'}
        },
        {
            'protocol': 'curve',
            'chain': 'ethereum',
            'tvl_usd': 4000000000,  # 40亿
            'volume_24h': 800000000,  # 8亿
            'metadata': {'category': 'dex'}
        },
        {
            'protocol': 'lido',
            'chain': 'ethereum',
            'tvl_usd': 15000000000,  # 150亿
            'volume_24h': 50000000,   # 5000万
            'metadata': {'category': 'staking'}
        },
        {
            'protocol': 'pancakeswap',
            'chain': 'bsc',
            'tvl_usd': 2000000000,  # 20亿
            'volume_24h': 400000000,  # 4亿
            'metadata': {'category': 'dex'}
        },
        {
            'protocol': 'aave',
            'chain': 'polygon',
            'tvl_usd': 500000000,   # 5亿
            'volume_24h': 50000000,  # 5000万
            'metadata': {'category': 'lending'}
        },
        {
            'protocol': 'uniswap',
            'chain': 'arbitrum',
            'tvl_usd': 1000000000,  # 10亿
            'volume_24h': 200000000,  # 2亿
            'metadata': {'category': 'dex'}
        },
        {
            'protocol': 'gmx',
            'chain': 'arbitrum',
            'tvl_usd': 800000000,   # 8亿
            'volume_24h': 300000000,  # 3亿
            'metadata': {'category': 'derivatives'}
        }
    ]
    
    return sample_data

async def test_network_construction():
    """测试网络构建"""
    print("\n🔗 Testing Network Construction")
    print("="*50)
    
    try:
        # 获取样本数据
        sample_data = create_sample_protocol_data()
        print(f"📊 Created {len(sample_data)} sample protocols")
        
        # 构建网络
        network = protocol_network_analyzer.build_network_from_data(
            sample_data, 
            tvl_threshold=100000000  # 1亿美元阈值
        )
        
        print(f"✅ Network built successfully:")
        print(f"   - Nodes: {network.number_of_nodes()}")
        print(f"   - Edges: {network.number_of_edges()}")
        
        # 显示网络摘要
        summary = protocol_network_analyzer.get_network_summary()
        print(f"\n📈 Network Summary:")
        print(f"   - Total TVL: ${summary['total_tvl']:,.0f}")
        print(f"   - Protocols by chain:")
        for chain, data in summary['protocols_by_chain'].items():
            print(f"     • {chain}: {data['count']} protocols, ${data['tvl']:,.0f} TVL")
        
        return True
        
    except Exception as e:
        print(f"❌ Network construction failed: {e}")
        return False

def test_centrality_analysis():
    """测试中心性分析"""
    print("\n📊 Testing Centrality Analysis")
    print("="*50)
    
    try:
        # 运行中心性分析
        centrality_results = protocol_network_analyzer.analyze_network_centrality()
        
        if not centrality_results:
            print("❌ No centrality results")
            return False
        
        print(f"✅ Centrality analysis completed for {len(centrality_results)} protocols")
        
        # 显示前5个协议的中心性指标
        print(f"\n🎯 Top 5 Protocols by Centrality:")
        sorted_protocols = sorted(
            centrality_results.items(),
            key=lambda x: x[1]['degree_centrality'] + x[1]['betweenness_centrality'],
            reverse=True
        )
        
        for i, (protocol, metrics) in enumerate(sorted_protocols[:5]):
            print(f"   {i+1}. {protocol.upper()}:")
            print(f"      • Degree Centrality: {metrics['degree_centrality']:.3f}")
            print(f"      • Betweenness Centrality: {metrics['betweenness_centrality']:.3f}")
            print(f"      • Closeness Centrality: {metrics['closeness_centrality']:.3f}")
            print(f"      • TVL: ${metrics['tvl_usd']:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Centrality analysis failed: {e}")
        return False

def test_critical_protocols():
    """测试关键协议识别"""
    print("\n🎯 Testing Critical Protocols Identification")
    print("="*50)
    
    try:
        # 识别关键协议
        critical_protocols = protocol_network_analyzer.identify_critical_protocols(top_n=5)
        
        if not critical_protocols:
            print("❌ No critical protocols identified")
            return False
        
        print(f"✅ Identified {len(critical_protocols)} critical protocols:")
        
        for i, protocol_info in enumerate(critical_protocols):
            print(f"\n   {i+1}. {protocol_info['protocol'].upper()} ({protocol_info['chain']})")
            print(f"      • Importance Score: {protocol_info['importance_score']:.3f}")
            print(f"      • TVL: ${protocol_info['tvl_usd']:,.0f}")
            print(f"      • Centrality Metrics:")
            metrics = protocol_info['centrality_metrics']
            print(f"        - Degree: {metrics['degree_centrality']:.3f}")
            print(f"        - Betweenness: {metrics['betweenness_centrality']:.3f}")
            print(f"        - Closeness: {metrics['closeness_centrality']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Critical protocols identification failed: {e}")
        return False

def test_systemic_risk():
    """测试系统性风险分析"""
    print("\n⚠️ Testing Systemic Risk Analysis")
    print("="*50)
    
    try:
        # 分析系统性风险
        risk_analysis = protocol_network_analyzer.analyze_systemic_risk()
        
        if not risk_analysis:
            print("❌ No risk analysis results")
            return False
        
        print(f"✅ Systemic risk analysis completed:")
        print(f"   • Network Density: {risk_analysis['network_density']:.3f}")
        print(f"   • Clustering Coefficient: {risk_analysis['clustering_coefficient']:.3f}")
        print(f"   • Connected Components: {risk_analysis['connected_components']}")
        print(f"   • Largest Component Size: {risk_analysis['largest_component_size']}")
        print(f"   • Overall Risk Score: {risk_analysis['risk_score']:.3f}")
        
        if risk_analysis['vulnerabilities']:
            print(f"\n⚠️ Identified Vulnerabilities:")
            for vuln in risk_analysis['vulnerabilities']:
                print(f"   • {vuln}")
        else:
            print(f"\n✅ No major vulnerabilities identified")
        
        return True
        
    except Exception as e:
        print(f"❌ Systemic risk analysis failed: {e}")
        return False

def test_contagion_simulation():
    """测试传染效应模拟"""
    print("\n🦠 Testing Contagion Simulation")
    print("="*50)
    
    try:
        # 选择一个高TVL协议作为初始冲击
        initial_shock = ['lido']  # Lido有最高的TVL
        
        # 运行传染模拟
        contagion_result = protocol_network_analyzer.simulate_contagion(
            initial_shock_protocols=initial_shock,
            shock_magnitude=0.6
        )
        
        print(f"✅ Contagion simulation completed:")
        print(f"   • Initial Shock Protocols: {', '.join(contagion_result['initial_protocols'])}")
        print(f"   • Total Affected Protocols: {len(contagion_result['affected_protocols'])}")
        print(f"   • Total TVL Affected: ${contagion_result['total_tvl_affected']:,.0f}")
        print(f"   • Contagion Rounds: {len(contagion_result['contagion_rounds'])}")
        print(f"   • Contagion Stopped: {contagion_result['contagion_stopped']}")
        
        if contagion_result['contagion_rounds']:
            print(f"\n📈 Contagion Spread:")
            for round_info in contagion_result['contagion_rounds']:
                print(f"   Round {round_info['round']}: {len(round_info['new_affected'])} new protocols affected")
                print(f"     New: {', '.join(round_info['new_affected'])}")
        
        # 计算影响比例
        total_protocols = len(protocol_network_analyzer.protocol_data)
        if total_protocols > 0:
            impact_ratio = len(contagion_result['affected_protocols']) / total_protocols
            print(f"\n📊 Impact Analysis:")
            print(f"   • Protocols Affected: {len(contagion_result['affected_protocols'])}/{total_protocols} ({impact_ratio:.1%})")
        
        return True
        
    except Exception as e:
        print(f"❌ Contagion simulation failed: {e}")
        return False

def test_data_export():
    """测试数据导出"""
    print("\n💾 Testing Data Export")
    print("="*50)
    
    try:
        # 导出网络数据
        export_data = protocol_network_analyzer.export_network_data()
        
        print(f"✅ Network data exported successfully:")
        print(f"   • Network Summary: {len(export_data['network_summary'])} fields")
        print(f"   • Protocol Data: {len(export_data['protocol_data'])} protocols")
        print(f"   • Analysis Results: {len(export_data['analysis_results'])} analyses")
        print(f"   • Network Edges: {len(export_data['network_edges'])} edges")
        
        # 显示分析结果类型
        if export_data['analysis_results']:
            print(f"   • Available Analyses: {', '.join(export_data['analysis_results'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data export failed: {e}")
        return False

async def main():
    """主测试函数"""
    print("🔬 OnChain Macro Analysis - Network Analysis Test")
    print("="*60)
    
    test_results = {}
    
    # 运行所有测试
    tests = [
        ("Network Construction", test_network_construction),
        ("Centrality Analysis", test_centrality_analysis),
        ("Critical Protocols", test_critical_protocols), 
        ("Systemic Risk", test_systemic_risk),
        ("Contagion Simulation", test_contagion_simulation),
        ("Data Export", test_data_export)
    ]
    
    passed_tests = 0
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            test_results[test_name] = result
            if result:
                passed_tests += 1
                
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with error: {e}")
            test_results[test_name] = False
    
    # 测试总结
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall Result: {passed_tests}/{len(tests)} tests passed")
    
    if passed_tests == len(tests):
        print("\n🎉 All network analysis tests passed!")
        print("🚀 Your DeFi network analysis system is ready!")
        print("\n📖 What you can do now:")
        print("   1. Analyze real DeFi protocol relationships")
        print("   2. Identify systemic risks in the ecosystem")
        print("   3. Simulate market contagion effects")
        print("   4. Find critical protocols for macro analysis")
    else:
        print(f"\n⚠️ {len(tests) - passed_tests} tests failed. Check the output above.")
    
    input("\nPress Enter to exit...")
    return 0 if passed_tests == len(tests) else 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Test execution failed: {e}")
        sys.exit(1)
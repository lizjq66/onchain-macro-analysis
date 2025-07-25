# run_project.py
#!/usr/bin/env python3
"""
项目运行脚本 - 提供统一的项目管理入口
"""

import argparse
import asyncio
import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """设置环境变量和路径"""
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    os.chdir(project_root)

def run_setup():
    """运行项目设置"""
    print("🔧 Running project setup...")
    try:
        subprocess.run([sys.executable, "setup_and_test.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Setup failed: {e}")
        return False
    return True

def run_tests():
    """运行项目测试"""
    print("🧪 Running project tests...")
    try:
        result = subprocess.run([sys.executable, "test_pipeline.py"], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Tests failed: {e}")
        return False

async def run_data_collection():
    """运行数据收集"""
    print("📊 Running data collection...")
    try:
        from scripts.daily_update import main as daily_main
        result = await daily_main()
        return result == 0
    except Exception as e:
        print(f"Data collection failed: {e}")
        return False

def run_dashboard():
    """启动仪表板"""
    print("🚀 Starting dashboard...")
    dashboard_file = Path("src/visualization/dashboard.py")
    
    if not dashboard_file.exists():
        print("⚠️ Dashboard file not found. Creating basic dashboard...")
        create_basic_dashboard()
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_file), "--server.port=8501"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Dashboard failed to start: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    return True

def create_basic_dashboard():
    """创建基础仪表板"""
    dashboard_content = '''import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.data_pipeline import data_pipeline
from src.data.storage.database import db_manager

st.set_page_config(
    page_title="OnChain Macro Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 OnChain Macro Trends Analysis")
st.markdown("Real-time analysis of blockchain data and macroeconomic indicators")

# Sidebar
st.sidebar.header("Control Panel")

if st.sidebar.button("🔄 Refresh Data"):
    with st.spinner("Collecting latest data..."):
        try:
            import asyncio
            result = asyncio.run(data_pipeline.run_full_pipeline())
            if result['errors']:
                st.sidebar.error(f"Collection completed with {len(result['errors'])} errors")
            else:
                st.sidebar.success("Data collection successful!")
        except Exception as e:
            st.sidebar.error(f"Data collection failed: {e}")

# Pipeline Status
st.header("📈 Pipeline Status")
try:
    status = data_pipeline.get_pipeline_status()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Status", status.get('status', 'Unknown'))
    with col2:
        st.metric("Sources", len(status.get('sources_processed', [])))
    with col3:
        st.metric("Records", status.get('total_records', 0))
    with col4:
        st.metric("Errors", status.get('error_count', 0))
    
    if status.get('sources_processed'):
        st.write("**Processed Sources:**", ", ".join(status['sources_processed']))
    
except Exception as e:
    st.error(f"Could not load pipeline status: {e}")

# Recent Data
st.header("📊 Recent Data")

try:
    # Try to get recent data from database
    chain_metrics = db_manager.get_latest_metrics('chain_metrics', limit=50)
    protocol_metrics = db_manager.get_latest_metrics('protocol_metrics', limit=50)
    macro_indicators = db_manager.get_latest_metrics('macro_indicators', limit=50)
    
    tab1, tab2, tab3 = st.tabs(["Chain Metrics", "Protocol Metrics", "Macro Indicators"])
    
    with tab1:
        if chain_metrics:
            df = pd.DataFrame(chain_metrics)
            st.subheader("Chain Metrics")
            st.dataframe(df.head(10))
            
            if 'gas_price_gwei' in df.columns and not df['gas_price_gwei'].isna().all():
                fig = px.line(df, x='timestamp', y='gas_price_gwei', title='Gas Price Trend')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No chain metrics data available. Run data collection first.")
    
    with tab2:
        if protocol_metrics:
            df = pd.DataFrame(protocol_metrics)
            st.subheader("Protocol Metrics")
            st.dataframe(df.head(10))
            
            if 'tvl_usd' in df.columns and not df['tvl_usd'].isna().all():
                fig = px.bar(df.head(10), x='protocol', y='tvl_usd', title='TVL by Protocol')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No protocol metrics data available. Run data collection first.")
    
    with tab3:
        if macro_indicators:
            df = pd.DataFrame(macro_indicators)
            st.subheader("Macro Indicators")
            st.dataframe(df.head(10))
            
            if len(df) > 0:
                fig = px.scatter(df, x='timestamp', y='value', color='indicator', 
                               title='Macro Indicators Timeline')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No macro indicators data available. Run data collection first.")

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure the database is set up and data has been collected.")

# Footer
st.markdown("---")
st.markdown("**OnChain Macro Analysis** - Built with Streamlit 🚀")
'''
    
    dashboard_path = Path("src/visualization/dashboard.py")
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(dashboard_content)
    
    print("✅ Basic dashboard created")

def run_analysis():
    """运行分析脚本"""
    print("🔍 Running analysis...")
    
    # 检查是否有Jupyter notebooks
    notebooks_dir = Path("notebooks")
    if notebooks_dir.exists():
        notebooks = list(notebooks_dir.glob("*.ipynb"))
        if notebooks:
            print(f"📓 Found {len(notebooks)} notebooks:")
            for nb in notebooks:
                print(f"  - {nb.name}")
            
            print("\nTo run notebooks:")
            print("jupyter notebook notebooks/")
        else:
            print("No notebooks found")
    else:
        print("Notebooks directory not found")

def show_status():
    """显示项目状态"""
    print("📊 Project Status")
    print("="*30)
    
    # 检查关键文件
    key_files = [
        ".env",
        "src/data/data_pipeline.py",
        "src/data/storage/database.py",
        "requirements.txt"
    ]
    
    for file_path in key_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
    
    # 检查数据目录
    data_dirs = ["data/raw", "data/processed", "logs"]
    for dir_path in data_dirs:
        if Path(dir_path).exists():
            files_count = len(list(Path(dir_path).glob("*")))
            print(f"📁 {dir_path} ({files_count} files)")
        else:
            print(f"❌ {dir_path}")
    
    # 检查数据库状态
    try:
        setup_environment()
        from src.data.storage.database import db_manager
        from src.data.data_pipeline import data_pipeline
        
        # 检查管道状态
        status = data_pipeline.get_pipeline_status()
        if status.get('status') != 'No pipeline runs found':
            print(f"🔄 Last pipeline run: {status.get('latest_run', 'Unknown')}")
            print(f"📈 Records processed: {status.get('total_records', 0)}")
        else:
            print("⚠️ No pipeline runs found")
    
    except Exception as e:
        print(f"❌ Could not check pipeline status: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="OnChain Macro Analysis Project Runner")
    parser.add_argument("command", choices=[
        "setup", "test", "collect", "dashboard", "analysis", "status", "all"
    ], help="Command to run")
    
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    
    success = True
    
    if args.command == "setup":
        success = run_setup()
    
    elif args.command == "test":
        success = run_tests()
    
    elif args.command == "collect":
        success = asyncio.run(run_data_collection())
    
    elif args.command == "dashboard":
        success = run_dashboard()
    
    elif args.command == "analysis":
        run_analysis()
    
    elif args.command == "status":
        show_status()
    
    elif args.command == "all":
        print("🚀 Running complete project workflow...")
        
        # 1. Setup
        if not run_setup():
            print("❌ Setup failed")
            return 1
        
        # 2. Tests
        if not run_tests():
            print("⚠️ Tests failed, but continuing...")
        
        # 3. Data collection
        if not asyncio.run(run_data_collection()):
            print("⚠️ Data collection failed, but continuing...")
        
        # 4. Show status
        show_status()
        
        # 5. Ask about dashboard
        start_dashboard = input("\n🚀 Start dashboard? (y/n): ").lower().strip() == 'y'
        if start_dashboard:
            run_dashboard()
    
    return 0 if success else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

# ---

# 创建简化的启动脚本
# start.sh (Linux/Mac)
"""
#!/bin/bash
echo "🚀 OnChain Macro Analysis - Quick Start"

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required"
    exit 1
fi

# 检查虚拟环境
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️ Virtual environment recommended"
    echo "Run: python -m venv venv && source venv/bin/activate"
fi

# 运行项目
python3 run_project.py all
"""

# start.bat (Windows)
"""
@echo off
echo 🚀 OnChain Macro Analysis - Quick Start

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is required
    pause
    exit /b 1
)

REM 检查虚拟环境
if "%VIRTUAL_ENV%"=="" (
    echo ⚠️ Virtual environment recommended
    echo Run: python -m venv venv && venv\Scripts\activate
)

REM 运行项目
python run_project.py all
pause
"""
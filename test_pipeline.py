# test_pipeline.py
#!/usr/bin/env python3
"""
数据管道测试脚本
用于验证整个数据收集和处理流程
"""

import asyncio
import os
import sys
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.data_pipeline import data_pipeline
from src.data.storage.database import db_manager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

async def test_individual_collectors():
    """测试各个数据收集器"""
    logger.info("=== Testing Individual Collectors ===")
    
    test_results = {}
    
    # 测试DeFi收集器
    try:
        logger.info("Testing DeFi collector...")
        defi_data = await data_pipeline.collectors['defi'].collect_data()
        test_results['defi'] = {
            'success': True,
            'records': len(defi_data) if isinstance(defi_data, list) else 0,
            'sample': defi_data[:2] if defi_data else []
        }
        logger.info(f"✓ DeFi collector: {test_results['defi']['records']} records")
    except Exception as e:
        test_results['defi'] = {'success': False, 'error': str(e)}
        logger.error(f"✗ DeFi collector failed: {e}")
    
    # 测试Gas收集器
    try:
        logger.info("Testing Gas collector...")
        gas_data = await data_pipeline.collectors['gas'].collect_data()
        test_results['gas'] = {
            'success': True,
            'records': len(gas_data) if isinstance(gas_data, list) else 0,
            'sample': gas_data[:2] if gas_data else []
        }
        logger.info(f"✓ Gas collector: {test_results['gas']['records']} records")
    except Exception as e:
        test_results['gas'] = {'success': False, 'error': str(e)}
        logger.error(f"✗ Gas collector failed: {e}")
    
    # 测试宏观数据收集器
    try:
        logger.info("Testing Macro collector...")
        macro_data = await data_pipeline.collectors['macro'].collect_data()
        test_results['macro'] = {
            'success': True,
            'records': len(macro_data) if isinstance(macro_data, list) else 0,
            'sample': macro_data[:2] if macro_data else []
        }
        logger.info(f"✓ Macro collector: {test_results['macro']['records']} records")
    except Exception as e:
        test_results['macro'] = {'success': False, 'error': str(e)}
        logger.error(f"✗ Macro collector failed: {e}")
    
    # 测试地址活跃度收集器
    try:
        logger.info("Testing Address collector...")
        address_data = await data_pipeline.collectors['address'].collect_data()
        test_results['address'] = {
            'success': True,
            'records': len(address_data) if isinstance(address_data, list) else 0,
            'sample': address_data[:2] if address_data else []
        }
        logger.info(f"✓ Address collector: {test_results['address']['records']} records")
    except Exception as e:
        test_results['address'] = {'success': False, 'error': str(e)}
        logger.error(f"✗ Address collector failed: {e}")
    
    return test_results

def test_data_processing():
    """测试数据处理组件"""
    logger.info("=== Testing Data Processing Components ===")
    
    # 创建测试数据
    test_data = [
        {
            'data_type': 'protocol_metrics',
            'protocol': 'uniswap',
            'chain': 'ethereum',
            'timestamp': datetime.utcnow(),
            'tvl_usd': 1000000.0,
            'volume_24h': 50000.0,
            'apy': 5.5,
            'metadata': {'test': True}
        },
        {
            'data_type': 'gas_metrics',
            'chain': 'ethereum',
            'timestamp': datetime.utcnow(),
            'gas_price_gwei': 25.0,
            'metadata': {'test': True}
        },
        {
            'data_type': 'macro_indicator',
            'indicator': 'federal_funds_rate',
            'timestamp': datetime.utcnow(),
            'value': 5.25,
            'source': 'test',
            'frequency': 'daily',
            'metadata': {'test': True}
        }
    ]
    
    processing_results = {}
    
    # 测试数据清洗
    try:
        logger.info("Testing data cleaner...")
        cleaned_data = data_pipeline.data_cleaner.clean_metrics_data(test_data)
        processing_results['cleaning'] = {
            'success': True,
            'input_records': len(test_data),
            'output_records': len(cleaned_data),
            'sample': cleaned_data[:1] if cleaned_data else []
        }
        logger.info(f"✓ Data cleaning: {len(cleaned_data)}/{len(test_data)} records passed")
    except Exception as e:
        processing_results['cleaning'] = {'success': False, 'error': str(e)}
        logger.error(f"✗ Data cleaning failed: {e}")
    
    # 测试特征工程
    try:
        logger.info("Testing feature engineering...")
        if 'cleaning' in processing_results and processing_results['cleaning']['success']:
            engineered_df = data_pipeline.feature_engineer.engineer_features(cleaned_data)
            processing_results['feature_engineering'] = {
                'success': True,
                'rows': len(engineered_df),
                'columns': len(engineered_df.columns) if not engineered_df.empty else 0,
                'features': list(engineered_df.columns) if not engineered_df.empty else []
            }
            logger.info(f"✓ Feature engineering: {len(engineered_df)} rows, {len(engineered_df.columns) if not engineered_df.empty else 0} features")
        else:
            raise Exception("Skipped due to cleaning failure")
    except Exception as e:
        processing_results['feature_engineering'] = {'success': False, 'error': str(e)}
        logger.error(f"✗ Feature engineering failed: {e}")
    
    # 测试数据验证
    try:
        logger.info("Testing data validation...")
        if 'feature_engineering' in processing_results and processing_results['feature_engineering']['success']:
            validation_result = data_pipeline.data_validator.validate_dataset(engineered_df, "test_dataset")
            processing_results['validation'] = {
                'success': True,
                'validation_passed': validation_result['validation_passed'],
                'issues_found': len(validation_result['issues']),
                'issues': validation_result['issues']
            }
            logger.info(f"✓ Data validation: {'PASSED' if validation_result['validation_passed'] else 'FAILED'}")
        else:
            raise Exception("Skipped due to feature engineering failure")
    except Exception as e:
        processing_results['validation'] = {'success': False, 'error': str(e)}
        logger.error(f"✗ Data validation failed: {e}")
    
    return processing_results

def test_database_connection():
    """测试数据库连接"""
    logger.info("=== Testing Database Connection ===")
    
    try:
        # 创建表
        db_manager.create_tables()
        logger.info("✓ Database tables created successfully")
        
        # 测试插入数据
        test_chain_metrics = [{
            'chain': 'ethereum',
            'timestamp': datetime.utcnow(),
            'tvl_usd': 1000000.0,
            'gas_price_gwei': 25.0,
            'metadata': {'test': True}
        }]
        
        success = db_manager.insert_chain_metrics(test_chain_metrics)
        if success:
            logger.info("✓ Database insertion test passed")
            return {'success': True}
        else:
            logger.error("✗ Database insertion test failed")
            return {'success': False, 'error': 'Insertion failed'}
    except Exception as e:
        logger.error(f"✗ Database connection test failed: {e}")
        return {'success': False, 'error': str(e)}

def print_test_summary(results):
    """打印测试结果摘要"""
    logger.info("=== TEST SUMMARY ===")
    
    total_tests = 0
    passed_tests = 0
    
    for category, result in results.items():
        logger.info(f"\n{category.upper()}:")
        
        if isinstance(result, dict):
            if result.get('success', False):
                logger.info(f"  ✓ PASSED")
                passed_tests += 1
            else:
                logger.info(f"  ✗ FAILED: {result.get('error', 'Unknown error')}")
            total_tests += 1
        elif isinstance(result, dict):
            # 处理嵌套结果（如collectors测试）
            for subtest, subresult in result.items():
                if isinstance(subresult, dict):
                    if subresult.get('success', False):
                        logger.info(f"  ✓ {subtest}: PASSED ({subresult.get('records', 0)} records)")
                        passed_tests += 1
                    else:
                        logger.info(f"  ✗ {subtest}: FAILED - {subresult.get('error', 'Unknown error')}")
                    total_tests += 1
    
    logger.info(f"\nOVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! Your data pipeline is ready.")
    else:
        logger.warning(f"⚠️ {total_tests - passed_tests} tests failed. Check configuration and API keys.")
    
    return passed_tests == total_tests

def check_environment():
    """检查环境配置"""
    logger.info("=== Environment Check ===")
    
    from config.settings import settings
    
    env_status = {
        'python_version': sys.version,
        'working_directory': os.getcwd(),
        'config_loaded': True
    }
    
    # 检查API密钥
    api_keys = {
        'INFURA_PROJECT_ID': getattr(settings, 'INFURA_PROJECT_ID', ''),
        'ETHERSCAN_API_KEY': getattr(settings, 'ETHERSCAN_API_KEY', ''),
        'FRED_API_KEY': getattr(settings, 'FRED_API_KEY', ''),
        'COINGECKO_API_KEY': getattr(settings, 'COINGECKO_API_KEY', ''),
    }
    
    configured_apis = []
    missing_apis = []
    
    for key, value in api_keys.items():
        if value and value.strip():
            configured_apis.append(key)
        else:
            missing_apis.append(key)
    
    logger.info(f"✓ Configured APIs: {', '.join(configured_apis) if configured_apis else 'None'}")
    if missing_apis:
        logger.warning(f"⚠️ Missing APIs: {', '.join(missing_apis)}")
        logger.info("Note: Some tests may be skipped or use mock data")
    
    env_status['configured_apis'] = configured_apis
    env_status['missing_apis'] = missing_apis
    
    return env_status

async def main():
    """主测试函数"""
    start_time = datetime.utcnow()
    logger.info("=== STARTING DATA PIPELINE TESTS ===")
    
    # 环境检查
    env_status = check_environment()
    
    test_results = {}
    
    # 1. 测试各个收集器
    test_results['collectors'] = await test_individual_collectors()
    
    # 2. 测试数据处理
    test_results['processing'] = test_data_processing()
    
    # 3. 测试数据库连接
    test_results['database'] = test_database_connection()
    
    # 4. 测试完整管道
    test_results['full_pipeline'] = await test_full_pipeline()
    
    # 打印测试摘要
    all_passed = print_test_summary(test_results)
    
    end_time = datetime.utcnow()
    execution_time = (end_time - start_time).total_seconds()
    
    logger.info(f"\nTest execution completed in {execution_time:.2f} seconds")
    
    # 生成详细报告
    generate_test_report(test_results, env_status, execution_time)
    
    return 0 if all_passed else 1

def generate_test_report(test_results, env_status, execution_time):
    """生成详细的测试报告"""
    report_lines = [
        "="*50,
        "DATA PIPELINE TEST REPORT",
        "="*50,
        f"Test Date: {datetime.utcnow().isoformat()}",
        f"Execution Time: {execution_time:.2f} seconds",
        f"Python Version: {env_status['python_version']}",
        "",
        "ENVIRONMENT STATUS:",
        f"  Configured APIs: {', '.join(env_status['configured_apis']) if env_status['configured_apis'] else 'None'}",
        f"  Missing APIs: {', '.join(env_status['missing_apis']) if env_status['missing_apis'] else 'None'}",
        "",
        "TEST RESULTS:",
    ]
    
    # 详细结果
    for category, results in test_results.items():
        report_lines.append(f"\n{category.upper()}:")
        
        if category == 'collectors':
            for collector, result in results.items():
                status = "PASSED" if result.get('success') else "FAILED"
                records = result.get('records', 0)
                report_lines.append(f"  {collector}: {status} ({records} records)")
                if not result.get('success'):
                    report_lines.append(f"    Error: {result.get('error', 'Unknown')}")
        
        elif category == 'processing':
            for process, result in results.items():
                status = "PASSED" if result.get('success') else "FAILED"
                report_lines.append(f"  {process}: {status}")
                if not result.get('success'):
                    report_lines.append(f"    Error: {result.get('error', 'Unknown')}")
        
        elif category == 'database':
            status = "PASSED" if results.get('success') else "FAILED"
            report_lines.append(f"  Connection: {status}")
            if not results.get('success'):
                report_lines.append(f"    Error: {results.get('error', 'Unknown')}")
        
        elif category == 'full_pipeline':
            status = "PASSED" if results.get('success') else "FAILED"
            report_lines.append(f"  Pipeline: {status}")
            if results.get('success'):
                report_lines.append(f"    Sources: {', '.join(results.get('sources_processed', []))}")
                report_lines.append(f"    Records: {results.get('total_records', 0)}")
                report_lines.append(f"    Errors: {results.get('error_count', 0)}")
            else:
                report_lines.append(f"    Error: {results.get('error', 'Unknown')}")
    
    report_content = "\n".join(report_lines)
    
    # 保存报告到文件
    report_file = f"test_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
    try:
        with open(report_file, 'w') as f:
            f.write(report_content)
        logger.info(f"📄 Detailed test report saved to: {report_file}")
    except Exception as e:
        logger.warning(f"Could not save test report: {e}")
    
    # 也输出到控制台
    print("\n" + report_content)

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)

# ---

# setup_and_test.py
#!/usr/bin/env python3
"""
项目设置和测试一体化脚本
包含环境设置、依赖安装检查和基础测试
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_virtual_environment():
    """检查是否在虚拟环境中"""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print("✅ Virtual environment detected")
        return True
    else:
        print("⚠️ Not in virtual environment (recommended)")
        return False

def check_dependencies():
    """检查关键依赖"""
    required_packages = [
        'pandas', 'numpy', 'aiohttp', 'sqlalchemy', 
        'web3', 'yfinance', 'plotly', 'streamlit'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_environment_file():
    """检查.env文件"""
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if not env_file.exists():
        if env_example.exists():
            print("⚠️ .env file not found")
            print("📝 Copy .env.example to .env and configure your API keys:")
            print("cp .env.example .env")
            return False
        else:
            print("❌ Neither .env nor .env.example found")
            return False
    else:
        print("✅ .env file found")
        return True

def create_directories():
    """创建必要的目录"""
    dirs = ['data/raw', 'data/processed', 'logs']
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created/verified: {dir_path}")

async def run_basic_tests():
    """运行基础测试"""
    print("\n🧪 Running basic tests...")
    
    try:
        # 导入测试模块
        from test_pipeline import test_data_processing, test_database_connection
        
        # 测试数据处理
        processing_result = test_data_processing()
        if processing_result.get('cleaning', {}).get('success'):
            print("✅ Data processing test passed")
        else:
            print("❌ Data processing test failed")
        
        # 测试数据库
        db_result = test_database_connection()
        if db_result.get('success'):
            print("✅ Database test passed")
        else:
            print("❌ Database test failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic tests failed: {e}")
        return False

def main():
    """主设置函数"""
    print("🚀 OnChain Macro Analysis - Setup & Test")
    print("="*50)
    
    # 环境检查
    checks_passed = 0
    total_checks = 5
    
    if check_python_version():
        checks_passed += 1
    
    if check_virtual_environment():
        checks_passed += 1
    
    if check_dependencies():
        checks_passed += 1
    
    if check_environment_file():
        checks_passed += 1
    
    # 创建目录
    create_directories()
    checks_passed += 1
    
    print(f"\n📊 Environment Check: {checks_passed}/{total_checks} passed")
    
    if checks_passed == total_checks:
        print("\n✅ Environment setup complete!")
        
        # 询问是否运行测试
        run_tests = input("\n🧪 Run basic tests? (y/n): ").lower().strip() == 'y'
        
        if run_tests:
            test_success = asyncio.run(run_basic_tests())
            
            if test_success:
                print("\n🎉 Setup and tests completed successfully!")
                print("\n📖 Next steps:")
                print("1. Configure API keys in .env file")
                print("2. Run: python test_pipeline.py")
                print("3. Run: python scripts/daily_update.py")
                print("4. Start dashboard: streamlit run src/visualization/dashboard.py")
            else:
                print("\n⚠️ Setup complete, but some tests failed")
        else:
            print("\n✅ Setup complete. You can run tests later with: python test_pipeline.py")
    
    else:
        print(f"\n❌ Environment setup incomplete ({total_checks - checks_passed} issues)")
        print("Please fix the issues above and run again.")

if __name__ == "__main__":
    main()

async def test_full_pipeline():
    """测试完整数据管道"""
    logger.info("=== Testing Full Data Pipeline ===")
    
    try:
        # 只测试有API key的数据源
        available_sources = []
        
        # 检查哪些数据源可用
        from config.settings import settings
        
        if hasattr(settings, 'ETHERSCAN_API_KEY') and settings.ETHERSCAN_API_KEY:
            available_sources.append('gas')
        
        if hasattr(settings, 'FRED_API_KEY') and settings.FRED_API_KEY:
            available_sources.append('macro')
        
        # DeFi数据通常不需要API key
        available_sources.append('defi')
        
        if not available_sources:
            logger.warning("No API keys configured, testing with mock data")
            available_sources = ['defi']  # 至少测试一个源
        
        logger.info(f"Testing pipeline with sources: {available_sources}")
        
        # 运行管道
        result = await data_pipeline.run_full_pipeline(sources=available_sources)
        
        pipeline_test_result = {
            'success': True,
            'sources_processed': result['sources_processed'],
            'total_records': result['total_records_processed'],
            'execution_time': result['execution_time_seconds'],
            'error_count': len(result['errors']),
            'errors': result['errors']
        }
        
        if result['errors']:
            logger.warning(f"✓ Pipeline completed with {len(result['errors'])} errors")
        else:
            logger.info("✓ Pipeline completed successfully")
        
        return pipeline_test_result
        
    except Exception as e:
        logger.error(f"✗ Full pipeline test failed: {e}")
        return {'success': False, 'error': str(e), 'sources_processed': available_sources, 'total_records': 0, 'execution_time': 0, 'errors': []}
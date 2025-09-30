"""
科大讯飞语音识别Python演示程序
展示基本语音识别功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iflytek_voice_recognition import FirefightingVoiceRecognition
from firefighting_voice_commands import FirefightingVoiceCommands
import json
import time

def demo_basic_recognition():
    """基本语音识别演示"""
    print("🎤 基本语音识别演示")
    print("=" * 40)
    
    # 配置信息（需要替换为实际的科大讯飞配置）
    config = {
        "app_id": "your_app_id_here",
        "api_key": "your_api_key_here",
        "api_secret": "your_api_secret_here"
    }
    
    # 初始化识别器
    recognizer = FirefightingVoiceRecognition(
        app_id=config["app_id"],
        api_key=config["api_key"],
        api_secret=config["api_secret"]
    )
    
    # 初始化识别器
    if not recognizer.initialize_recognizers():
        print("❌ 识别器初始化失败，请检查配置")
        return
    
    print("✅ 识别器初始化成功")
    
    # 录制音频
    print("\n开始录制音频（5秒）...")
    print("请说话...")
    audio_file = recognizer.record_audio(duration=5)
    
    # 在线识别
    print("\n进行在线识别...")
    online_result = recognizer.online_recognition(audio_file)
    print(f"在线识别结果: {json.dumps(online_result, ensure_ascii=False, indent=2)}")
    
    # 离线识别
    print("\n进行离线识别...")
    offline_result = recognizer.offline_recognition(audio_file)
    print(f"离线识别结果: {json.dumps(offline_result, ensure_ascii=False, indent=2)}")
    
    # 清理文件
    try:
        os.remove(audio_file)
    except:
        pass

def demo_firefighting_commands():
    """消防命令识别演示"""
    print("\n🔥 消防命令识别演示")
    print("=" * 40)
    
    # 初始化命令处理器
    command_processor = FirefightingVoiceCommands()
    
    # 示例消防命令
    firefighting_commands = [
        "发现火情，三楼东侧房间起火，烟雾很大，需要立即救援",
        "有人被困在二楼，需要紧急救援",
        "启动水枪，准备灭火",
        "火势已经控制，现场安全",
        "需要更多呼吸器和梯子",
        "疏散人员到安全区域"
    ]
    
    print("处理消防命令:")
    for i, command in enumerate(firefighting_commands, 1):
        print(f"\n{i}. 命令: {command}")
        result = command_processor.process_firefighting_command(command)
        
        print(f"   命令类型: {result['analysis']['command_type']}")
        print(f"   紧急程度: {result['analysis']['urgency_level']}")
        print(f"   设备提及: {', '.join(result['analysis']['equipment_mentioned'])}")
        print(f"   位置提及: {', '.join(result['analysis']['location_mentioned'])}")
        print(f"   紧急指标: {', '.join(result['analysis']['emergency_indicators'])}")
        print(f"   需要行动: {result['analysis']['action_required']}")
        print(f"   置信度: {result['analysis']['confidence']:.2f}")
        print(f"   响应: {result['response']['acknowledgment']}")

def demo_real_time_recognition():
    """实时语音识别演示"""
    print("\n🔄 实时语音识别演示")
    print("=" * 40)
    
    # 配置信息
    config = {
        "app_id": "your_app_id_here",
        "api_key": "your_api_key_here",
        "api_secret": "your_api_secret_here"
    }
    
    # 初始化识别器
    recognizer = FirefightingVoiceRecognition(
        app_id=config["app_id"],
        api_key=config["api_key"],
        api_secret=config["api_secret"]
    )
    
    if not recognizer.initialize_recognizers():
        print("❌ 识别器初始化失败")
        return
    
    print("开始实时语音识别（10秒）...")
    print("请说话...")
    
    # 实时识别回调
    def result_callback(result):
        print(f"识别结果: {result['text']}")
        if result.get('firefighting_analysis', {}).get('action_required'):
            print("🚨 检测到紧急情况，需要立即行动！")
    
    # 开始实时识别
    recognizer.real_time_recognition(result_callback, duration=10)
    recognizer.stop_real_time_recognition()
    
    print("实时识别结束")

def demo_batch_recognition():
    """批量识别演示"""
    print("\n📁 批量识别演示")
    print("=" * 40)
    
    # 配置信息
    config = {
        "app_id": "your_app_id_here",
        "api_key": "your_api_key_here",
        "api_secret": "your_api_secret_here"
    }
    
    # 初始化识别器
    recognizer = FirefightingVoiceRecognition(
        app_id=config["app_id"],
        api_key=config["api_key"],
        api_secret=config["api_secret"]
    )
    
    if not recognizer.initialize_recognizers():
        print("❌ 识别器初始化失败")
        return
    
    # 录制多个音频文件
    audio_files = []
    for i in range(3):
        print(f"录制第{i+1}个音频文件...")
        audio_file = recognizer.record_audio(duration=3)
        audio_files.append(audio_file)
        time.sleep(1)
    
    # 批量识别
    print("\n进行批量识别...")
    results = recognizer.batch_recognition(audio_files, use_offline=False)
    
    print("批量识别结果:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. 文件: {result['file']}")
        if 'result' in result:
            print(f"   识别结果: {result['result']['text']}")
            print(f"   置信度: {result['result']['confidence']}")
        else:
            print(f"   错误: {result['error']}")
    
    # 清理文件
    for audio_file in audio_files:
        try:
            os.remove(audio_file)
        except:
            pass

def main():
    """主函数"""
    print("🚀 科大讯飞语音识别演示程序")
    print("=" * 50)
    
    while True:
        print("\n请选择演示功能:")
        print("1. 基本语音识别")
        print("2. 消防命令识别")
        print("3. 实时语音识别")
        print("4. 批量识别")
        print("0. 退出")
        
        choice = input("请输入选择 (0-4): ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            demo_basic_recognition()
        elif choice == "2":
            demo_firefighting_commands()
        elif choice == "3":
            demo_real_time_recognition()
        elif choice == "4":
            demo_batch_recognition()
        else:
            print("❌ 无效选择")
    
    print("\n👋 演示结束，谢谢使用！")

if __name__ == "__main__":
    main()




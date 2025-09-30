"""
简化版语音识别演示程序
不依赖外部SDK，用于演示功能
"""

import os
import time
import json
import random
from datetime import datetime
from typing import Dict, List, Optional

class SimpleVoiceRecognition:
    """简化版语音识别类"""
    
    def __init__(self, app_id: str = "demo_app", api_key: str = "demo_key", api_secret: str = "demo_secret"):
        """初始化语音识别器"""
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        
        # 消防相关语音命令词典
        self.firefighting_commands = {
            "火警": ["火警", "火灾", "起火", "燃烧", "烟雾", "报警"],
            "救援": ["救援", "救人", "被困", "疏散", "逃生", "安全"],
            "设备": ["设备", "器材", "水枪", "泡沫", "梯子", "呼吸器"],
            "位置": ["位置", "地点", "楼层", "房间", "方向", "坐标"],
            "状态": ["状态", "情况", "程度", "严重", "紧急", "危险"],
            "行动": ["行动", "执行", "开始", "停止", "继续", "完成"]
        }
        
        print("✅ 简化版语音识别器初始化成功")
        print("📝 注意：这是演示版本，使用模拟数据")
    
    def create_mock_audio_file(self, duration: int = 5) -> str:
        """创建模拟音频文件"""
        audio_file = f"mock_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        
        # 创建模拟音频文件（实际应用中这里会是真实的音频录制）
        with open(audio_file, 'w') as f:
            f.write(f"# 模拟音频文件，时长: {duration}秒\n")
            f.write(f"# 创建时间: {datetime.now().isoformat()}\n")
            f.write("# 实际应用中这里会是WAV格式的音频数据\n")
        
        print(f"🎤 模拟录制音频文件: {audio_file}")
        return audio_file
    
    def mock_online_recognition(self, audio_file: str) -> Dict:
        """模拟在线语音识别"""
        print("🔄 进行在线语音识别...")
        time.sleep(random.uniform(1, 3))  # 模拟网络延迟
        
        # 模拟识别结果
        mock_results = [
            "发现火情，三楼东侧房间起火，烟雾很大，需要立即救援",
            "有人被困在二楼，需要紧急救援",
            "启动水枪，准备灭火",
            "火势已经控制，现场安全",
            "需要更多呼吸器和梯子",
            "疏散人员到安全区域",
            "这里是消防指挥中心，请报告现场情况",
            "收到，正在调度消防力量前往现场",
            "现场温度很高，注意安全",
            "救援人员已到达，开始救援行动"
        ]
        
        result_text = random.choice(mock_results)
        confidence = random.uniform(0.85, 0.98)
        
        # 分析消防相关内容
        firefighting_analysis = self.analyze_firefighting_content(result_text)
        
        result = {
            "type": "online",
            "text": result_text,
            "confidence": confidence,
            "firefighting_analysis": firefighting_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"✅ 在线识别完成: {result_text}")
        return result
    
    def mock_offline_recognition(self, audio_file: str) -> Dict:
        """模拟离线语音识别"""
        print("🔄 进行离线语音识别...")
        time.sleep(random.uniform(0.5, 1.5))  # 模拟本地处理延迟
        
        # 模拟识别结果（离线识别通常准确率稍低）
        mock_results = [
            "发现火情，需要救援",
            "有人被困，紧急救援",
            "启动设备，准备行动",
            "火势控制，现场安全",
            "需要器材，请求支援",
            "疏散人员，注意安全",
            "消防指挥，报告情况",
            "收到指令，开始行动",
            "现场高温，注意防护",
            "救援到达，开始行动"
        ]
        
        result_text = random.choice(mock_results)
        confidence = random.uniform(0.75, 0.90)
        
        # 分析消防相关内容
        firefighting_analysis = self.analyze_firefighting_content(result_text)
        
        result = {
            "type": "offline",
            "text": result_text,
            "confidence": confidence,
            "firefighting_analysis": firefighting_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"✅ 离线识别完成: {result_text}")
        return result
    
    def analyze_firefighting_content(self, text: str) -> Dict:
        """分析消防相关内容"""
        analysis = {
            "categories": [],
            "keywords": [],
            "urgency_level": "normal",
            "action_required": False
        }
        
        text_lower = text.lower()
        
        # 检查消防类别
        for category, keywords in self.firefighting_commands.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if category not in analysis["categories"]:
                        analysis["categories"].append(category)
                    if keyword not in analysis["keywords"]:
                        analysis["keywords"].append(keyword)
        
        # 判断紧急程度
        urgent_keywords = ["火警", "火灾", "起火", "燃烧", "被困", "危险", "紧急"]
        if any(keyword in text_lower for keyword in urgent_keywords):
            analysis["urgency_level"] = "urgent"
            analysis["action_required"] = True
        
        return analysis
    
    def demo_recording_and_recognition(self):
        """演示录制和识别功能"""
        print("\n🎤 录制和识别演示")
        print("=" * 40)
        
        # 模拟录制音频
        audio_file = self.create_mock_audio_file(duration=5)
        
        # 在线识别
        print("\n1. 在线识别:")
        online_result = self.mock_online_recognition(audio_file)
        self.display_result(online_result)
        
        # 离线识别
        print("\n2. 离线识别:")
        offline_result = self.mock_offline_recognition(audio_file)
        self.display_result(offline_result)
        
        # 清理文件
        try:
            os.remove(audio_file)
        except:
            pass
    
    def demo_firefighting_commands(self):
        """演示消防命令识别"""
        print("\n🔥 消防命令识别演示")
        print("=" * 40)
        
        # 示例消防命令
        firefighting_commands = [
            "发现火情，三楼东侧房间起火，烟雾很大，需要立即救援",
            "有人被困在二楼，需要紧急救援",
            "启动水枪，准备灭火",
            "火势已经控制，现场安全",
            "需要更多呼吸器和梯子",
            "疏散人员到安全区域"
        ]
        
        for i, command in enumerate(firefighting_commands, 1):
            print(f"\n{i}. 命令: {command}")
            
            # 模拟识别
            result = self.mock_online_recognition("mock_audio.wav")
            result["text"] = command  # 使用预设命令
            
            # 分析消防内容
            analysis = self.analyze_firefighting_content(command)
            result["firefighting_analysis"] = analysis
            
            print(f"   命令类型: {', '.join(analysis['categories']) if analysis['categories'] else '无'}")
            print(f"   关键词: {', '.join(analysis['keywords']) if analysis['keywords'] else '无'}")
            print(f"   紧急程度: {analysis['urgency_level']}")
            print(f"   需要行动: {'是' if analysis['action_required'] else '否'}")
    
    def demo_real_time_simulation(self):
        """演示实时识别模拟"""
        print("\n🔄 实时识别模拟演示")
        print("=" * 40)
        
        print("开始模拟实时识别（10秒）...")
        print("模拟语音输入...")
        
        start_time = time.time()
        while time.time() - start_time < 10:
            # 模拟实时识别
            result = self.mock_online_recognition("mock_audio.wav")
            
            print(f"实时识别: {result['text']}")
            
            if result.get('firefighting_analysis', {}).get('action_required'):
                print("🚨 检测到紧急情况，需要立即行动！")
            
            time.sleep(2)  # 模拟识别间隔
        
        print("实时识别模拟结束")
    
    def display_result(self, result: Dict):
        """显示识别结果"""
        print(f"识别文本: {result['text']}")
        print(f"置信度: {result['confidence']:.2f}")
        print(f"识别时间: {result['timestamp']}")
        
        if result.get('firefighting_analysis'):
            analysis = result['firefighting_analysis']
            print(f"消防分析:")
            print(f"  类别: {', '.join(analysis['categories']) if analysis['categories'] else '无'}")
            print(f"  关键词: {', '.join(analysis['keywords']) if analysis['keywords'] else '无'}")
            print(f"  紧急程度: {analysis['urgency_level']}")
            print(f"  需要行动: {'是' if analysis['action_required'] else '否'}")

def main():
    """主函数"""
    print("🚀 简化版语音识别演示程序")
    print("=" * 50)
    print("📝 这是演示版本，使用模拟数据")
    print("💡 实际应用中需要安装科大讯飞SDK")
    print("=" * 50)
    
    # 初始化识别器
    recognizer = SimpleVoiceRecognition()
    
    while True:
        print("\n请选择演示功能:")
        print("1. 录制和识别演示")
        print("2. 消防命令识别演示")
        print("3. 实时识别模拟")
        print("4. 批量识别演示")
        print("0. 退出")
        
        choice = input("请输入选择 (0-4): ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            recognizer.demo_recording_and_recognition()
        elif choice == "2":
            recognizer.demo_firefighting_commands()
        elif choice == "3":
            recognizer.demo_real_time_simulation()
        elif choice == "4":
            print("\n📁 批量识别演示")
            print("=" * 40)
            for i in range(3):
                print(f"\n处理第{i+1}个音频文件:")
                result = recognizer.mock_online_recognition(f"mock_audio_{i}.wav")
                recognizer.display_result(result)
        else:
            print("❌ 无效选择")
    
    print("\n👋 演示结束，谢谢使用！")
    print("\n💡 要使用真实功能，请:")
    print("1. 安装科大讯飞SDK: pip install iflytek-voice-sdk")
    print("2. 配置真实的app_id、api_key、api_secret")
    print("3. 使用真实的音频录制功能")

if __name__ == "__main__":
    main()




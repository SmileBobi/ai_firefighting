"""
科大讯飞语音识别程序
支持在线和离线语音识别，结合消防项目应用场景
"""

import os
import json
import time
import threading
import wave
import pyaudio
from datetime import datetime
from typing import Dict, List, Optional, Callable
import logging

# 科大讯飞SDK相关导入
try:
    from iflytek_voice_sdk import VoiceRecognition
    from iflytek_voice_sdk import OnlineRecognition
    from iflytek_voice_sdk import OfflineRecognition
except ImportError:
    print("警告：未安装科大讯飞SDK，请先安装 iflytek-voice-sdk")
    print("安装命令：pip install iflytek-voice-sdk")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FirefightingVoiceRecognition:
    """消防项目语音识别类"""
    
    def __init__(self, app_id: str, api_key: str, api_secret: str):
        """
        初始化语音识别
        
        Args:
            app_id: 科大讯飞应用ID
            api_key: API密钥
            api_secret: API密钥
        """
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
        
        # 初始化识别器
        self.online_recognizer = None
        self.offline_recognizer = None
        self.is_recording = False
        self.recording_thread = None
        
    def initialize_recognizers(self):
        """初始化在线和离线识别器"""
        try:
            # 初始化在线识别器
            self.online_recognizer = OnlineRecognition(
                app_id=self.app_id,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            logger.info("在线语音识别器初始化成功")
            
            # 初始化离线识别器
            self.offline_recognizer = OfflineRecognition(
                app_id=self.app_id,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            logger.info("离线语音识别器初始化成功")
            
            return True
            
        except Exception as e:
            logger.error(f"识别器初始化失败: {e}")
            return False
    
    def record_audio(self, duration: int = 5, sample_rate: int = 16000) -> str:
        """
        录制音频
        
        Args:
            duration: 录制时长（秒）
            sample_rate: 采样率
            
        Returns:
            音频文件路径
        """
        audio_file = f"firefighting_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        
        # 音频参数
        chunk = 1024
        format = pyaudio.paInt16
        channels = 1
        
        # 初始化PyAudio
        p = pyaudio.PyAudio()
        
        # 打开音频流
        stream = p.open(
            format=format,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk
        )
        
        logger.info(f"开始录制音频，时长: {duration}秒")
        frames = []
        
        for i in range(0, int(sample_rate / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)
        
        # 停止录制
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # 保存音频文件
        with wave.open(audio_file, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(format))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))
        
        logger.info(f"音频录制完成，保存到: {audio_file}")
        return audio_file
    
    def online_recognition(self, audio_file: str) -> Dict:
        """
        在线语音识别
        
        Args:
            audio_file: 音频文件路径
            
        Returns:
            识别结果字典
        """
        if not self.online_recognizer:
            return {"error": "在线识别器未初始化"}
        
        try:
            logger.info("开始在线语音识别...")
            result = self.online_recognizer.recognize(audio_file)
            
            # 分析消防相关关键词
            firefighting_analysis = self.analyze_firefighting_content(result.get('text', ''))
            
            return {
                "type": "online",
                "text": result.get('text', ''),
                "confidence": result.get('confidence', 0),
                "firefighting_analysis": firefighting_analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"在线识别失败: {e}")
            return {"error": str(e)}
    
    def offline_recognition(self, audio_file: str) -> Dict:
        """
        离线语音识别
        
        Args:
            audio_file: 音频文件路径
            
        Returns:
            识别结果字典
        """
        if not self.offline_recognizer:
            return {"error": "离线识别器未初始化"}
        
        try:
            logger.info("开始离线语音识别...")
            result = self.offline_recognizer.recognize(audio_file)
            
            # 分析消防相关关键词
            firefighting_analysis = self.analyze_firefighting_content(result.get('text', ''))
            
            return {
                "type": "offline",
                "text": result.get('text', ''),
                "confidence": result.get('confidence', 0),
                "firefighting_analysis": firefighting_analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"离线识别失败: {e}")
            return {"error": str(e)}
    
    def analyze_firefighting_content(self, text: str) -> Dict:
        """
        分析消防相关内容
        
        Args:
            text: 识别的文本
            
        Returns:
            分析结果
        """
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
    
    def real_time_recognition(self, callback: Callable[[Dict], None], duration: int = 10):
        """
        实时语音识别
        
        Args:
            callback: 识别结果回调函数
            duration: 识别时长
        """
        def recognition_worker():
            self.is_recording = True
            start_time = time.time()
            
            while self.is_recording and (time.time() - start_time) < duration:
                # 录制短音频片段
                audio_file = self.record_audio(duration=2)
                
                # 在线识别
                online_result = self.online_recognition(audio_file)
                if online_result.get('text'):
                    callback(online_result)
                
                # 清理临时文件
                try:
                    os.remove(audio_file)
                except:
                    pass
                
                time.sleep(0.5)  # 短暂休息
        
        self.recording_thread = threading.Thread(target=recognition_worker)
        self.recording_thread.start()
    
    def stop_real_time_recognition(self):
        """停止实时识别"""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
    
    def batch_recognition(self, audio_files: List[str], use_offline: bool = False) -> List[Dict]:
        """
        批量语音识别
        
        Args:
            audio_files: 音频文件列表
            use_offline: 是否使用离线识别
            
        Returns:
            识别结果列表
        """
        results = []
        
        for audio_file in audio_files:
            if use_offline:
                result = self.offline_recognition(audio_file)
            else:
                result = self.online_recognition(audio_file)
            
            results.append({
                "file": audio_file,
                "result": result
            })
        
        return results

def demo_firefighting_voice_recognition():
    """消防语音识别演示"""
    print("🔥 消防项目语音识别演示")
    print("=" * 50)
    
    # 配置信息（需要替换为实际的科大讯飞配置）
    config = {
        "app_id": "your_app_id_here",
        "api_key": "your_api_key_here", 
        "api_secret": "your_api_secret_here"
    }
    
    # 初始化语音识别器
    recognizer = FirefightingVoiceRecognition(
        app_id=config["app_id"],
        api_key=config["api_key"],
        api_secret=config["api_secret"]
    )
    
    # 初始化识别器
    if not recognizer.initialize_recognizers():
        print("❌ 识别器初始化失败，请检查配置")
        return
    
    print("✅ 语音识别器初始化成功")
    
    # 演示菜单
    while True:
        print("\n请选择操作:")
        print("1. 录制并识别语音")
        print("2. 在线语音识别")
        print("3. 离线语音识别")
        print("4. 实时语音识别")
        print("5. 批量识别")
        print("0. 退出")
        
        choice = input("请输入选择 (0-5): ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            # 录制并识别
            print("\n开始录制语音（5秒）...")
            audio_file = recognizer.record_audio(duration=5)
            
            print("进行在线识别...")
            online_result = recognizer.online_recognition(audio_file)
            print(f"在线识别结果: {json.dumps(online_result, ensure_ascii=False, indent=2)}")
            
            print("进行离线识别...")
            offline_result = recognizer.offline_recognition(audio_file)
            print(f"离线识别结果: {json.dumps(offline_result, ensure_ascii=False, indent=2)}")
            
            # 清理文件
            try:
                os.remove(audio_file)
            except:
                pass
                
        elif choice == "2":
            # 在线识别
            audio_file = input("请输入音频文件路径: ").strip()
            if os.path.exists(audio_file):
                result = recognizer.online_recognition(audio_file)
                print(f"在线识别结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
            else:
                print("❌ 文件不存在")
                
        elif choice == "3":
            # 离线识别
            audio_file = input("请输入音频文件路径: ").strip()
            if os.path.exists(audio_file):
                result = recognizer.offline_recognition(audio_file)
                print(f"离线识别结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
            else:
                print("❌ 文件不存在")
                
        elif choice == "4":
            # 实时识别
            print("\n开始实时语音识别（10秒）...")
            print("请说话...")
            
            def result_callback(result):
                print(f"识别结果: {result['text']}")
                if result.get('firefighting_analysis', {}).get('action_required'):
                    print("🚨 检测到紧急情况，需要立即行动！")
            
            recognizer.real_time_recognition(result_callback, duration=10)
            recognizer.stop_real_time_recognition()
            
        elif choice == "5":
            # 批量识别
            audio_files = input("请输入音频文件路径（用逗号分隔）: ").strip().split(',')
            audio_files = [f.strip() for f in audio_files if f.strip()]
            
            use_offline = input("是否使用离线识别？(y/n): ").strip().lower() == 'y'
            
            results = recognizer.batch_recognition(audio_files, use_offline)
            print(f"批量识别结果: {json.dumps(results, ensure_ascii=False, indent=2)}")
        
        else:
            print("❌ 无效选择")

if __name__ == "__main__":
    demo_firefighting_voice_recognition()

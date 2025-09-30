"""
消防项目专用语音命令识别功能
结合消防场景的语音命令处理和响应
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

class FirefightingCommandType(Enum):
    """消防命令类型枚举"""
    EMERGENCY = "emergency"      # 紧急情况
    RESCUE = "rescue"           # 救援行动
    EQUIPMENT = "equipment"     # 设备操作
    LOCATION = "location"       # 位置信息
    STATUS = "status"           # 状态报告
    ACTION = "action"           # 行动指令

class UrgencyLevel(Enum):
    """紧急程度枚举"""
    LOW = "low"         # 低
    NORMAL = "normal"   # 正常
    HIGH = "high"       # 高
    CRITICAL = "critical"  # 紧急

class FirefightingVoiceCommands:
    """消防语音命令处理器"""
    
    def __init__(self):
        """初始化消防语音命令处理器"""
        self.command_patterns = self._initialize_command_patterns()
        self.equipment_commands = self._initialize_equipment_commands()
        self.location_keywords = self._initialize_location_keywords()
        self.emergency_keywords = self._initialize_emergency_keywords()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_command_patterns(self) -> Dict[str, List[str]]:
        """初始化命令模式"""
        return {
            "火警报告": [
                "发现火情", "起火", "燃烧", "烟雾", "火警", "火灾", "失火",
                "有火", "烧起来了", "冒烟", "火势", "燃烧物"
            ],
            "救援请求": [
                "救人", "被困", "疏散", "逃生", "救援", "帮助", "救命",
                "有人被困", "需要救援", "紧急救援", "人员疏散"
            ],
            "设备操作": [
                "启动", "关闭", "开启", "停止", "操作", "使用", "准备",
                "水枪", "泡沫", "梯子", "呼吸器", "设备", "器材"
            ],
            "位置报告": [
                "位置", "地点", "楼层", "房间", "方向", "坐标", "在",
                "位于", "坐标", "方位", "具体位置", "详细地址"
            ],
            "状态报告": [
                "状态", "情况", "程度", "严重", "紧急", "危险", "安全",
                "火势", "烟雾", "温度", "能见度", "现场情况"
            ],
            "行动指令": [
                "行动", "执行", "开始", "停止", "继续", "完成", "准备",
                "立即", "马上", "现在", "开始行动", "执行任务"
            ]
        }
    
    def _initialize_equipment_commands(self) -> Dict[str, List[str]]:
        """初始化设备命令"""
        return {
            "水枪": ["水枪", "水炮", "水龙", "喷水", "射水"],
            "泡沫": ["泡沫", "泡沫枪", "泡沫炮", "泡沫液"],
            "梯子": ["梯子", "云梯", "伸缩梯", "爬梯"],
            "呼吸器": ["呼吸器", "面罩", "氧气", "空气呼吸器"],
            "水带": ["水带", "水龙带", "软管", "水管"],
            "泵浦": ["泵浦", "水泵", "增压泵", "消防泵"],
            "照明": ["照明", "探照灯", "手电", "应急灯"],
            "通讯": ["通讯", "对讲机", "电台", "通信设备"]
        }
    
    def _initialize_location_keywords(self) -> List[str]:
        """初始化位置关键词"""
        return [
            "一楼", "二楼", "三楼", "四楼", "五楼", "六楼", "七楼", "八楼", "九楼", "十楼",
            "地下室", "地下一层", "地下二层", "地下三层",
            "大厅", "走廊", "楼梯间", "电梯间", "机房", "配电室",
            "东", "西", "南", "北", "东南", "西南", "东北", "西北",
            "前门", "后门", "侧门", "入口", "出口", "安全出口"
        ]
    
    def _initialize_emergency_keywords(self) -> List[str]:
        """初始化紧急关键词"""
        return [
            "紧急", "危险", "严重", "爆炸", "泄漏", "中毒", "窒息",
            "被困", "受伤", "死亡", "失踪", "失控", "蔓延", "扩大"
        ]
    
    def analyze_voice_command(self, text: str) -> Dict:
        """
        分析语音命令
        
        Args:
            text: 识别的文本
            
        Returns:
            分析结果字典
        """
        analysis = {
            "command_type": None,
            "urgency_level": UrgencyLevel.NORMAL.value,
            "equipment_mentioned": [],
            "location_mentioned": [],
            "emergency_indicators": [],
            "action_required": False,
            "confidence": 0.0,
            "parsed_commands": [],
            "timestamp": datetime.now().isoformat()
        }
        
        text_lower = text.lower()
        confidence_score = 0
        
        # 分析命令类型
        for command_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    analysis["command_type"] = command_type
                    confidence_score += 1
                    analysis["parsed_commands"].append({
                        "type": command_type,
                        "pattern": pattern,
                        "confidence": 0.8
                    })
        
        # 分析设备提及
        for equipment, keywords in self.equipment_commands.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if equipment not in analysis["equipment_mentioned"]:
                        analysis["equipment_mentioned"].append(equipment)
                    confidence_score += 0.5
        
        # 分析位置提及
        for location in self.location_keywords:
            if location in text_lower:
                if location not in analysis["location_mentioned"]:
                    analysis["location_mentioned"].append(location)
                confidence_score += 0.3
        
        # 分析紧急指标
        for emergency in self.emergency_keywords:
            if emergency in text_lower:
                if emergency not in analysis["emergency_indicators"]:
                    analysis["emergency_indicators"].append(emergency)
                confidence_score += 1
        
        # 判断紧急程度
        if analysis["emergency_indicators"]:
            if len(analysis["emergency_indicators"]) >= 3:
                analysis["urgency_level"] = UrgencyLevel.CRITICAL.value
            elif len(analysis["emergency_indicators"]) >= 2:
                analysis["urgency_level"] = UrgencyLevel.HIGH.value
            else:
                analysis["urgency_level"] = UrgencyLevel.NORMAL.value
        
        # 判断是否需要行动
        action_keywords = ["立即", "马上", "现在", "开始", "执行", "行动"]
        if any(keyword in text_lower for keyword in action_keywords):
            analysis["action_required"] = True
        
        # 计算置信度
        analysis["confidence"] = min(confidence_score / max(len(text.split()), 1), 1.0)
        
        return analysis
    
    def generate_response(self, analysis: Dict) -> Dict:
        """
        生成响应
        
        Args:
            analysis: 分析结果
            
        Returns:
            响应字典
        """
        response = {
            "acknowledgment": "",
            "instructions": [],
            "priority": "normal",
            "estimated_time": None,
            "resources_needed": [],
            "safety_warnings": []
        }
        
        # 根据命令类型生成响应
        if analysis["command_type"] == "火警报告":
            response["acknowledgment"] = "收到火警报告，正在调度消防力量"
            response["instructions"] = [
                "立即启动火警应急预案",
                "疏散现场人员到安全区域",
                "准备消防器材",
                "联系医疗救护"
            ]
            response["priority"] = "high"
            response["resources_needed"] = ["消防车", "水枪", "呼吸器", "梯子"]
            response["safety_warnings"] = ["注意安全", "佩戴防护装备", "避免烟雾吸入"]
        
        elif analysis["command_type"] == "救援请求":
            response["acknowledgment"] = "收到救援请求，立即组织救援"
            response["instructions"] = [
                "组织救援队伍",
                "准备救援器材",
                "确定被困人员位置",
                "制定救援方案"
            ]
            response["priority"] = "critical"
            response["resources_needed"] = ["救援队", "呼吸器", "绳索", "担架"]
            response["safety_warnings"] = ["救援人员注意安全", "确保自身安全"]
        
        elif analysis["command_type"] == "设备操作":
            response["acknowledgment"] = "收到设备操作指令"
            response["instructions"] = [
                "检查设备状态",
                "按照操作规程执行",
                "确保设备正常运行"
            ]
            response["priority"] = "normal"
            response["resources_needed"] = analysis["equipment_mentioned"]
        
        elif analysis["command_type"] == "位置报告":
            response["acknowledgment"] = "收到位置信息"
            response["instructions"] = [
                "确认具体位置",
                "更新位置信息",
                "通知相关人员"
            ]
            response["priority"] = "normal"
        
        elif analysis["command_type"] == "状态报告":
            response["acknowledgment"] = "收到状态报告"
            response["instructions"] = [
                "记录状态信息",
                "评估当前情况",
                "制定应对措施"
            ]
            response["priority"] = "normal"
        
        elif analysis["command_type"] == "行动指令":
            response["acknowledgment"] = "收到行动指令"
            response["instructions"] = [
                "确认行动方案",
                "分配任务",
                "开始执行"
            ]
            response["priority"] = "high"
        
        # 根据紧急程度调整响应
        if analysis["urgency_level"] == UrgencyLevel.CRITICAL.value:
            response["priority"] = "critical"
            response["acknowledgment"] = "🚨 紧急情况！" + response["acknowledgment"]
            response["safety_warnings"].append"]("紧急情况，立即行动！")
        
        return response
    
    def process_firefighting_command(self, text: str) -> Dict:
        """
        处理消防语音命令
        
        Args:
            text: 识别的文本
            
        Returns:
            处理结果
        """
        # 分析命令
        analysis = self.analyze_voice_command(text)
        
        # 生成响应
        response = self.generate_response(analysis)
        
        # 记录日志
        self.logger.info(f"处理消防命令: {text}")
        self.logger.info(f"分析结果: {analysis}")
        self.logger.info(f"响应: {response}")
        
        return {
            "original_text": text,
            "analysis": analysis,
            "response": response,
            "processing_time": datetime.now().isoformat()
        }
    
    def get_command_suggestions(self, partial_text: str) -> List[str]:
        """
        获取命令建议
        
        Args:
            partial_text: 部分文本
            
        Returns:
            建议列表
        """
        suggestions = []
        partial_lower = partial_text.lower()
        
        # 基于部分文本提供建议
        for command_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                if pattern.startswith(partial_lower) or partial_lower in pattern:
                    suggestions.append(f"{pattern} - {command_type}")
        
        return suggestions[:5]  # 返回前5个建议
    
    def validate_command(self, text: str) -> Tuple[bool, List[str]]:
        """
        验证命令
        
        Args:
            text: 命令文本
            
        Returns:
            (是否有效, 错误列表)
        """
        errors = []
        text_lower = text.lower()
        
        # 检查是否包含消防相关关键词
        has_firefighting_keywords = False
        for patterns in self.command_patterns.values():
            if any(pattern in text_lower for pattern in patterns):
                has_firefighting_keywords = True
                break
        
        if not has_firefighting_keywords:
            errors.append("未包含消防相关关键词")
        
        # 检查命令完整性
        if len(text.strip()) < 3:
            errors.append("命令过短，请提供更详细的信息")
        
        # 检查紧急情况描述
        if any(keyword in text_lower for keyword in ["紧急", "危险", "严重"]):
            if not any(keyword in text_lower for keyword in ["位置", "地点", "在"]):
                errors.append("紧急情况需要提供具体位置信息")
        
        return len(errors) == 0, errors

def demo_firefighting_voice_commands():
    """消防语音命令演示"""
    print("🔥 消防项目语音命令识别演示")
    print("=" * 50)
    
    # 初始化命令处理器
    command_processor = FirefightingVoiceCommands()
    
    # 示例命令
    example_commands = [
        "发现火情，三楼东侧房间起火，烟雾很大，需要立即救援",
        "有人被困在二楼，需要紧急救援",
        "启动水枪，准备灭火",
        "火势已经控制，现场安全",
        "需要更多呼吸器和梯子",
        "疏散人员到安全区域"
    ]
    
    print("示例命令处理:")
    print("-" * 30)
    
    for i, command in enumerate(example_commands, 1):
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
        
        if result['response']['instructions']:
            print("   指令:")
            for instruction in result['response']['instructions']:
                print(f"     - {instruction}")
    
    # 交互式命令处理
    print("\n" + "=" * 50)
    print("交互式命令处理 (输入 'quit' 退出)")
    print("=" * 50)
    
    while True:
        user_input = input("\n请输入消防命令: ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if not user_input:
            continue
        
        # 验证命令
        is_valid, errors = command_processor.validate_command(user_input)
        if not is_valid:
            print("❌ 命令验证失败:")
            for error in errors:
                print(f"   - {error}")
            continue
        
        # 处理命令
        result = command_processor.process_firefighting_command(user_input)
        
        print(f"\n✅ 命令处理结果:")
        print(f"   命令类型: {result['analysis']['command_type']}")
        print(f"   紧急程度: {result['analysis']['urgency_level']}")
        print(f"   置信度: {result['analysis']['confidence']:.2f}")
        print(f"   响应: {result['response']['acknowledgment']}")
        
        if result['response']['instructions']:
            print("   执行指令:")
            for instruction in result['response']['instructions']:
                print(f"     - {instruction}")
        
        if result['response']['safety_warnings']:
            print("   安全警告:")
            for warning in result['response']['safety_warnings']:
                print(f"     - {warning}")

if __name__ == "__main__":
    demo_firefighting_voice_commands()

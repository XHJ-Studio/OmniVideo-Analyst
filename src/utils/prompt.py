"""
Prompt 模板库

针对不同分析场景的优化 Prompt
"""

# 通用分析 Prompt
GENERAL_ANALYSIS_PROMPT = """
请仔细分析这段视频内容，回答以下问题：

{query}

要求：
1. 如果视频中出现了相关内容，请描述具体是什么、在什么时间出现
2. 如果有动作或状态变化，请按时间顺序描述
3. 如果不确定或看不清，请说明
4. 回答要简洁明了，避免冗长

请按以下格式输出：
- 时间范围：[开始时间 - 结束时间]
- 描述：具体内容
- 置信度：高/中/低
"""

# 门/窗状态检测
DOOR_WINDOW_PROMPT = """
请仔细分析这段视频中的门/窗状态：

1. 视频中是否出现了门或窗？如果有，是什么类型（木门/铁门/玻璃门/窗户等）？
2. 在视频开始时，门/窗的状态是什么（开/关）？
3. 视频过程中，门/窗的状态有没有发生变化？
   - 如果有，请说明具体时间点和变化内容（谁打开/关闭的）
   - 如果没有，请说明始终保持什么状态

请详细描述，给出精确的时间点。
"""

# 人员检测
PERSON_DETECTION_PROMPT = """
请分析这段视频中出现的人员：

1. 视频中有几个人出现？
2. 每个人的特征是什么（衣着颜色、性别、大致年龄等）
3. 每个人在视频中做了什么动作？
4. 每个人出现的时间段是什么？

请按时间顺序详细描述。
"""

# 车辆检测
VEHICLE_DETECTION_PROMPT = """
请分析这段视频中出现的车辆：

1. 视频中有几辆车出现？
2. 每辆车的特征是什么（颜色、类型：轿车/摩托车/自行车/卡车等）
3. 每辆车的动作是什么（行驶/停放/进入/离开）
4. 每辆车出现的时间段是什么？

请按时间顺序详细描述。
"""

# 可疑行为检测
SUSPICIOUS_BEHAVIOR_PROMPT = """
请分析这段视频中是否有可疑行为：

可疑行为包括但不限于：
- 翻越围墙/窗户
- 强行破门
- 鬼鬼祟祟的徘徊
- 异常的物品搬运
- 其他不寻常的行为

如果有可疑行为，请详细描述：
1. 什么时间发生
2. 是谁做的（描述特征）
3. 具体做了什么
4. 持续了多久

如果没有可疑行为，也请说明。
"""

# 物体搜索
OBJECT_SEARCH_PROMPT = """
请在视频中寻找以下物体：

目标：{object_name}

请回答：
1. 视频中是否出现了这个物体？
2. 如果出现，在什么时间段？
3. 物体的状态/位置是什么？
4. 有没有与这个物体相关的动作或事件？

如果不确定是否是目标物体，请说明你的判断依据。
"""


def get_prompt(template_name: str, **kwargs) -> str:
    """
    获取 Prompt 模板
    
    Args:
        template_name: 模板名称
        **kwargs: 模板变量
        
    Returns:
        格式化后的 Prompt
    """
    templates = {
        "general": GENERAL_ANALYSIS_PROMPT,
        "door_window": DOOR_WINDOW_PROMPT,
        "person": PERSON_DETECTION_PROMPT,
        "vehicle": VEHICLE_DETECTION_PROMPT,
        "suspicious": SUSPICIOUS_BEHAVIOR_PROMPT,
        "object": OBJECT_SEARCH_PROMPT,
    }
    
    template = templates.get(template_name, GENERAL_ANALYSIS_PROMPT)
    return template.format(**kwargs)

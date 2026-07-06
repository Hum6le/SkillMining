"""
基于memory机制的迭代抽取客服操作（operator）脚本

功能：
1. 从Excel文件中读取"人工会话"列
2. 使用LLM提取每个对话中的客服操作
3. 维护一个操作库（memory），迭代地完善和扩展
4. 支持批量处理和断点续跑
5. 将结果保存为JSON和文本文件
"""

import json
import os
import re
from difflib import SequenceMatcher
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Tuple
from tqdm import tqdm

import sys as _sys
_skill_dir = Path(__file__).resolve().parent
if str(_skill_dir) not in _sys.path:
    _sys.path.insert(0, str(_skill_dir))
from llm_utils import QWEN_WORKFLOW_ID, ds_api_retry, run_flow_retry

BASE_DIR = _skill_dir
DATA_DIR = BASE_DIR / "data"
DATA_BY_INTENT_DIR = DATA_DIR / "data_by_intent"
OPERATOR_OUTPUT_JSON = DATA_DIR / "operator_results.json"
OPERATOR_OUTPUT_TXT = DATA_DIR / "operator_results.txt"
OPERATOR_MEMORY_JSON = DATA_DIR / "operator_memory.json"
USER_BEHAVIOR_MEMORY_JSON = DATA_DIR / "user_behavior_memory.json"
STATE_PATH = DATA_DIR / "operator_extract_state.json"
BATCH_SIZE = 10
ENV_MAX_BATCHES = "MAX_BATCHES"
# 默认使用的LLM API: "deepseek" 或 "qwen"
DEFAULT_LLM = os.getenv("LLM_API", "deepseek")


def load_excel_data(excel_path: Path) -> pd.DataFrame:
    """加载Excel数据"""
    print(f"正在读取Excel文件: {excel_path.name}...")
    df = pd.read_excel(excel_path)
    print(f"  - 读取成功，共 {len(df)} 行")
    
    # 查找"人工会话"列
    dialogue_col = None
    for col in df.columns:
        if "人工会话" in str(col) or "人工对话" in str(col):
            dialogue_col = col
            break
    
    if dialogue_col is None:
        raise ValueError("未找到'人工会话'或'人工对话'列")
    
    print(f"  - 检测到对话列: {dialogue_col}")
    
    # 过滤掉空对话
    df = df[df[dialogue_col].notna() & (df[dialogue_col].astype(str).str.strip() != "")].copy()
    print(f"  - 过滤后有效对话数: {len(df)}")
    
    return df, dialogue_col


def load_operator_memory() -> Dict[str, str]:
    """加载已有的操作库（memory），返回KV对字典"""
    if not OPERATOR_MEMORY_JSON.exists():
        return {}
    
    try:
        with open(OPERATOR_MEMORY_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 支持多种格式：直接是dict，或者在operators字段中
            if isinstance(data, dict):
                if "operators" in data:
                    operators = data["operators"]
                    # 如果是列表，转换为字典（key为操作名，value为空字符串或描述）
                    if isinstance(operators, list):
                        return {op: "" for op in operators}
                    elif isinstance(operators, dict):
                        return operators
                else:
                    # 直接是字典格式
                    return data
            elif isinstance(data, list):
                # 兼容旧格式：列表
                return {op: "" for op in data}
            return {}
    except Exception as e:
        print(f"警告: 加载操作库失败: {e}")
        return {}


def save_operator_memory(operators: Dict[str, str]):
    """保存操作库（memory），KV对格式"""
    OPERATOR_MEMORY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OPERATOR_MEMORY_JSON, "w", encoding="utf-8") as f:
        json.dump({"operators": operators}, f, ensure_ascii=False, indent=2)


def load_user_behavior_memory() -> Dict[str, str]:
    """加载用户行为库（memory），返回KV对字典"""
    if not USER_BEHAVIOR_MEMORY_JSON.exists():
        return {}
    
    try:
        with open(USER_BEHAVIOR_MEMORY_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                if "behaviors" in data:
                    behaviors = data["behaviors"]
                    if isinstance(behaviors, list):
                        return {beh: "" for beh in behaviors}
                    elif isinstance(behaviors, dict):
                        return behaviors
                else:
                    return data
            elif isinstance(data, list):
                return {beh: "" for beh in data}
            return {}
    except Exception as e:
        print(f"警告: 加载用户行为库失败: {e}")
        return {}


def save_user_behavior_memory(behaviors: Dict[str, str]):
    """保存用户行为库（memory），KV对格式"""
    USER_BEHAVIOR_MEMORY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(USER_BEHAVIOR_MEMORY_JSON, "w", encoding="utf-8") as f:
        json.dump({"behaviors": behaviors}, f, ensure_ascii=False, indent=2)


def load_state() -> int:
    """加载处理状态（断点续跑）"""
    if not STATE_PATH.exists():
        return 0
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return int(data.get("next_index", 0))
    except Exception:
        return 0


def save_state(next_index: int):
    """保存处理状态"""
    STATE_PATH.write_text(
        json.dumps({"next_index": next_index}, ensure_ascii=False), 
        encoding="utf-8"
    )


def build_extract_prompt(
    dialogue: str, 
    operator_lib: Dict[str, str], 
    user_behavior_lib: Dict[str, str],
    allow_new: bool = True
) -> str:
    """构建提取操作的提示词，包含operator库和用户行为库"""
    operator_hint = ""
    if operator_lib:
        # 显示operator库（KV对）
        lib_items = list(operator_lib.items())[:50]
        lib_text = "\n".join(
            f"- {key}" + (f": {value}" if value else "") 
            for key, value in lib_items
        )
        operator_hint = (
            f"\n<客服操作库>\n"
            + lib_text
            + (f"\n- ...（共{len(operator_lib)}个操作）" if len(operator_lib) > 50 else "")
            + "\n</客服操作库>"
        )
    
    user_behavior_hint = ""
    if user_behavior_lib:
        # 显示用户行为库（KV对）
        beh_items = list(user_behavior_lib.items())[:50]
        beh_text = "\n".join(
            f"- {key}" + (f": {value}" if value else "") 
            for key, value in beh_items
        )
        user_behavior_hint = (
            f"\n<用户行为库>\n"
            + beh_text
            + (f"\n- ...（共{len(user_behavior_lib)}个行为）" if len(user_behavior_lib) > 50 else "")
            + "\n</用户行为库>"
        )
    
    # 根据阶段设置不同的提示
    new_operator_instruction = ""
    if allow_new:
        new_operator_instruction = """4. 优先使用已有操作库中的操作（如果库中存在符合具体行为或与其**语义相近**的operator，直接使用该operator的key），**尽可能检索已有操作，避免添加重复操作。**，当然，操作库最开始可能是空的，你需要将抽象出来的操作全部放到new标签中。
5. 如果库中不存在相近语义的操作，先查看能否和现有操作合并，如果可以则进行合并且修改操作名，返回到<modified>标签中；如果不能则进行抽象并新增操作，但是**不要新增/抽象不包含具体行为的操作，例如问候，礼貌用语，安抚等**，**不要新增/抽象不包含具体行为的操作，例如问候，礼貌用语，安抚等**，**不要新增/抽象不包含具体行为的操作，例如问候，礼貌用语，安抚等**，可以忽略掉这些
6. 如果在处理过程中认为已存在的操作符的语义构成不够完整需要修改，也一并输出出来"""
    else:
        new_operator_instruction = """4. **只能使用已有操作库中的操作**，必须逐字复制库中已有的 operator key，不得修改措辞、不得缩写、不得扩写。如果库中存在符合具体行为或与其**语义相近**的operator，直接使用该operator的key。
5. **严禁新增操作、严禁自创操作名**。如果对话中的操作在库中找不到相近的，选择库中最接近的操作，或者直接忽略该操作。输出的每个操作名必须能在库中精确匹配到。
6. 如果在处理过程中认为已存在的操作符的语义构成不够完整需要修改，也一并输出出来（但修改后的操作名必须在库中存在）"""
    
    new_behavior_instruction = ""
    if allow_new:
        new_behavior_instruction = """4. 优先使用已有用户行为库中的行为（如果库中存在符合具体行为或与其**语义相近**的用户行为，直接使用该行为的key），先查看能否和现有行为合并，如果可以则进行合并且修改行为名；如果不能则进行抽象并新增行为，但是**不要新增不包含具体行为的操作，例如问候，礼貌用语，抱怨等**，**不要新增不包含具体行为的操作，例如问候，礼貌用语，抱怨等**，**不要新增不包含具体行为的操作，例如问候，礼貌用语，抱怨等**
5. 如果库中不存在，则进行抽象并新增行为，但是不要新增不包含具体行为的操作，例如问候，礼貌用语，抱怨等；
6. 如果在处理过程中认为行为的语义构成不够完整需要修改，也一并输出出来"""
    else:
        new_behavior_instruction = """4. **只能使用已有用户行为库中的行为**，必须逐字复制库中已有的 behavior key，不得修改措辞、不得缩写、不得扩写。如果库中存在符合具体行为或与其**语义相近**的用户行为，直接使用该行为的key。
5. **严禁新增行为、严禁自创行为名**。如果对话中的行为在库中找不到相近的，选择库中最接近的行为，或者直接忽略该行为。输出的每个行为名必须能在库中精确匹配到。
6. 如果在处理过程中认为行为的语义构成不够完整需要修改，也一并输出出来（但修改后的行为名必须在库中存在）"""
    
    # 构建输出格式模板（避免在f-string中使用反斜杠）
    new_section_op = "<new>\n[新增客服操作1]\n[新增客服操作2]\n</new>\n" if allow_new else ""
    new_section_beh = "<new>\n[新增用户行为1]\n[新增用户行为2]\n</new>\n" if allow_new else ""
    ordered_seq_section = "<ordered_sequence>\n[角色]:[操作/行为名称]\n[角色]:[操作/行为名称]\n[角色]:[操作/行为名称]\n</ordered_sequence>\n" if not allow_new else ""
    
    # 构建示例部分
    stage_name = "第一阶段" if allow_new else "第二阶段"
    new_example_op = "<new>\n告知物流异常原因,建议重新下单\n</new>\n" if allow_new else ""
    new_example_beh = "<new>\n反馈订单未收到问题\n</new>\n" if allow_new else ""
    ordered_seq_example = ""
    if not allow_new:
        ordered_seq_example = "<ordered_sequence>\n用户:反馈订单未收到问题\n客服:索要订单号\n客服:执行订单状态查询"
        if allow_new:
            ordered_seq_example += "\n客服:告知物流异常原因,建议重新下单"
        ordered_seq_example += "\n</ordered_sequence>\n"
    
    new_hint = "（此阶段需要输出）" if allow_new else "（此阶段不需要输出）"
    ordered_hint = "（此阶段需要输出）" if not allow_new else "（此阶段不需要输出）"
    ordered_required = "（但ordered_sequence必须要有）" if not allow_new else "（但ordered_sequence不需要要有）"
    note_text = "此阶段不需要输出ordered_sequence，只需要输出new和modified的操作。" if allow_new else "ordered_sequence必须严格按照对话的时间顺序排列，先出现的操作/行为在前，后出现的在后。此阶段不需要输出new标签。"
    
    prompt = f"""请从以下客服对话中同时识别客服人员的操作和用户的查询/行为/意图，要求：

【客服操作提取要求】：
0. 给出一段人工客服和用户的对话，请你根据对话内容抽象出客服的操作，在这一段对话中可能会包含多个操作
1. 以动词开头描述操作，操作粒度可以更粗粒度
2. 每个操作对应对话中的一个具体行为，不一定一行是一个操作，也有可能多行是一个操作，请你根据对话自行判断
3. 排除用户行为和寒暄内容
{new_operator_instruction}
   

【用户行为提取要求】：
0. 同时识别用户的查询/行为/意图，在这一段对话中可能会包含多个用户行为
1. 以动词或名词开头描述用户行为，操作粒度可以适当粗粒度
2. 每个行为对应对话中用户的一个具体行为或意图，不一定一行是一个行为，也有可能多行是一个行为
3. 排除客服行为和寒暄内容
{new_behavior_instruction}


输出格式（必须严格按照以下格式）：
```
<operator>
<used>
[使用的已有客服操作1]
[使用的已有客服操作2]
</used>{new_section_op}<modified>
[原操作名] -> [修改后的操作名]
</modified>
</operator>

<user_behavior>
<used>
[使用的已有用户行为1]
[使用的已有用户行为2]
</used>{new_section_beh}<modified>
[原行为名] -> [修改后的行为名]
</modified>
</user_behavior>{ordered_seq_section}
```

说明：
- <operator>标签内：客服操作相关
- <user_behavior>标签内：用户行为相关
- <used>标签内：列出从库中直接使用的（使用库中的key）
- <new>标签内：列出需要新增到库的{new_hint}
- <modified>标签内：列出需要修改的，格式为"原名 -> 修改后的名"
- <ordered_sequence>标签内：按照对话的原始时间顺序，列出所有操作和行为，格式为"角色:操作/行为名称"，角色必须是"用户"或"客服"{ordered_hint}
- 如果某个标签内没有内容，可以省略该标签{ordered_required}

**示例对话：**

【已有客服操作库（请优先使用库中已有的操作，如果库中存在符合具体行为的operator，直接使用该operator的key）】

[索要订单号]
[执行订单状态查询]

用户：我的订单一直没收到
客服：您好，麻烦提供下订单号好吗？我帮您查询
客服：查询到订单因地址不详被退回，建议您核对后重新下单

**假设客服操作库中有"索要订单号"和"执行订单状态查询"**
**假设用户行为库中有"查询订单状态"**

**应提取的内容（{stage_name}）：**
```
<operator>
<used>
索要订单号
执行订单状态查询
</used>{new_example_op}</operator>

<user_behavior>
<used>
</used>{new_example_beh}</user_behavior>{ordered_seq_example}
```

注意：{note_text}

{operator_hint}
{user_behavior_hint}

请按上述规范输出，保持每个操作和行为可执行且无歧义。

**当前待分析对话：**
{dialogue}
"""
    return prompt


def parse_extract_output(text: str) -> Tuple[
    List[str], List[str], List[Tuple[str, str]],  # 客服操作：used, new, modified
    List[str], List[str], List[Tuple[str, str]],  # 用户行为：used, new, modified
    List[Tuple[str, str]]                          # ordered_sequence: [(角色, 操作/行为)]
]:
    """
    解析LLM输出的操作列表和用户行为列表
    返回: (客服used, 客服new, 客服modified, 用户used, 用户new, 用户modified, ordered_sequence)
    """
    def parse_section(section_text: str) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
        """解析一个section（客服操作或用户行为）"""
        used_items = []
        new_items = []
        modified_items = []
        
        # 提取<used>标签内的内容
        used_pattern = r'<used>(.*?)</used>'
        used_match = re.search(used_pattern, section_text, re.DOTALL)
        if used_match:
            used_content = used_match.group(1).strip()
            for line in used_content.split('\n'):
                line = line.strip()
                if line and not line.startswith('<'):
                    line = re.sub(r'^[-\d\.、\s]*', '', line).strip()
                    if line:
                        used_items.append(line)
        
        # 提取<new>标签内的内容
        new_pattern = r'<new>(.*?)</new>'
        new_match = re.search(new_pattern, section_text, re.DOTALL)
        if new_match:
            new_content = new_match.group(1).strip()
            for line in new_content.split('\n'):
                line = line.strip()
                if line and not line.startswith('<'):
                    line = re.sub(r'^[-\d\.、\s]*', '', line).strip()
                    if line:
                        new_items.append(line)
        
        # 提取<modified>标签内的内容
        modified_pattern = r'<modified>(.*?)</modified>'
        modified_match = re.search(modified_pattern, section_text, re.DOTALL)
        if modified_match:
            modified_content = modified_match.group(1).strip()
            for line in modified_content.split('\n'):
                line = line.strip()
                if line and not line.startswith('<'):
                    line = re.sub(r'^[-\d\.、\s]*', '', line).strip()
                    # 解析 "原操作 -> 新操作" 格式
                    if '->' in line:
                        parts = line.split('->', 1)
                        old_item = parts[0].strip()
                        new_item = parts[1].strip() if len(parts) > 1 else ""
                        if old_item and new_item:
                            modified_items.append((old_item, new_item))
        
        return used_items, new_items, modified_items
    
    # 提取客服操作部分
    operator_section = ""
    operator_pattern = r'<operator>(.*?)</operator>'
    operator_match = re.search(operator_pattern, text, re.DOTALL)
    if operator_match:
        operator_section = operator_match.group(1)
    
    # 提取用户行为部分
    user_behavior_section = ""
    user_behavior_pattern = r'<user_behavior>(.*?)</user_behavior>'
    user_behavior_match = re.search(user_behavior_pattern, text, re.DOTALL)
    if user_behavior_match:
        user_behavior_section = user_behavior_match.group(1)
    
    # 解析两个部分
    op_used, op_new, op_modified = parse_section(operator_section) if operator_section else ([], [], [])
    ub_used, ub_new, ub_modified = parse_section(user_behavior_section) if user_behavior_section else ([], [], [])
    
    # 提取ordered_sequence部分
    ordered_sequence = []
    sequence_pattern = r'<ordered_sequence>(.*?)</ordered_sequence>'
    sequence_match = re.search(sequence_pattern, text, re.DOTALL)
    if sequence_match:
        sequence_content = sequence_match.group(1).strip()
        for line in sequence_content.split('\n'):
            line = line.strip()
            if line and not line.startswith('<'):
                # 解析 "角色:操作/行为名称" 格式
                if ':' in line:
                    parts = line.split(':', 1)
                    role = parts[0].strip()
                    operation = parts[1].strip() if len(parts) > 1 else ""
                    if role and operation:
                        # 规范化角色名称
                        if role in ["用户", "user", "User"]:
                            role = "用户"
                        elif role in ["客服", "客服人员", "assistant", "Assistant", "客服人员"]:
                            role = "客服"
                        ordered_sequence.append((role, operation))
    
    # 兼容旧格式（如果没有找到新格式标签）
    if not op_used and not op_new and not op_modified and not ub_used and not ub_new and not ub_modified:
        # 尝试解析旧格式（只有operator）
        pattern = r'<operator>(.*?)</operator>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            for line in lines:
                if line and not line.startswith('<'):
                    line = re.sub(r'^[-\d\.、\s]*', '', line).strip()
                    if line:
                        op_new.append(line)
    
    return op_used, op_new, op_modified, ub_used, ub_new, ub_modified, ordered_sequence


def extract_workflow_text(res: Any) -> str:
    data = res.get("data") if isinstance(res, dict) else res
    if isinstance(data, dict):
        for key in ("res", "output", "text", "result", "data"):
            if key in data:
                value = data[key]
                if isinstance(value, str):
                    return value
                if isinstance(value, dict):
                    for sub_key in ("res", "text", "output", "result"):
                        if sub_key in value and isinstance(value[sub_key], str):
                            return value[sub_key]
        return json.dumps(data, ensure_ascii=False)
    return str(data).strip()


def call_llm_api(prompt: str, api_type: str = "deepseek") -> str:
    """调用LLM API（支持deepseek和qwen）"""
    if api_type.lower() == "deepseek":
        res = ds_api_retry(prompt)
        return extract_workflow_text(res)
    elif api_type.lower() == "qwen":
        res = run_flow_retry(QWEN_WORKFLOW_ID, {"system_prompt": prompt})
        return extract_workflow_text(res)
    else:
        raise ValueError(f"不支持的API类型: {api_type}，请使用 'deepseek' 或 'qwen'")


def update_operator_lib(
    operator_lib: Dict[str, str],
    used_operators: List[str],
    new_operators: List[str],
    modified_operators: List[Tuple[str, str]]
) -> Dict[str, str]:
    """
    在代码中直接更新操作库
    返回更新后的操作库
    """
    updated_lib = operator_lib.copy()
    
    # 处理修改的操作
    for old_op, new_op in modified_operators:
        if old_op in updated_lib:
            # 如果原操作存在，更新为新操作
            value = updated_lib.pop(old_op)
            updated_lib[new_op] = value
        else:
            # 如果原操作不存在，直接添加新操作
            updated_lib[new_op] = ""
    
    # 添加新操作
    for new_op in new_operators:
        if new_op not in updated_lib:
            updated_lib[new_op] = ""
    
    return updated_lib


def update_user_behavior_lib(
    behavior_lib: Dict[str, str],
    used_behaviors: List[str],
    new_behaviors: List[str],
    modified_behaviors: List[Tuple[str, str]]
) -> Dict[str, str]:
    """
    在代码中直接更新用户行为库
    返回更新后的行为库
    """
    updated_lib = behavior_lib.copy()
    
    # 处理修改的行为
    for old_beh, new_beh in modified_behaviors:
        if old_beh in updated_lib:
            # 如果原行为存在，更新为新行为
            value = updated_lib.pop(old_beh)
            updated_lib[new_beh] = value
        else:
            # 如果原行为不存在，直接添加新行为
            updated_lib[new_beh] = ""
    
    # 添加新行为
    for new_beh in new_behaviors:
        if new_beh not in updated_lib:
            updated_lib[new_beh] = ""
    
    return updated_lib


def extract_from_dialogue(
    dialogue: str, 
    operator_lib: Dict[str, str],
    user_behavior_lib: Dict[str, str],
    api_type: str = "deepseek",
    allow_new: bool = True
) -> Tuple[
    List[str], List[str], List[Tuple[str, str]],  # 客服操作：used, new, modified
    List[str], List[str], List[Tuple[str, str]],  # 用户行为：used, new, modified
    List[Tuple[str, str]]                          # ordered_sequence: [(角色, 操作/行为)]
]:
    """
    从单个对话中提取客服操作和用户行为
    返回: (客服used, 客服new, 客服modified, 用户used, 用户new, 用户modified, ordered_sequence)
    """
    prompt = build_extract_prompt(dialogue, operator_lib, user_behavior_lib, allow_new=allow_new)
    output = call_llm_api(prompt, api_type)
    op_used, op_new, op_modified, ub_used, ub_new, ub_modified, ordered_sequence = parse_extract_output(output)
    return op_used, op_new, op_modified, ub_used, ub_new, ub_modified, ordered_sequence


def deduplicate_memory_with_llm(
    operator_lib: Dict[str, str],
    user_behavior_lib: Dict[str, str],
    api_type: str = "deepseek",
    *,
    round_index: int = 1,
    total_rounds: int = 1,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    使用LLM对memory库进行总结和去重
    返回: (去重后的operator_lib, 去重后的user_behavior_lib)
    """
    round_note = ""
    if total_rounds > 1:
        round_note = (
            f"\n（第 {round_index}/{total_rounds} 轮去重"
            + ("，请在上轮基础上进一步合并仍相近的条目" if round_index > 1 else "")
            + "）"
        )

    print(f"\n开始使用LLM对memory库进行总结和去重{round_note}...")
    
    # 构建operator库去重提示
    operator_list = list(operator_lib.keys())
    operator_text = "\n".join([f"{i+1}. {op}" for i, op in enumerate(operator_list)])
    
    operator_prompt = f"""请对以下客服操作库进行总结和去重{round_note}：

当前操作库（共{len(operator_list)}个操作）：
{operator_text}

要求：
1. 识别语义相同或相似的操作，将它们合并为一个操作
2. 对于可以合并的操作，选择一个最合适的名称作为最终操作名
3. 对于语义不完整或表达不清晰的操作，进行优化和修改
4. 保留所有有意义的操作，只去除重复项
5. 输出格式：每行一个操作名，合并的操作只保留一个

请输出去重后的操作列表（每行一个操作名）："""
    
    # 构建用户行为库去重提示
    behavior_list = list(user_behavior_lib.keys())
    behavior_text = "\n".join([f"{i+1}. {beh}" for i, beh in enumerate(behavior_list)])
    
    behavior_prompt = f"""请对以下用户行为库进行总结和去重{round_note}：

当前行为库（共{len(behavior_list)}个行为）：
{behavior_text}

要求：
1. 识别语义相同或非常相似的行为，将它们合并为一个行为
2. 对于可以合并的行为，选择一个最合适的名称作为最终行为名
3. 对于语义不完整或表达不清晰的行为，进行优化和修改
4. 保留所有有意义的行为，只去除重复项
5. 输出格式：每行一个行为名，合并的行为只保留一个

请输出去重后的行为列表（每行一个行为名）："""
    
    # 调用LLM进行去重
    print("  正在对客服操作库进行去重...")
    operator_output = call_llm_api(operator_prompt, api_type)
    
    print("  正在对用户行为库进行去重...")
    behavior_output = call_llm_api(behavior_prompt, api_type)
    
    # 解析输出
    new_operator_lib = {}
    for line in operator_output.split('\n'):
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('//'):
            # 移除行号等前缀
            line = re.sub(r'^\d+[\.\)、]\s*', '', line)
            if line:
                new_operator_lib[line] = ""
    
    new_behavior_lib = {}
    for line in behavior_output.split('\n'):
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('//'):
            # 移除行号等前缀
            line = re.sub(r'^\d+[\.\)、]\s*', '', line)
            if line:
                new_behavior_lib[line] = ""
    
    print(f"  去重完成：")
    print(f"    - 客服操作库：{len(operator_lib)} -> {len(new_operator_lib)}")
    print(f"    - 用户行为库：{len(user_behavior_lib)} -> {len(new_behavior_lib)}")

    if not new_operator_lib:
        print("  [WARN] 客服操作库去重结果为空，保留原库")
        new_operator_lib = dict(operator_lib)
    elif set(new_operator_lib) == set(operator_lib):
        print("  [NOTE] 客服操作库条目集合未变化")

    if not new_behavior_lib:
        print("  [WARN] 用户行为库去重结果为空，保留原库")
        new_behavior_lib = dict(user_behavior_lib)
    elif set(new_behavior_lib) == set(user_behavior_lib):
        print("  [NOTE] 用户行为库条目集合未变化")
    
    return new_operator_lib, new_behavior_lib


def find_similar_key_pairs(
    keys: List[str],
    threshold: float = 0.78,
    max_pairs: int = 40,
) -> List[Tuple[str, str, float]]:
    """Return similar (a, b, ratio) pairs for candidate merge."""
    pairs: List[Tuple[str, str, float]] = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ratio = SequenceMatcher(None, a, b).ratio()
            if ratio >= threshold:
                pairs.append((a, b, ratio))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:max_pairs]


def merge_candidate_pairs_with_llm(
    lib: Dict[str, str],
    api_type: str,
    *,
    label: str = "操作",
    threshold: float = 0.78,
    max_pairs: int = 40,
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Round-2 style dedup: ask LLM to merge high-similarity pairs only."""
    keys = sorted(lib.keys())
    candidates = find_similar_key_pairs(keys, threshold=threshold, max_pairs=max_pairs)
    if not candidates:
        print(f"  [NOTE] {label}库无相似候选对（threshold={threshold}），跳过候选合并")
        return lib, []

    pair_lines = "\n".join(
        f"{i + 1}. {a}  /  {b}  （相似度 {ratio:.2f}）"
        for i, (a, b, ratio) in enumerate(candidates)
    )
    prompt = f"""以下是 {label}库中语义可能重复的候选对。请逐对判断是否要合并为一条。

{pair_lines}

输出要求（每对一行，共 {len(candidates)} 行）：
- 不合并：KEEP
- 合并：MERGE|最终保留的名称

「最终保留的名称」必须从该对的两个原名中选一个，或给出更简洁的合并名。
不要输出解释或其它内容。"""

    print(f"  正在对 {len(candidates)} 对相似{label}做候选合并…")
    output = call_llm_api(prompt, api_type)
    updated = dict(lib)
    merges: List[Dict[str, Any]] = []
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]

    for idx, (a, b, ratio) in enumerate(candidates):
        action = "KEEP"
        canonical = a
        if idx < len(lines):
            line = lines[idx]
            line = re.sub(r"^\d+[\.\)、]\s*", "", line)
            if line.upper().startswith("MERGE|"):
                parts = line.split("|", 1)
                if len(parts) == 2 and parts[1].strip():
                    action = "MERGE"
                    canonical = parts[1].strip()
            elif line.upper().startswith("MERGE:"):
                action = "MERGE"
                canonical = line.split(":", 1)[1].strip()
            elif line.upper() == "KEEP":
                action = "KEEP"

        if action != "MERGE":
            continue
        if a not in updated and b not in updated:
            continue
        if canonical not in (a, b):
            # Prefer canonical; drop both old if neither matches
            if a in updated:
                updated.pop(a, None)
            if b in updated and b != a:
                updated.pop(b, None)
            updated.setdefault(canonical, "")
        else:
            drop = b if canonical == a else a
            if drop in updated:
                updated.pop(drop, None)
            updated.setdefault(canonical, updated.get(canonical, ""))
        merges.append({
            "a": a, "b": b, "ratio": round(ratio, 4),
            "canonical": canonical, "action": action,
        })

    print(f"  候选合并：{len(candidates)} 对中实际合并 {len(merges)} 对")
    return updated, merges


def deduplicate_memory_rounds(
    operator_lib: Dict[str, str],
    user_behavior_lib: Dict[str, str],
    api_type: str = "deepseek",
    rounds: int = 2,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Any]]:
    """Run LLM dedup: round1 full list, round2+ similar-pair merge."""
    rounds = max(1, int(rounds))
    report: Dict[str, Any] = {
        "rounds": rounds,
        "steps": [],
        "operator_before": len(operator_lib),
        "behavior_before": len(user_behavior_lib),
    }

    n_op_before = len(operator_lib)
    n_ub_before = len(user_behavior_lib)
    operator_lib, user_behavior_lib = deduplicate_memory_with_llm(
        operator_lib,
        user_behavior_lib,
        api_type,
        round_index=1,
        total_rounds=rounds,
    )
    report["steps"].append({
        "round": 1,
        "mode": "full_list",
        "operator": {"before": n_op_before, "after": len(operator_lib)},
        "behavior": {"before": n_ub_before, "after": len(user_behavior_lib)},
    })
    print(
        f"  第 1/{rounds} 轮去重: "
        f"客服 {n_op_before} -> {len(operator_lib)}, "
        f"用户 {n_ub_before} -> {len(user_behavior_lib)}"
    )

    for i in range(1, rounds):
        n_op = len(operator_lib)
        n_ub = len(user_behavior_lib)
        operator_lib, op_merges = merge_candidate_pairs_with_llm(
            operator_lib, api_type, label="客服操作",
        )
        user_behavior_lib, ub_merges = merge_candidate_pairs_with_llm(
            user_behavior_lib, api_type, label="用户行为",
        )
        report["steps"].append({
            "round": i + 1,
            "mode": "similar_pair",
            "operator": {
                "before": n_op,
                "after": len(operator_lib),
                "merges": op_merges,
            },
            "behavior": {
                "before": n_ub,
                "after": len(user_behavior_lib),
                "merges": ub_merges,
            },
        })
        print(
            f"  第 {i + 1}/{rounds} 轮候选合并: "
            f"客服 {n_op} -> {len(operator_lib)}, "
            f"用户 {n_ub} -> {len(user_behavior_lib)}"
        )

    report["operator_after"] = len(operator_lib)
    report["behavior_after"] = len(user_behavior_lib)
    report["operator_changed"] = report["operator_before"] != report["operator_after"]
    report["behavior_changed"] = report["behavior_before"] != report["behavior_after"]
    if not report["operator_changed"] and not report["behavior_changed"]:
        print(
            "  [WARN] 去重后 operator/behavior 数量均未变化；"
            "可能 LLM 未合并，详见 operator_dedup_report.json"
        )
    return operator_lib, user_behavior_lib, report


def normalize_session_operator_sequences(
    results: List[Dict[str, Any]],
    operator_lib: Dict[str, str],
    user_behavior_lib: Dict[str, str],
    *,
    min_similarity: float = 0.52,
) -> List[Dict[str, Any]]:
    """
    Stage2 已经要求 LLM 只能从库中选择操作名，此处仅做统计校验，
    不再做模糊匹配归一化（prompt 已强调逐字复制 key）。
    """
    all_keys = set(operator_lib) | set(user_behavior_lib)
    exact_count = 0
    dropped_count = 0

    for item in results:
        ordered_ops = item.get("ordered_operations") or []
        validated_ops: List[List[str]] = []
        for role, op in ordered_ops:
            if not op or not role:
                continue
            op = op.strip()
            if not op:
                continue
            if op in all_keys:
                validated_ops.append([str(role).strip(), op])
                exact_count += 1
            else:
                dropped_count += 1
        item["ordered_operations"] = validated_ops

    steps = sum(len(item.get("ordered_operations") or []) for item in results)
    unique_ops = len(
        {
            f"{role}:{op}"
            for item in results
            for role, op in item.get("ordered_operations") or []
            if role and op
        }
    )
    print(
        "  会话->标准操作符序列："
        f"{len(results)} 条会话，{steps} 步，"
        f"图中约 {unique_ops} 个节点；"
        f"精确 {exact_count}，"
        f"丢弃 {dropped_count}"
    )
    return results


def normalize_results_file_in_dir(
    output_dir: Path,
    *,
    min_similarity: float = 0.52,
) -> List[Dict[str, Any]]:
    """仅对已有 operator_results.json 做序列归一化（不重新调用 LLM 抽取）。"""
    global OPERATOR_OUTPUT_JSON, OPERATOR_OUTPUT_TXT
    global OPERATOR_MEMORY_JSON, USER_BEHAVIOR_MEMORY_JSON

    results_path = output_dir / "operator_results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"未找到 {results_path}")

    OPERATOR_OUTPUT_JSON = results_path
    OPERATOR_OUTPUT_TXT = output_dir / "operator_results.txt"
    OPERATOR_MEMORY_JSON = output_dir / "operator_memory.json"
    USER_BEHAVIOR_MEMORY_JSON = output_dir / "user_behavior_memory.json"

    operator_lib = load_operator_memory()
    user_behavior_lib = load_user_behavior_memory()
    with results_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"\n正在归一化 {results_path.name} …")
    normalized = normalize_session_operator_sequences(
        results,
        operator_lib,
        user_behavior_lib,
        min_similarity=min_similarity,
    )
    save_results(normalized, operator_lib, user_behavior_lib)
    return normalized


def dedup_memory_in_dir(
    output_dir: Path,
    api_type: str,
    *,
    dedup_rounds: int = 2,
    rerun_stage2: bool = True,
    excel_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, str]]:
    """仅对已有 memory 库做 LLM 去重；可选重跑 Stage2 重标注（跳过 Stage1）。

    需要 ``operator_memory.json`` / ``user_behavior_memory.json`` 已存在。
    若 ``rerun_stage2=True``，还需 ``excel_path``（或 output_dir/对话数据.xlsx）。
    """
    global OPERATOR_OUTPUT_JSON, OPERATOR_OUTPUT_TXT
    global OPERATOR_MEMORY_JSON, USER_BEHAVIOR_MEMORY_JSON, STATE_PATH

    OPERATOR_OUTPUT_JSON = output_dir / "operator_results.json"
    OPERATOR_OUTPUT_TXT = output_dir / "operator_results.txt"
    OPERATOR_MEMORY_JSON = output_dir / "operator_memory.json"
    USER_BEHAVIOR_MEMORY_JSON = output_dir / "user_behavior_memory.json"
    STATE_PATH = output_dir / "operator_extract_state.json"

    operator_lib = load_operator_memory()
    user_behavior_lib = load_user_behavior_memory()
    if not operator_lib and not user_behavior_lib:
        raise FileNotFoundError(
            f"未找到可用的 operator/user memory：{OPERATOR_MEMORY_JSON}"
        )

    n_op_before = len(operator_lib)
    n_ub_before = len(user_behavior_lib)
    print(
        f"\n加载已有 memory：客服操作 {n_op_before} 个，"
        f"用户行为 {n_ub_before} 个"
    )

    operator_lib, user_behavior_lib, dedup_report = deduplicate_memory_rounds(
        operator_lib, user_behavior_lib, api_type, rounds=max(1, dedup_rounds),
    )
    save_operator_memory(operator_lib)
    save_user_behavior_memory(user_behavior_lib)
    report_path = output_dir / "operator_dedup_report.json"
    report_path.write_text(
        json.dumps(dedup_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  [OK] 去重报告 -> {report_path.name}")
    print(
        f"  [OK] 去重完成：客服 {n_op_before} -> {len(operator_lib)}，"
        f"用户 {n_ub_before} -> {len(user_behavior_lib)}"
    )

    if not rerun_stage2:
        if OPERATOR_OUTPUT_JSON.is_file():
            with OPERATOR_OUTPUT_JSON.open("r", encoding="utf-8") as f:
                results = json.load(f)
        else:
            results = []
        print(
            "  [NOTE] 未重跑 Stage2，operator_results.json 未更新；"
            "后续建图请加上 --operator-dedup-only（默认会重跑 Stage2）"
        )
        return results, operator_lib, user_behavior_lib

    excel_path = excel_path or (output_dir / "对话数据.xlsx")
    if not excel_path.is_file():
        raise FileNotFoundError(
            f"重跑 Stage2 需要对话 Excel：{excel_path}"
        )

    df, dialogue_col = load_excel_data(excel_path)
    print("\n  > Stage2 重标注（只从去重后的库中选择，不新增）")
    stage2_results, operator_lib, user_behavior_lib = process_stage(
        df,
        dialogue_col,
        operator_lib,
        user_behavior_lib,
        api_type,
        allow_new=False,
        stage_name="Stage2（去重后重标注）",
    )

    print("\n  > 会话映射为标准操作符序列")
    stage2_results = normalize_session_operator_sequences(
        stage2_results, operator_lib, user_behavior_lib,
    )

    from session2hg_v2 import collapse_operator_results_sequences

    stage2_results, collapsed_steps = collapse_operator_results_sequences(
        stage2_results,
    )
    if collapsed_steps:
        print(f"  > 合并连续重复操作 {collapsed_steps} 步")

    save_results(stage2_results, operator_lib, user_behavior_lib)
    print(f"  [OK] operator_results.json 已更新（{len(stage2_results)} 条）")
    return stage2_results, operator_lib, user_behavior_lib


def save_results(results: List[Dict[str, Any]], operator_lib: Dict[str, str], user_behavior_lib: Dict[str, str]):
    """保存结果"""
    # 保存JSON结果
    OPERATOR_OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OPERATOR_OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存文本结果（所有操作和行为）
    with open(OPERATOR_OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("=== 客服操作库 ===\n")
        for op, desc in operator_lib.items():
            if desc:
                f.write(f"{op}: {desc}\n")
            else:
                f.write(f"{op}\n")
        
        f.write("\n=== 用户行为库 ===\n")
        for beh, desc in user_behavior_lib.items():
            if desc:
                f.write(f"{beh}: {desc}\n")
            else:
                f.write(f"{beh}\n")


def process_stage(
    df: pd.DataFrame,
    dialogue_col: str,
    operator_lib: Dict[str, str],
    user_behavior_lib: Dict[str, str],
    api_type: str,
    allow_new: bool,
    stage_name: str
) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, str]]:
    """
    处理一个阶段的对话数据
    返回: (results, updated_operator_lib, updated_user_behavior_lib)
    """
    print(f"\n{'=' * 60}")
    print(f"{stage_name}")
    print(f"{'=' * 60}")
    
    results = []
    total_dialogues = len(df)
    total_batches = (total_dialogues + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\n数据统计:")
    print(f"  - 总对话数: {total_dialogues}")
    print(f"  - 总批次数: {total_batches}")
    
    # 使用tqdm显示详细进度
    with tqdm(
        total=total_dialogues, 
        desc=f"{stage_name} - 处理对话", 
        unit="条",
        ncols=100
    ) as pbar:
        for batch_idx in range(total_batches):
            pos = batch_idx * BATCH_SIZE
            batch = df.iloc[pos:pos + BATCH_SIZE]
            batch_used_ops = []
            batch_new_ops = []
            batch_modified_ops = []
            batch_used_behaviors = []
            batch_new_behaviors = []
            batch_modified_behaviors = []
            
            # 处理批次中的每个对话
            for idx, row in batch.iterrows():
                dialogue = str(row[dialogue_col]).strip()
                if not dialogue or dialogue == "nan":
                    # 空对话也更新进度条，但不处理
                    pbar.update(1)
                    continue
                
                try:
                    # 提取客服操作和用户行为
                    op_used, op_new, op_modified, ub_used, ub_new, ub_modified, ordered_sequence = extract_from_dialogue(
                        dialogue, operator_lib, user_behavior_lib, api_type, allow_new=allow_new
                    )

                    if allow_new:
                        # Qwen 偶尔会把空库中不存在的条目放到 <used>。
                        # 第一阶段允许新增时，将这些条目提升为 new，避免库一直不增长。
                        promoted_ops = [
                            op for op in op_used
                            if op and op not in operator_lib and op not in op_new
                        ]
                        if promoted_ops:
                            op_new.extend(promoted_ops)
                            op_used = [op for op in op_used if op not in promoted_ops]

                        promoted_behaviors = [
                            beh for beh in ub_used
                            if beh and beh not in user_behavior_lib and beh not in ub_new
                        ]
                        if promoted_behaviors:
                            ub_new.extend(promoted_behaviors)
                            ub_used = [beh for beh in ub_used if beh not in promoted_behaviors]
                    
                    # 收集批次的客服操作
                    batch_used_ops.extend(op_used)
                    if allow_new:
                        batch_new_ops.extend(op_new)
                        batch_modified_ops.extend(op_modified)
                    
                    # 收集批次的用户行为
                    batch_used_behaviors.extend(ub_used)
                    if allow_new:
                        batch_new_behaviors.extend(ub_new)
                        batch_modified_behaviors.extend(ub_modified)
                    
                    # 处理ordered_sequence
                    if allow_new:
                        # 第一阶段：不需要ordered_sequence，设为空列表
                        ordered_sequence = []
                    else:
                        # 第二阶段：必须有ordered_sequence，如果没有则从used构建
                        if not ordered_sequence:
                            ordered_operations = []
                            for beh in ub_used:
                                ordered_operations.append(("用户", beh))
                            for old_beh, new_beh in ub_modified:
                                ordered_operations.append(("用户", new_beh))
                            for op in op_used:
                                ordered_operations.append(("客服", op))
                            for old_op, new_op in op_modified:
                                ordered_operations.append(("客服", new_op))
                            ordered_sequence = ordered_operations
                    
                    # 记录结果
                    result_item = {
                        "index": int(idx),
                        "session_id": str(row.get("session_id", "")),
                        "dialogue": dialogue[:200] + "..." if len(dialogue) > 200 else dialogue,
                        "客服操作": {
                            "used_operators": op_used,
                            "new_operators": op_new if allow_new else [],
                            "modified_operators": [{"old": old, "new": new} for old, new in op_modified] if allow_new else []
                        },
                        "用户行为": {
                            "used_behaviors": ub_used,
                            "new_behaviors": ub_new if allow_new else [],
                            "modified_behaviors": [{"old": old, "new": new} for old, new in ub_modified] if allow_new else []
                        }
                    }
                    
                    # 只在第二阶段添加ordered_operations
                    if not allow_new:
                        result_item["ordered_operations"] = [[role, op] for role, op in ordered_sequence]
                    
                    results.append(result_item)
                    
                    # 更新进度条信息
                    pbar.set_postfix({
                        "客服库": len(operator_lib),
                        "用户库": len(user_behavior_lib),
                        "新增客服": len(batch_new_ops) if allow_new else 0,
                        "新增用户": len(batch_new_behaviors) if allow_new else 0
                    })
                    
                except Exception as e:
                    tqdm.write(f"处理第 {idx} 条记录时出错: {e}")
                    results.append({
                        "index": int(idx),
                        "session_id": str(row.get("session_id", "")),
                        "dialogue": dialogue[:200] + "..." if len(dialogue) > 200 else dialogue,
                        "客服操作": {
                            "used_operators": [],
                            "new_operators": [],
                            "modified_operators": []
                        },
                        "用户行为": {
                            "used_behaviors": [],
                            "new_behaviors": [],
                            "modified_behaviors": []
                        },
                        "ordered_operations": [],
                        "error": str(e)
                    })
                
                # 更新进度条
                pbar.update(1)
            
            # 从new和modified中提取所有操作，确保所有操作都被添加到库中（仅第一阶段）
            if allow_new:
                # 确保new和modified中的所有操作都被添加到库中
                for op in batch_new_ops:
                    if op not in operator_lib:
                        operator_lib[op] = ""
                
                for old_op, new_op in batch_modified_ops:
                    if new_op not in operator_lib:
                        operator_lib[new_op] = ""
                
                for beh in batch_new_behaviors:
                    if beh not in user_behavior_lib:
                        user_behavior_lib[beh] = ""
                
                for old_beh, new_beh in batch_modified_behaviors:
                    if new_beh not in user_behavior_lib:
                        user_behavior_lib[new_beh] = ""
                
                # 在代码中直接更新操作库（不再调用LLM）
                if batch_new_ops or batch_modified_ops:
                    operator_lib = update_operator_lib(
                        operator_lib, 
                        batch_used_ops, 
                        batch_new_ops, 
                        batch_modified_ops
                    )
                    save_operator_memory(operator_lib)
                    tqdm.write(f"批次 {batch_idx + 1}: 客服操作库已更新 - 新增 {len(batch_new_ops)} 个操作, 修改 {len(batch_modified_ops)} 个操作 (当前库大小: {len(operator_lib)})")
                
                # 在代码中直接更新用户行为库
                if batch_new_behaviors or batch_modified_behaviors:
                    user_behavior_lib = update_user_behavior_lib(
                        user_behavior_lib,
                        batch_used_behaviors,
                        batch_new_behaviors,
                        batch_modified_behaviors
                    )
                    save_user_behavior_memory(user_behavior_lib)
                    tqdm.write(f"批次 {batch_idx + 1}: 用户行为库已更新 - 新增 {len(batch_new_behaviors)} 个行为, 修改 {len(batch_modified_behaviors)} 个行为 (当前库大小: {len(user_behavior_lib)})")
    
    return results, operator_lib, user_behavior_lib


def main():
    """主函数 - 两阶段处理流程"""
    print("=" * 60)
    print("客服操作提取脚本（两阶段处理：抽取 -> 去重 -> 重新标注）")
    print("=" * 60)
    
    # 选择LLM API
    api_type = DEFAULT_LLM.lower()
    if api_type not in ["deepseek", "qwen"]:
        print(f"\n警告: 无效的API类型 '{api_type}'，使用默认值 'deepseek'")
        api_type = "deepseek"
    
    print(f"\n使用LLM API: {api_type.upper()}")
    print("提示: 可通过环境变量 LLM_API=deepseek 或 LLM_API=qwen 来切换")
    
    # 1. 选择Excel文件
    excel_files = list(DATA_BY_INTENT_DIR.glob("*.xlsx"))
    if not excel_files:
        print(f"\n错误: 在 {DATA_BY_INTENT_DIR} 中未找到Excel文件")
        print("请先运行 single_intent_data.py 生成数据文件")
        return
    
    print(f"\n找到 {len(excel_files)} 个Excel文件:")
    for idx, file in enumerate(excel_files, 1):
        print(f"  {idx}. {file.name}")
    
    if len(excel_files) == 1:
        selected_file = excel_files[0]
        print(f"\n自动选择: {selected_file.name}")
    else:
        try:
            choice = input(f"\n请选择要处理的文件（1-{len(excel_files)}）: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(excel_files):
                selected_file = excel_files[idx]
            else:
                print("无效选择")
                return
        except (ValueError, KeyboardInterrupt):
            print("已取消")
            return
    
    # 2. 加载数据
    try:
        df, dialogue_col = load_excel_data(selected_file)
    except Exception as e:
        print(f"\n错误: {e}")
        return
    
    # 3. 加载已有操作库和用户行为库
    operator_lib = load_operator_memory()
    user_behavior_lib = load_user_behavior_memory()
    print(f"\n已加载操作库，当前有 {len(operator_lib)} 个客服操作")
    print(f"已加载用户行为库，当前有 {len(user_behavior_lib)} 个用户行为")
    
    # 4. 第一阶段：从会话中抽取新增/修改的操作符号
    print("\n" + "=" * 60)
    print("第一阶段：从会话中抽取操作符号（允许新增和修改）")
    print("=" * 60)
    
    stage1_results, operator_lib, user_behavior_lib = process_stage(
        df, dialogue_col, operator_lib, user_behavior_lib, api_type, 
        allow_new=True, stage_name="第一阶段"
    )
    
    print(f"\n第一阶段完成！")
    print(f"  - 处理了 {len(stage1_results)} 条有效对话")
    print(f"  - 客服操作库中共有 {len(operator_lib)} 个操作")
    print(f"  - 用户行为库中共有 {len(user_behavior_lib)} 个行为")
    
    # 5. 中间阶段：使用LLM对memory库进行总结和去重（2 轮）
    print("\n" + "=" * 60)
    print("中间阶段：使用LLM对memory库进行总结和去重（2 轮）")
    print("=" * 60)
    
    operator_lib, user_behavior_lib, _dedup_report = deduplicate_memory_rounds(
        operator_lib, user_behavior_lib, api_type, rounds=2,
    )
    
    # 保存去重后的memory库
    save_operator_memory(operator_lib)
    save_user_behavior_memory(user_behavior_lib)
    
    print(f"\n中间阶段完成！")
    print(f"  - 去重后的客服操作库已保存到: {OPERATOR_MEMORY_JSON.name}")
    print(f"  - 去重后的用户行为库已保存到: {USER_BEHAVIOR_MEMORY_JSON.name}")
    
    # 6. 第二阶段：重新处理会话，只从memory库中选择，不新增
    print("\n" + "=" * 60)
    print("第二阶段：重新处理会话（只从memory库中选择，不新增）")
    print("=" * 60)
    
    stage2_results, operator_lib, user_behavior_lib = process_stage(
        df, dialogue_col, operator_lib, user_behavior_lib, api_type,
        allow_new=False, stage_name="第二阶段"
    )
    
    # 7. 保存最终结果
    save_results(stage2_results, operator_lib, user_behavior_lib)
    
    # 8. 完成
    print(f"\n" + "=" * 60)
    print("所有阶段处理完成！")
    print("=" * 60)
    print(f"  - 第一阶段处理了 {len(stage1_results)} 条有效对话")
    print(f"  - 第二阶段处理了 {len(stage2_results)} 条有效对话")
    print(f"  - 最终客服操作库中共有 {len(operator_lib)} 个操作")
    print(f"  - 最终用户行为库中共有 {len(user_behavior_lib)} 个行为")
    print(f"  - 最终结果已保存到: {OPERATOR_OUTPUT_JSON.name}")
    print(f"  - 最终客服操作库已保存到: {OPERATOR_MEMORY_JSON.name}")
    print(f"  - 最终用户行为库已保存到: {USER_BEHAVIOR_MEMORY_JSON.name}")


if __name__ == "__main__":
    DATA_BY_INTENT_DIR.mkdir(parents=True, exist_ok=True)
    main()


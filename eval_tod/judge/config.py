"""LLM Judge 配置 —— 多智能体打分系统的维度、评判官定义、LLM 参数。"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════
# 评分维度（ToD / MultiWOZ 场景）
# ═══════════════════════════════════════════════════════════════════
SCORING_DIMENSIONS: dict[str, dict] = {
    "task_completion": {
        "range": [1, 5],
        "description": "任务是否成功完成？Agent 是否达成了用户 goal 中的所有约束条件（inform + request + booking）？",
    },
    "slot_accuracy": {
        "range": [1, 5],
        "description": "Agent 填充的槽位值是否准确？是否有遗漏、错误或虚构的信息？",
    },
    "dialogue_fluency": {
        "range": [1, 5],
        "description": "对话是否自然流畅？Agent 的回复是否连贯、无重复、逻辑清晰？",
    },
    "helpfulness": {
        "range": [1, 5],
        "description": "Agent 的回复是否对用户有实际帮助？是否主动理解用户需求并提供有效信息？",
    },
    "efficiency": {
        "range": [1, 5],
        "description": "Agent 是否以合理的对话轮次完成了任务？是否有不必要的重复或低效的信息收集？",
    },
}

# ═══════════════════════════════════════════════════════════════════
# 评判官定义 —— 每位评判官有不同的视角
# ═══════════════════════════════════════════════════════════════════
JUDGE_DEFINITIONS: dict[str, dict] = {
    "task_judge": {
        "name": "Task Completion Judge",
        "focus": "task_completion",
        "role": (
            "You are a task-oriented dialogue evaluation specialist focused on "
            "task completion. Your role is to assess whether the agent successfully "
            "fulfilled the user's goal — including all inform constraints, "
            "requested information, and booking requirements. Check each constraint "
            "against the agent's predictions and the dialogue flow."
        ),
    },
    "slot_judge": {
        "name": "Slot Accuracy Judge",
        "focus": "slot_accuracy",
        "role": (
            "You are a slot-filling accuracy specialist for task-oriented dialogue "
            "systems. Your role is to assess whether the agent correctly identified "
            "and communicated slot values (hotel type, price range, location, etc.) "
            "matching the user's goal. Check for hallucinated values, missing slots, "
            "and incorrect slot substitutions."
        ),
    },
    "fluency_judge": {
        "name": "Fluency & Coherence Judge",
        "focus": "dialogue_fluency",
        "role": (
            "You are a dialogue quality specialist focused on conversational fluency. "
            "Your role is to assess whether the agent's utterances are natural, "
            "coherent, well-structured, and free of repetition or nonsensical output. "
            "Evaluate the overall flow and readability of the conversation."
        ),
    },
    "helpfulness_judge": {
        "name": "Helpfulness Judge",
        "focus": "helpfulness",
        "role": (
            "You are a user-experience specialist evaluating how helpful the agent's "
            "responses are. Your role is to assess whether the agent proactively "
            "understands user needs, provides relevant and actionable information, "
            "and demonstrates genuine assistance throughout the dialogue."
        ),
    },
    "efficiency_judge": {
        "name": "Efficiency Judge",
        "focus": "efficiency",
        "role": (
            "You are a dialogue efficiency specialist. Your role is to assess "
            "whether the agent completed the task with an appropriate number of "
            "turns, avoided unnecessary repetition or redundant information "
            "collection, and drove the conversation toward resolution efficiently."
        ),
    },
}

# Combiner 定义
COMBINER_DEFINITION: dict[str, str] = {
    "name": "Chief Evaluator",
    "role": (
        "You are a senior quality assurance expert for task-oriented dialogue "
        "systems. Your role is to review the independent evaluations from "
        "multiple specialist judges, identify areas of agreement and disagreement, "
        "and synthesize their findings into a final, balanced assessment. "
        "Give appropriate weight to each specialist's domain expertise when "
        "resolving conflicts."
    ),
}

# ═══════════════════════════════════════════════════════════════════
# LLM 配置（OpenAI 兼容接口）
# ═══════════════════════════════════════════════════════════════════
LLM_CONFIG: dict = {
    "model": "deepseek-chat",
    "max_tokens": 1024,
    "temperature": 0.3,
}

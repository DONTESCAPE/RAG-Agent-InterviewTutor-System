from __future__ import annotations

from typing import Dict, List

from utils.logger_handler import logger


class InterviewRoleManager:
    """根据岗位生成更专业的面试问题与追问方向。"""

    DEFAULT_ROLE = "通用技术岗位"
#模拟面试这一板块后续可以采用LoRA微调去定制实现面试官风格的多样性。另外以下各岗位题库可以去进行扩充，使得面试时更加多样化，
#后续还可根据题目难易以及知识点分类来进行标注处理，制作可复用的数据集，在后续的LoRA微调里使用
#另外也可增添更多岗位，本次只挑选了部分岗位。
    ROLE_PROFILES: Dict[str, Dict[str, List[str]]] = {
        "后端开发": {
            "questions": [
                "请介绍一下你对后端系统分层设计的理解，以及你如何组织项目架构。",
                "如果要优化接口响应速度，你会从哪些方面入手？",
                "请说说你如何设计一个高并发场景下的缓存策略。",
                "如果数据库成为瓶颈，你会如何排查并优化？",
            ],
            "keywords": ["接口", "缓存", "数据库", "并发", "架构", "性能", "事务", "消息队列"],
        },
        "前端开发": {
            "questions": [
                "请介绍一下你对前端工程化的理解，以及项目通常如何组织。",
                "如果页面渲染卡顿，你会从哪些角度排查性能问题？",
                "请说说你如何设计组件复用与状态管理。",
                "如果接口请求很多且复杂，你会如何处理页面数据流？",
            ],
            "keywords": ["组件", "状态管理", "性能优化", "工程化", "渲染", "响应式", "数据流"],
        },
        "数据分析": {
            "questions": [
                "请介绍一下你常用的数据分析方法，以及如何保证分析结论可靠。",
                "如果数据样本存在缺失值，你通常如何处理？",
                "请说说你如何从业务目标出发设计分析指标。",
                "如果需要向业务方解释分析结果，你会如何表达？",
            ],
            "keywords": ["指标", "样本", "缺失值", "分析", "业务", "报表", "SQL", "可视化"],
        },
        "算法工程": {
            "questions": [
                "请介绍一下你对模型训练和推理流程的理解。",
                "如果模型效果不好，你会从哪些方向排查？",
                "请说说你如何评估一个算法模型的效果。",
                "如果需要落地一个模型服务，你会关注哪些工程问题？",
            ],
            "keywords": ["模型", "训练", "推理", "评估", "特征", "服务化", "召回", "精度"],
        },
        "产品经理": {
            "questions": [
                "请介绍一下你如何理解一个产品从需求到落地的完整流程。",
                "如果用户反馈功能不好用，你会如何分析问题？",
                "请说说你如何评估一个产品功能是否值得做。",
                "如果要推进跨部门协作，你会如何沟通？",
            ],
            "keywords": ["需求", "用户", "评估", "协作", "功能", "数据", "增长"],
        },
    }

    def normalize_role(self, role: str) -> str:
        text = (role or "").strip()
        if not text:
            return self.DEFAULT_ROLE
        return text

    def get_role_profile(self, role: str) -> Dict[str, List[str]]:
        normalized = self.normalize_role(role)
        for key, profile in self.ROLE_PROFILES.items():
            if key in normalized:
                return profile
        logger.info(f"[InterviewRoleManager] 未命中预设岗位，使用通用岗位：{normalized}")
        return {
            "questions": [
                "请先介绍一下你的项目经历，以及你在其中承担的核心职责。",
                "请说说你在项目中遇到过的一个主要技术难点，以及你是怎么解决的。",
                "如果让你继续优化这个项目，你会优先做哪些改进？",
                "请结合你的经历，说说你最有代表性的技术亮点。",
            ],
            "keywords": ["项目", "职责", "难点", "优化", "技术亮点", "经验"],
        }

    def get_first_question(self, role: str) -> str:
        profile = self.get_role_profile(role)
        return profile["questions"][0] if profile["questions"] else "请先简单做一下自我介绍。"

    def get_next_question(self, role: str, index: int) -> str:
        profile = self.get_role_profile(role)
        questions = profile["questions"]
        if not questions:
            return "请继续介绍一下你的相关经验。"
        return questions[index % len(questions)]

    def get_role_keywords(self, role: str) -> List[str]:
        profile = self.get_role_profile(role)
        return profile.get("keywords", [])

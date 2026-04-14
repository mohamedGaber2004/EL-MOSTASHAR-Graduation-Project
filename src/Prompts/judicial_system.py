"""
نظام دعم القرار القضائي للمحاكم المصرية
Egyptian Judicial Decision Support System

هذا الملف يجمع جميع وكلاء النظام السبعة
"""

from typing import Dict, List, Any

# استيراد جميع الوكلاء
from orchestrator_agent import (
    OrchestratorAgentConfig,
    get_orchestrator_prompt
)
from data_ingestion_agent import (
    DataIngestionAgentConfig,
    get_data_ingestion_prompt,
    NodeTemplate
)
from procedural_auditor_agent import (
    ProceduralAuditorAgentConfig,
    get_procedural_auditor_prompt,
    ProceduralAuditResult
)
from legal_researcher_agent import (
    LegalResearcherAgentConfig,
    get_legal_researcher_prompt,
    LegalResearchResult
)
from evidence_analyst_agent import (
    EvidenceAnalystAgentConfig,
    get_evidence_analyst_prompt,
    EvidenceAnalysis
)
from defense_analysis_agent import (
    DefenseAnalysisAgentConfig,
    get_defense_analysis_prompt,
    DefenseAnalysis
)
from judge_agent import (
    JudgeAgentConfig,
    get_judge_prompt,
    JudgmentDraft
)


class JudicialDecisionSupportSystem:
    """
    نظام دعم القرار القضائي المتكامل
    """
    
    def __init__(self):
        """تهيئة النظام وجميع وكلائه"""
        self.agents = {
            "orchestrator": OrchestratorAgentConfig(),
            "data_ingestion": DataIngestionAgentConfig(),
            "procedural_auditor": ProceduralAuditorAgentConfig(),
            "legal_researcher": LegalResearcherAgentConfig(),
            "evidence_analyst": EvidenceAnalystAgentConfig(),
            "defense_analyst": DefenseAnalysisAgentConfig(),
            "judge": JudgeAgentConfig()
        }
    
    def get_all_prompts(self) -> Dict[str, str]:
        """
        الحصول على جميع نصوص التعليمات لكل وكيل
        
        Returns:
            قاموس يحتوي على اسم الوكيل ونص تعليماته
        """
        return {
            "orchestrator": get_orchestrator_prompt(),
            "data_ingestion": get_data_ingestion_prompt(),
            "procedural_auditor": get_procedural_auditor_prompt(),
            "legal_researcher": get_legal_researcher_prompt(),
            "evidence_analyst": get_evidence_analyst_prompt(),
            "defense_analyst": get_defense_analysis_prompt(),
            "judge": get_judge_prompt()
        }
    
    def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """
        الحصول على معلومات وكيل محدد
        
        Args:
            agent_name: اسم الوكيل
            
        Returns:
            معلومات الوكيل
        """
        if agent_name not in self.agents:
            return {"error": f"الوكيل '{agent_name}' غير موجود"}
        
        agent = self.agents[agent_name]
        return {
            "name": agent.AGENT_NAME,
            "role": agent.AGENT_ROLE,
            "prompt": agent.get_prompt()
        }
    
    def list_all_agents(self) -> List[Dict[str, str]]:
        """
        عرض قائمة بجميع الوكلاء في النظام
        
        Returns:
            قائمة بمعلومات جميع الوكلاء
        """
        agents_info = []
        for key, agent in self.agents.items():
            agents_info.append({
                "key": key,
                "name": agent.AGENT_NAME,
                "role": agent.AGENT_ROLE
            })
        return agents_info


def print_system_info():
    """طباعة معلومات النظام"""
    print("=" * 80)
    print("نظام دعم القرار القضائي للمحاكم المصرية")
    print("Egyptian Judicial Decision Support System")
    print("=" * 80)
    print()
    
    system = JudicialDecisionSupportSystem()
    agents = system.list_all_agents()
    
    print(f"عدد الوكلاء في النظام: {len(agents)}")
    print()
    
    for i, agent in enumerate(agents, 1):
        print(f"{i}. {agent['name']}")
        print(f"   Key: {agent['key']}")
        print(f"   Role: {agent['role']}")
        print()


if __name__ == "__main__":
    # عرض معلومات النظام
    print_system_info()
    
    # مثال على الاستخدام
    system = JudicialDecisionSupportSystem()
    
    print("=" * 80)
    print("مثال: الحصول على prompt وكيل المراجعة الإجرائية")
    print("=" * 80)
    
    auditor_info = system.get_agent_info("procedural_auditor")
    print(f"الاسم: {auditor_info['name']}")
    print(f"الدور: {auditor_info['role']}")
    print()
    print("يمكنك الآن استخدام auditor_info['prompt'] لإرساله إلى نموذج اللغة")

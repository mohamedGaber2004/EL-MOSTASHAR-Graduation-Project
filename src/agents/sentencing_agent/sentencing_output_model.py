from typing import Optional, List
from pydantic import BaseModel, Field
from src.agents.agents_enums import (
    CivilClaimStatus,
)


class CivilClaim(BaseModel):
    # ── بيانات المدّعي المدني ─────────────────────────────────────
    plaintiff_name:     Optional[str] = Field(description="اسم المدّعي المدني (المجني عليه أو ورثته)")
    plaintiff_capacity: Optional[str] = Field(default=None,description="صفة المدّعي: مجني عليه / وارث / ولي / وكيل")
 
    # ── طلبات التعويض ─────────────────────────────────────────────
    compensation_requested: Optional[float] = Field(default=None,description="مبلغ التعويض المطلوب بالجنيه المصري")
    compensation_basis:     Optional[str]   = Field(default=None,description="أساس طلب التعويض: أضرار مادية / معنوية / عدم كسب / نفقات علاج")
    material_damages:       Optional[float] = Field(default=None,description="قيمة الأضرار المادية الموثقة بالجنيه")
    moral_damages:          Optional[float] = Field(default=None,description="قيمة الأضرار الأدبية / المعنوية بالجنيه")
 
    # ── الوثائق الداعمة ───────────────────────────────────────────
    supporting_documents: List[str] = Field(default_factory=list,description="المستندات الداعمة للادعاء المدني (فواتير / تقارير طبية / ...)")
 
    # ── حالة الدعوى ───────────────────────────────────────────────
    status: CivilClaimStatus = Field(default=CivilClaimStatus.NOT_FILED,description="الحالة الراهنة للدعوى المدنية")
 
    # ── التقدير المقترح ───────────────────────────────────────────
    suggested_award:         Optional[float] = Field(default=None,description="مبلغ التعويض المقترح بناءً على تقدير وكيل التسوية")
    award_reasoning:         Optional[str] = Field(default=None,description="أسباب التقدير المقترح")
    award_against_defendant: Optional[str] = Field(default=None,description="اسم المتهم الملزَم بالتعويض إن تعدد المتهمون")
    solidarity_liability:    bool = Field(default=False,description="هل المسؤولية تضامنية بين المتهمين المتعددين؟")
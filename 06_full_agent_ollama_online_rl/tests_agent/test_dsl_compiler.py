from pcb_agent.config import LLMConfig
from pcb_agent.llm import DSLCompiler

def test_heuristic_dsl():
    dsl=DSLCompiler(LLMConfig(provider="mock")).compile("J1靠左，U1居中，去耦电容靠近U1，间距0.2mm")
    assert any(c.type=="edge" and c.side=="left" for c in dsl.hard_constraints)
    assert any(c.type=="near" and c.target=="U1" for c in dsl.soft_constraints)
    assert any(c.type=="clearance" and c.value_mm==0.2 for c in dsl.hard_constraints)

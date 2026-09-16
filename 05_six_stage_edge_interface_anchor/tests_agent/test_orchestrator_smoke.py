import json, shutil
from pathlib import Path
from pcb_agent.agent import AgentOrchestrator
from pcb_agent.config import AgentConfig

def test_mock_orchestrator(tmp_path):
    fixture=Path(__file__).parent/"fixtures";bundle=tmp_path/"bundle";bundle.mkdir()
    shutil.copy2(fixture/"tiny.kicad_pcb",bundle/"tiny.kicad_pcb");shutil.copy2(fixture/"tiny_task.json",bundle/"tiny_task.json")
    (bundle/"bundle.json").write_text(json.dumps({"project_dir":".","board_path":"tiny.kicad_pcb","task_json_path":"tiny_task.json","user_text":"U1居中，去耦电容靠近U1"}),encoding="utf-8")
    cfg=AgentConfig();cfg.placement.mock=True;cfg.router.mode="mock";cfg.kicad.mock_drc=True;cfg.llm.provider="mock"
    state=AgentOrchestrator(cfg).run(bundle,tmp_path/"run")
    assert state.status=="EXPORT_RESULT";assert (tmp_path/"run"/"final.kicad_pcb").exists();assert (tmp_path/"run"/"state.json").exists();assert (tmp_path/"run"/"report.md").exists()

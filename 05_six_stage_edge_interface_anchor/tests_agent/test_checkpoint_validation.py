from pathlib import Path
import pytest
from pcb_agent.config import PlacementConfig
from pcb_agent.tools import PlacementModelTool

def test_missing_checkpoint():
    with pytest.raises(FileNotFoundError):PlacementModelTool(PlacementConfig(checkpoint_path=Path("missing.pt"))).validate_checkpoint()

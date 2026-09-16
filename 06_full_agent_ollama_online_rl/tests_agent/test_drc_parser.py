from pcb_agent.tools.drc_kicad import KicadDRCTool

def test_parse_drc():
    report=KicadDRCTool.parse_report({"violations":[{"severity":"error","description":"Clearance violation","items":[{"ref":"U1"},{"net":"GND"}],"position":{"x":1,"y":2}}]})
    assert not report.drc_clean and report.error_count==1
    assert report.violations[0].category=="clearance" and report.violations[0].refs==["U1"]

def test_parse_clean():
    assert KicadDRCTool.parse_report({"violations":[]}).drc_clean

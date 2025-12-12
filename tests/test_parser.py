from src.parser import extract_target_from_instruction


def test_heuristic_simple():
    instr = "Please pick up the carrot on the table."
    t = extract_target_from_instruction(instr)
    assert t == "carrot"


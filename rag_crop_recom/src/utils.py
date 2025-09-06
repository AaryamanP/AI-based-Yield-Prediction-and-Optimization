import json
def parse_soil_input(soil_input: str):
    """
    Example input: "LOAMY:60, CLAY:40"
    Returns: {"LOAMY":60, "CLAY":40}
    """
    result = {}
    for pair in soil_input.split(","):
        if ":" in pair:
            k,v = pair.strip().split(":")
            result[k.strip().upper()] = float(v.strip().replace("%",""))
    return result
def safe_parse_json(output_text: str):
    try:
        return json.loads(output_text)
    except json.JSONDecodeError:
        # Fallback wrapping
        return {
            "best_action": output_text.strip(),
            "conservative_action": "No alternative recommendation generated."
        }

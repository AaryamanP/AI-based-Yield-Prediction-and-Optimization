import pandas as pd

# -----------------------------
# Crop safety caps (kg/ha for N)
# -----------------------------
SAFE_N_CAPS = {
    'RICE': 180, 'WHEAT': 150, 'MAIZE': 160, 'SORGHUM': 100,
    'KHARIF SORGHUM':100, 'RABI SORGHUM':100, 'PEARL MILLET': 80,
    'FINGER MILLET':80, 'BARLEY':120, 'CHICKPEA':50, 'PIGEONPEA':50,
    'MINOR PULSES':50, 'PULSES':50, 'GROUNDNUT':50, 'SESAMUM':50,
    'LINSEED':50, 'SUGARCANE':200, 'COTTON':100, 'FRUITS AND VEGETABLES':150,
    'FODDER':100, 'DEFAULT':150
}


SAFE_P_CAPS = {
    'RICE': 60, 'WHEAT': 50, 'MAIZE': 60, 'SORGHUM': 40,
    'KHARIF SORGHUM': 40, 'RABI SORGHUM': 40, 'PEARL MILLET': 30,
    'FINGER MILLET': 30, 'BARLEY': 40, 'CHICKPEA': 20, 'PIGEONPEA': 20,
    'MINOR PULSES': 20, 'PULSES': 20, 'GROUNDNUT': 40, 'SESAMUM': 20,
    'LINSEED': 20, 'SUGARCANE': 120, 'COTTON': 40, 'FRUITS AND VEGETABLES': 80,
    'FODDER': 60, 'DEFAULT': 50
}

SAFE_K_CAPS = {
    'RICE': 80, 'WHEAT': 60, 'MAIZE': 100, 'SORGHUM': 60,
    'KHARIF SORGHUM': 60, 'RABI SORGHUM': 60, 'PEARL MILLET': 40,
    'FINGER MILLET': 40, 'BARLEY': 50, 'CHICKPEA': 30, 'PIGEONPEA': 30,
    'MINOR PULSES': 30, 'PULSES': 30, 'GROUNDNUT': 80, 'SESAMUM': 40,
    'LINSEED': 40, 'SUGARCANE': 200, 'COTTON': 120, 'FRUITS AND VEGETABLES': 150,
    'FODDER': 120, 'DEFAULT': 100
}


def get_safe_N_cap(crop_name):
    return SAFE_N_CAPS.get(str(crop_name).strip().upper(), SAFE_N_CAPS['DEFAULT'])
def get_safe_K_cap(crop_name):
    return SAFE_K_CAPS.get(str(crop_name).strip().upper(), SAFE_K_CAPS['DEFAULT'])
def get_safe_P_cap(crop_name):
    return SAFE_P_CAPS.get(str(crop_name).strip().upper(), SAFE_P_CAPS['DEFAULT'])

# -----------------------------
# Irrigation effect mapping
# -----------------------------
def irrigation_effect_pct(irrigation_pct, dIrr):
    if dIrr <= 0:
        return 0.0
    if pd.isna(irrigation_pct):
        return 0.02
    if irrigation_pct < 0.05:
        return None
    if irrigation_pct >= 0.5:
        return 0.06
    if irrigation_pct >= 0.2:
        return 0.04
    return 0.02

# -----------------------------
# Optimization Model
# -----------------------------
def recommend_actions_realworld(predicted_yield, crop_name, area, irrigated_area, soil_dict,
                               N_kg, P_kg, K_kg):
    """
    predicted_yield: baseline predicted yield from teammates' ML model
    crop_name: str
    area, irrigated_area: float (ha)
    soil_dict: dict {soil_type: percentage} (optional)
    N_kg, P_kg, K_kg: fertilizer applied per ha (kg/ha)
    """
    irrigation_pct = irrigated_area / area if area > 0 else 0

    # Candidate deltas (realistic fertilizer & irrigation adjustments)
    dN_options = [-20, 0, 20, 40]
    dP_options = [-10, 0, 10]
    dK_options = [-10, 0, 10]
    dIrr_options = [-1, 0, 1]  # -1 = decrease, 0 = same, +1 = increase irrigation

    alpha = 0.5  # uncertainty weight
    candidates = []

    for dN in dN_options:
        for dP in dP_options:
            for dK in dK_options:
                for dIrr in dIrr_options:
                    N_new = N_kg + dN
                    if N_new > get_safe_N_cap(crop_name):
                        continue
                    irr_pct_effect = irrigation_effect_pct(irrigation_pct, dIrr)
                    if irr_pct_effect is None:
                        continue

                    # Realistic yield effect model
                    # Assumes slope factors (kg/ha yield per kg fertilizer) from agronomic research
                    slopeN, slopeP, slopeK = 0.02, 0.01, 0.005
                    y_mean = predicted_yield + slopeN*dN + slopeP*dP + slopeK*dK + predicted_yield*irr_pct_effect
                    y_std = max(5.0, 0.05*predicted_yield)  # 5% uncertainty

                    magnitude_penalty = 0.01*(abs(dN)+abs(dP)+abs(dK))
                    score = y_mean - alpha*y_std - magnitude_penalty

                    candidates.append({
                        'dN': dN, 'dP': dP, 'dK': dK, 'dIrr': dIrr,
                        'y_mean': y_mean, 'y_std': y_std, 'score': score
                    })

    if not candidates:
        return {"Predicted Yield": predicted_yield, "Best Action":"No feasible action", "Conservative Action":"No feasible action"}

    # Best action: max expected yield considering uncertainty
    best = max(candidates, key=lambda x: x['score'])

    # Conservative action: minimal fertilizer change with positive impact
    conservative = min(candidates, key=lambda x: abs(x['dN']) + abs(x['dP']) + abs(x['dK']))

    return {
        "Predicted Yield": round(predicted_yield,2),
        "Best Action": f"dN={best['dN']}, dP={best['dP']}, dK={best['dK']}, dIrr={best['dIrr']}",
        "Conservative Action": f"dN={conservative['dN']}, dP={conservative['dP']}, dK={conservative['dK']}, dIrr={conservative['dIrr']}",
        "Expected % gain": round(100*(best['y_mean']-predicted_yield)/predicted_yield,2)
    }

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    predicted_yield = 600  # kg/ha from teammates' model
    crop = "RICE"
    area = 100
    irrigated_area = 50
    soil = {"LOAMY_ALFISOLS":60,"USTALF":40}
    N_kg, P_kg, K_kg = 100, 20, 10

    reco = recommend_actions_realworld(predicted_yield, crop, area, irrigated_area, soil, N_kg, P_kg, K_kg)
    print(reco)

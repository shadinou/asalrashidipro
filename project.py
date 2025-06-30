import os
import re
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, griddata
from matplotlib.path import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PUMP_DATA_FOLDER = os.path.join(SCRIPT_DIR, 'pump_data')


def find_available_pumps(data_folder: str) -> list:

    if not os.path.isdir(data_folder):
        print(f"Error: Directory '{data_folder}' not found.")
        return []
    return [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]


def get_pump_boundary_polygon(pump_name: str, data_folder: str) -> Path | None:

    boundary_file = os.path.join(data_folder, pump_name, 'Pump_boundary.csv')
    if not os.path.exists(boundary_file):
        print(f"Warning: Boundary file 'Pump_boundary.csv' not found for pump '{pump_name}'.")
        return None
    try:
        df = pd.read_csv(boundary_file)
        if 'x' not in df.columns:
            print(f"Error: 'x' column not found in boundary file for '{pump_name}'.")
            return None
        
        curve_cols = [c for c in df.columns if c.lower() != 'x']
        if not curve_cols:
            print(f"Error: No curve columns found in boundary file for '{pump_name}'.")
            return None
            
        upper_h, lower_h = df[curve_cols].max(axis=1), df[curve_cols].min(axis=1)
        flow = df['x']
        
        upper_b, lower_b = list(zip(flow, upper_h)), list(zip(flow, lower_h))
        lower_b.reverse()
        
        vertices = upper_b + lower_b
        return Path(vertices) if len(vertices) >= 3 else None
    except Exception as e:
        print(f"Error processing boundary file for {pump_name}: {e}")
        return None


def get_performance_details(pump_name: str, required_flow: float, required_head: float) -> dict:

    results = {"diameter": None, "efficiency": None, "power": None}
    pump_folder_path = os.path.join(PUMP_DATA_FOLDER, pump_name)
    curves_file = os.path.join(pump_folder_path, 'Head_Efficiency.csv')
    power_file = os.path.join(pump_folder_path, 'Power.csv')

    if os.path.exists(curves_file):
        try:
            df_curves = pd.read_csv(curves_file)
            flow_col_name = next((col for col in df_curves.columns if col.strip().lower() in ['flow', 'x']), None)
            
            if flow_col_name:
                df_curves = df_curves.dropna(subset=[flow_col_name])
                # محاسبه قطر
                phi_cols = [col for col in df_curves.columns if col.lower().startswith('phi_')]
                if phi_cols:
                    # ... (منطق انتخاب قطر پروانه)
                    curves_above = []
                    for col_name in phi_cols:
                        curve_df = df_curves[[flow_col_name, col_name]].dropna().sort_values(by=flow_col_name)
                        if len(curve_df) < 2: continue
                        interp_func = interp1d(curve_df[flow_col_name], curve_df[col_name], bounds_error=False, fill_value="extrapolate")
                        head_on_curve = interp_func(required_flow)
                        if head_on_curve >= required_head:
                            distance = head_on_curve - required_head
                            diameter_val = int(re.search(r'\d+', col_name).group())
                            curves_above.append({'diameter': diameter_val, 'distance': distance})
                    if curves_above:
                        results['diameter'] = min(curves_above, key=lambda x: x['distance'])['diameter']
                
                # محاسبه بازده
                eff_cols = [col for col in df_curves.columns if col.lower().startswith('eff_')]
                if eff_cols:
                    # ... (منطق درون‌یابی بازده)
                    points, values = [], []
                    for col_name in eff_cols:
                        efficiency_val = float(re.search(r'(\d+\.?\d*)', col_name).group())
                        curve_df = df_curves[[flow_col_name, col_name]].dropna()
                        for _, row in curve_df.iterrows():
                            points.append([row[flow_col_name], row[col_name]])
                            values.append(efficiency_val)
                    if len(points) >= 4:
                        est_eff = griddata(points, values, (required_flow, required_head), method='cubic', fill_value=np.nan)
                        if np.isnan(est_eff): est_eff = griddata(points, values, (required_flow, required_head), method='linear', fill_value=np.nan)
                        if not np.isnan(est_eff): results['efficiency'] = round(float(est_eff), 2)

        except Exception as e:
            print(f"    - Warning: Could not process 'Head_Efficiency.csv'. Reason: {e}")

    # --- بخش ۳: محاسبه توان از Power.csv ---
    if results.get("diameter") and os.path.exists(power_file):
        try:
            df_power = pd.read_csv(power_file)
            power_flow_col = next((col for col in df_power.columns if col.strip().lower() in ['flow', 'x']), None)

            if power_flow_col:
                selected_diameter = int(results["diameter"])
                power_col_pattern = f"p_{selected_diameter}"
                target_power_col = next((col for col in df_power.columns if col.strip().lower() == power_col_pattern), None)

                if target_power_col:
                    power_curve_df = df_power[[power_flow_col, target_power_col]].dropna().sort_values(by=power_flow_col)
                    if len(power_curve_df) >= 2:
                        interp_func_power = interp1d(power_curve_df[power_flow_col], power_curve_df[target_power_col], bounds_error=False, fill_value="extrapolate")
                        est_power = interp_func_power(required_flow)
                        results["power"] = round(float(est_power), 2)
        except Exception as e:
            print(f"    - Warning: Could not calculate power. Reason: {e}")
            
    return results


def find_suitable_pumps(required_flow: float, required_head: float, data_folder: str):

    print("\nSearching for suitable pumps...")
    available_pumps = find_available_pumps(data_folder)
    final_results = []
    
    for pump_name in available_pumps:
        print(f"\n--- Evaluating pump: {pump_name} ---")
        pump_polygon = get_pump_boundary_polygon(pump_name, data_folder)
        
        if pump_polygon and pump_polygon.contains_point((required_flow, required_head)):
            print(f"  -> Result: Pump '{pump_name}' is SUITABLE.")
            
            details = get_performance_details(pump_name, required_flow, required_head)
            final_results.append({"pump": pump_name, **details})

            # نمایش نتایج جزئی
            dia_str = f"{details['diameter']} mm (Closest curve above)" if details.get("diameter") else "Not Available"
            eff_str = f"{details['efficiency']} %" if details.get("efficiency") else "Not Available"
            pow_str = f"{details['power']} W" if details.get("power") else "Not Available"
            
            print(f"  -> Selected Impeller Diameter: {dia_str}")
            print(f"  -> Estimated Efficiency: {eff_str}")
            print(f"  -> Estimated Power: {pow_str}")
        else:
            print(f"  -> Result: Operating point is outside the boundary of pump '{pump_name}'.")

    return final_results


if __name__ == "__main__":

    try:
        user_flow = float(input("Please enter the required flow (Q): "))
        user_head = float(input("Please enter the required head (H): "))
        
        suitable_pumps = find_suitable_pumps(user_flow, user_head, PUMP_DATA_FOLDER)

        print("\n\n" + "="*20 + " SUMMARY " + "="*20)
        if suitable_pumps:
            print(f"Found {len(suitable_pumps)} suitable pump(s) for the operating point (Q={user_flow}, H={user_head}):")
            for pump_info in suitable_pumps:
                dia_str = f"{pump_info.get('diameter', 'N/A')} mm"
                eff_str = f"{pump_info.get('efficiency', 'N/A')} %"
                pow_str = f"{pump_info.get('power', 'N/A')} W"
                print(f"- Pump: {pump_info['pump']:<15} | Selected Dia: {dia_str:<12} | Est. Efficiency: {eff_str:<12} | Est. Power: {pow_str}")
        else:
            print(f"Unfortunately, no suitable pump was found.")

    except ValueError:
        print("\nError: Please enter valid numerical values for flow and head.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
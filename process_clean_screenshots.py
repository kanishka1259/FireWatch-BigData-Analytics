import os
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BDA_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
SS_DIR = os.path.join(BDA_ROOT, "SS")
CLEAN_SS_DIR = os.path.join(BASE_DIR, "outputs", "clean_ss")

os.makedirs(CLEAN_SS_DIR, exist_ok=True)

# 1. Fig 1: Scope & PySpark Workflow (from 213350)
im1 = Image.open(os.path.join(SS_DIR, "Screenshot 2026-10-15 213350.png"))
c1 = im1.crop((0, 210, im1.width, 905))
c1.save(os.path.join(CLEAN_SS_DIR, "fig1_scope_pyspark.png"))

# 2. Fig 2: Runtime Benchmark Table (from 155228)
im2 = Image.open(os.path.join(SS_DIR, "Screenshot 2026-10-15 155228.png"))
c2 = im2.crop((0, 130, im2.width, 595))
c2.save(os.path.join(CLEAN_SS_DIR, "fig2_runtime_benchmark.png"))

# 3. Fig 3: PySpark GroupBy Code Walkthrough (from 213423)
im3 = Image.open(os.path.join(SS_DIR, "Screenshot 2026-10-15 213423.png"))
c3 = im3.crop((0, 510, im3.width, 885))
c3.save(os.path.join(CLEAN_SS_DIR, "fig3_pyspark_code.png"))

# 4. Fig 4: Peak Hotspot Dates Table (from 154902)
im4 = Image.open(os.path.join(SS_DIR, "Screenshot 2026-10-15 154902.png"))
c4 = im4.crop((0, 0, im4.width, 375))
c4.save(os.path.join(CLEAN_SS_DIR, "fig4_peak_dates.png"))

# 5. Fig 6: Monthly Seasonality & Gap (from 154937)
im6 = Image.open(os.path.join(SS_DIR, "Screenshot 2026-10-15 154937.png"))
c6 = im6.crop((0, 0, im6.width, 422))
c6.save(os.path.join(CLEAN_SS_DIR, "fig6_monthly_gap.png"))

# 6. Fig 8: Diurnal Cycle (from 155012)
im8 = Image.open(os.path.join(SS_DIR, "Screenshot 2026-10-15 155012.png"))
c8 = im8.crop((0, 0, im8.width, 242))
c8.save(os.path.join(CLEAN_SS_DIR, "fig8_diurnal_cycle.png"))

# 7. Fig 10: Day vs Night & WARD Rule (from 155025)
im10 = Image.open(os.path.join(SS_DIR, "Screenshot 2026-10-15 155025.png"))
c10 = im10.crop((0, 0, im10.width, 157))
c10.save(os.path.join(CLEAN_SS_DIR, "fig10_day_night.png"))

# 8. Fig 13: ML Performance Table (from 155043)
im13 = Image.open(os.path.join(SS_DIR, "Screenshot 2026-10-15 155043.png"))
c13 = im13.crop((0, 230, im13.width, 680))
c13.save(os.path.join(CLEAN_SS_DIR, "fig13_ml_benchmark.png"))

# 9. Fig 14: ML Holdout Execution Log (from 160612)
im14 = Image.open(os.path.join(SS_DIR, "Screenshot 2026-10-15 160612.png"))
c14 = im14.crop((0, 45, im14.width, 490))
c14.save(os.path.join(CLEAN_SS_DIR, "fig14_ml_holdout_log.png"))

# 10. Fig 17: Source Code & Tech Architecture Table (from 213423)
im17 = Image.open(os.path.join(SS_DIR, "Screenshot 2026-10-15 213423.png"))
c17 = im17.crop((0, 45, im17.width, 510))
c17.save(os.path.join(CLEAN_SS_DIR, "fig17_code_architecture_table.png"))

# 11. Fig 18: Live Pitch Script (from 163324)
im18 = Image.open(os.path.join(SS_DIR, "Screenshot 2026-10-15 163324.png"))
c18 = im18.crop((0, 0, im18.width, 498))
c18.save(os.path.join(CLEAN_SS_DIR, "fig18_viva_pitch.png"))

# 12. Fig 19: Temporal Leakage Code Walkthrough (from 213436)
im19 = Image.open(os.path.join(SS_DIR, "Screenshot 2026-10-15 213436.png"))
c19 = im19.crop((0, 160, im19.width, 660))
c19.save(os.path.join(CLEAN_SS_DIR, "fig19_leakage_code.png"))

# 13. Fig 20: Full Viva Cheat Sheet (from 213450)
im20 = Image.open(os.path.join(SS_DIR, "Screenshot 2026-10-15 213450.png"))
c20 = im20.crop((0, 25, im20.width, 680))
c20.save(os.path.join(CLEAN_SS_DIR, "fig20_viva_cheat_sheet.png"))

print("All screenshots successfully cropped to clean terminal figures in:", CLEAN_SS_DIR)

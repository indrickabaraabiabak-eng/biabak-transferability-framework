"""Write every step-3 table in full to the supplementary workbook (Tables S35 onward)."""
import pickle
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter
from biabak import config as C

OUT = C.OUT_DIR / "step3"
A = pickle.load(open(OUT / "step3a_tables.pkl", "rb"))
B = pickle.load(open(OUT / "step3b_tables.pkl", "rb"))
order = [
 (A["aoa_summary"].T.reset_index().rename(columns={"index": "quantity", 0: "value"}), "Area of applicability: feature space, threshold and coverage of the Centre Region"),
 (A["aoa_by_distance"], "Share of the prediction grid inside the area of applicability by distance to the nearest borehole"),
 (A["aoa_training_DI"], "Training dissimilarity index of every borehole under its NNDM fold"),
 (B["benchmark_design"], "Design of the synthetic detection benchmark"),
 (B["benchmark_power"], "Detection rate, median and 5-95% range of the estimate for every scenario, design, learner and prevalence"),
 (B["real_data_same_evaluators"], "Observed archive analysed with the benchmark learners and tests"),
 (B["compatibility_observed"], "Position of each observed statistic within the simulated distribution of every scenario"),
 (B["benchmark_all_replicates"], "Every replicate of every scenario"),
]
path = OUT / "Supplementary_Tables_Step3_Applicability_Benchmark.xlsx"
readme = []
with pd.ExcelWriter(path, engine="openpyxl") as xw:
    for n, (df, title) in enumerate(order, start=35):
        df = df.copy()
        for c in df.columns:
            if df[c].dtype == bool: df[c] = df[c].map({True: "yes", False: "no"})
            if str(df[c].dtype) == "category": df[c] = df[c].astype(str)
        sheet = f"S{n}"
        pd.DataFrame([[f"Table S{n}. {title}"]]).to_excel(xw, sheet_name=sheet, index=False, header=False)
        df.to_excel(xw, sheet_name=sheet, index=False, startrow=2)
        readme.append(dict(table=f"Table S{n}", sheet=sheet, title=title, rows=len(df), columns=len(df.columns)))
    pd.DataFrame(readme).to_excel(xw, sheet_name="README", index=False)
wb = load_workbook(path); wb.move_sheet("README", offset=-len(wb.sheetnames) + 1)
for ws in wb.worksheets:
    hdr = 1 if ws.title == "README" else 3
    for row in ws.iter_rows():
        for c in row:
            c.font = Font(name="Times New Roman", size=10)
            if isinstance(c.value, float):
                c.number_format = "0.000E+00" if (c.value != 0 and (abs(c.value) < 1e-3 or abs(c.value) >= 1e5)) else "0.0000"
    if hdr == 3: ws["A1"].font = Font(name="Times New Roman", size=11, bold=True)
    for c in ws[hdr]:
        c.font = Font(name="Times New Roman", size=10, bold=True); c.fill = PatternFill("solid", fgColor="DDDDDD")
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = ws.cell(row=hdr + 1, column=1)
    for i, col in enumerate(ws.iter_cols(min_row=hdr, max_row=min(ws.max_row, hdr + 60)), start=1):
        L = max(len(str(c.value)) if c.value is not None else 0 for c in col)
        ws.column_dimensions[get_column_letter(i)].width = min(max(10, L + 2), 50)
wb.save(path)
print(path, [(r["sheet"], r["rows"]) for r in readme])

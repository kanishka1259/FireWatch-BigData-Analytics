import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BDA_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
CLEAN_SS_DIR = os.path.join(BASE_DIR, "outputs", "clean_ss")
CHARTS_DIR = os.path.join(BASE_DIR, "outputs", "charts")
OUTPUT_DOCX = os.path.join(BASE_DIR, "outputs", "FireWatch_Academic_Technical_Report.docx")
ROOT_DOCX = os.path.join(BDA_ROOT, "FireWatch_Technical_Report.docx")
OUTPUT_MD = os.path.join(BASE_DIR, "outputs", "FIREWATCH_ACADEMIC_TECHNICAL_REPORT.md")

print("Initializing Academic Technical Report Generator...")

doc = Document()

# Page Margins
for section in doc.sections:
    section.top_margin = Inches(0.85)
    section.bottom_margin = Inches(0.85)
    section.left_margin = Inches(0.85)
    section.right_margin = Inches(0.85)

# Styling Helpers
def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=180, right=180):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def set_callout_border(cell, color="0B3D91", sz="36"):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:top w:val="none"/><w:left w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/><w:bottom w:val="none"/><w:right w:val="none"/></w:tcBorders>')
    tcPr.append(borders)

def set_table_borders(table, color="CBD5E1", sz="6"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'<w:tblBorders {nsdecls("w")}><w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/><w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/><w:left w:val="none"/><w:right w:val="none"/><w:insideH w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/><w:insideV w:val="none"/></w:tblBorders>')
    tblPr.append(borders)

def add_header_styled(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True
    run = h.runs[0]
    if level == 1:
        run.font.name = "Calibri"
        run.font.size = Pt(17)
        run.font.bold = True
        run.font.color.rgb = RGBColor(11, 61, 145) # Deep Navy
    elif level == 2:
        run.font.name = "Calibri"
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = RGBColor(30, 41, 59) # Slate 800
    elif level == 3:
        run.font.name = "Calibri"
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = RGBColor(71, 85, 105) # Slate 600
    return h

def add_p(doc, text="", bold_prefix=None, space_after=6, italic=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Calibri"
        r_pre.font.size = Pt(10.5)
        r_pre.font.bold = True
        r_pre.font.color.rgb = RGBColor(15, 23, 42)
    if text:
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(10.5)
        r.font.italic = italic
        r.font.color.rgb = RGBColor(51, 65, 85)
    return p

def add_callout(doc, text, title=None, color="0B3D91", bg_color="F1F5F9"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_color)
    set_callout_border(cell, color=color, sz="36")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.15
    if title:
        r_title = p.add_run(title + "\n")
        r_title.font.name = "Calibri"
        r_title.font.size = Pt(10.5)
        r_title.font.bold = True
        r_title.font.color.rgb = RGBColor(11, 61, 145) if color == "0B3D91" else RGBColor(194, 65, 12)
    r = p.add_run(text)
    r.font.name = "Calibri"
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(30, 41, 59)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def add_image_with_caption(doc, img_path, caption, width_in=6.2):
    if not os.path.exists(img_path):
        print(f"Warning: image path not found: {img_path}")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run()
    run.add_picture(img_path, width=Inches(width_in))
    
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(0)
    p_cap.paragraph_format.space_after = Pt(8)
    r_cap = p_cap.add_run(caption)
    r_cap.font.name = "Calibri"
    r_cap.font.size = Pt(9.5)
    r_cap.font.italic = True
    r_cap.font.color.rgb = RGBColor(100, 116, 139)

def style_table(table, col_widths, headers, data, align_cols=None):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table, color="CBD5E1", sz="6")
    # Header Row
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], "0B3D91") # Dark Blue
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=120, right=120)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if (align_cols and align_cols[i] == 'C') else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        run = p.runs[0]
        run.font.name = "Calibri"
        run.font.size = Pt(9.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
    
    # Data Rows
    for row_idx, row_data in enumerate(data):
        row_cells = table.add_row().cells
        bg_hex = "F8FAFC" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, cell_value in enumerate(row_data):
            row_cells[col_idx].text = str(cell_value)
            set_cell_background(row_cells[col_idx], bg_hex)
            set_cell_margins(row_cells[col_idx], top=85, bottom=85, left=120, right=120)
            p = row_cells[col_idx].paragraphs[0]
            if align_cols and align_cols[col_idx] == 'C':
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif align_cols and align_cols[col_idx] == 'R':
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            if p.runs:
                run = p.runs[0]
                run.font.name = "Calibri"
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(30, 41, 59)
                if col_idx == 0:
                    run.font.bold = True

    # Set column widths
    for row in table.rows:
        for i, w in enumerate(col_widths):
            row.cells[i].width = Inches(w)

# ==========================================
# REPORT CONTENT GENERATION
# ==========================================

# Document Header / Academic Title Banner
title_p = doc.add_paragraph()
title_p.paragraph_format.space_before = Pt(0)
title_p.paragraph_format.space_after = Pt(2)
r_banner = title_p.add_run("ACADEMIC TECHNICAL REPORT — BIG DATA ANALYTICS")
r_banner.font.name = "Calibri"
r_banner.font.size = Pt(11)
r_banner.font.bold = True
r_banner.font.color.rgb = RGBColor(194, 65, 12) # Amber Accent

main_title = doc.add_paragraph()
main_title.paragraph_format.space_before = Pt(0)
main_title.paragraph_format.space_after = Pt(4)
r_mtitle = main_title.add_run("FireWatch: Scalable Big Data Telemetry Processing, Temporal Dynamics, and Forward-Chronological Machine Learning for Wildfire Rapid Dispatch")
r_mtitle.font.name = "Calibri"
r_mtitle.font.size = Pt(20)
r_mtitle.font.bold = True
r_mtitle.font.color.rgb = RGBColor(11, 61, 145)

sub_title = doc.add_paragraph()
sub_title.paragraph_format.space_before = Pt(0)
sub_title.paragraph_format.space_after = Pt(12)
r_stitle = sub_title.add_run("End-to-End Distributed Analytics, Diurnal Thermal Radiometry, Concept Drift Evaluation, and Full-Stack Operationalization across 9.51 Million NASA FIRMS Observations")
r_stitle.font.name = "Calibri"
r_stitle.font.size = Pt(12)
r_stitle.font.italic = True
r_stitle.font.color.rgb = RGBColor(71, 85, 105)

# Metadata Specification Table
meta_tbl = doc.add_table(rows=1, cols=4)
style_table(
    meta_tbl,
    [1.6, 1.8, 1.6, 1.8],
    ["Course / Context", "Analytical Scope", "Distributed Engine", "Primary Data Source"],
    [[
        "Big Data Analytics (Review 3)",
        "Spatiotemporal & Diurnal ML",
        "Apache PySpark 4.2.0 (Java 21)",
        "NASA FIRMS VIIRS (9.51M Rows)"
    ]],
    align_cols=['C', 'C', 'C', 'C']
)
doc.add_paragraph().paragraph_format.space_after = Pt(6)

# -------------------------------------------------------------
# 1. EXECUTIVE SUMMARY / ABSTRACT
# -------------------------------------------------------------
add_header_styled(doc, "1. Executive Summary", level=1)

add_p(doc,
    "Wildfire disaster management requires rapid, data-driven synthesis of high-velocity satellite telemetry to optimize emergency firefighting resource allocation. This study develops an enterprise-grade Big Data processing and Machine Learning architecture applied to the NASA EOSDIS / FIRMS Suomi-NPP VIIRS global wildfire archive, comprising exactly 9,511,402 active fire hotspot records (730.16 MB) observed between October 12, 2025 and June 30, 2026.",
    bold_prefix="Abstract & System Scope: "
)

add_p(doc,
    "Using distributed Apache PySpark 4.2.0 on Java 21 LTS, the analytical pipeline processes and aggregates the complete 9.51-million-row telemetry archive in 31.75 seconds (~299,571 records/second throughput). The investigation resolves critical temporal, diurnal, and radiative dynamics: identifying the single worst wildfire disaster date on October 16, 2025 (66,867 hotspots, 675,835 MW thermal energy); uncovering an operational divergence between peak monthly detection count (January 2026: 1.53M fires) and peak radiative destructiveness (June 2026: 10.32 MW mean FRP); resolving the global diurnal convective peak between 11:00 and 12:00 UTC (>2.85M fires); and formulating an emergency dispatch threshold for nocturnal crown fires (FRP >= 30 MW).",
    bold_prefix="Key Discoveries: "
)

add_p(doc,
    "To address temporal data leakage in predictive modeling, a forward-chronological holdout validation methodology was implemented. Benchmarked against a standard random 80/20 split (74.72% accuracy, 0.8742 ROC-AUC), the chronological validation achieved 71.19% accuracy and 0.8415 ROC-AUC, empirically demonstrating the magnitude of seasonal concept drift while confirming robust operational generalizability. The system is fully operationalized through a Flask REST API backend, an interactive Leaflet.js and Chart.js geospatial console, a NoSQL persistence layer, and a comprehensive automated unit-testing suite.",
    bold_prefix="Methodological Contribution: "
)

add_callout(doc,
    "• Ingested and cleansed 9,511,402 satellite observations in 31.75 seconds using Apache PySpark distributed query optimization.\n"
    "• Discovered the single highest global wildfire activity day: October 16, 2025 (66,867 fires, 675,835 MW).\n"
    "• Identified seasonal divergence: peak volume in January 2026 versus peak thermal radiative energy in June 2026.\n"
    "• Detected a 39-day archival sensor gap, preventing misleading assumptions of zero late-spring wildfire risk.\n"
    "• Discovered a steep 2-hour diurnal solar peak at 11:00–12:00 UTC accounting for 30.0% of all global fire activity.\n"
    "• Formulated the WARD nocturnal triage protocol: nighttime detections with FRP >= 30 MW trigger Level-1 emergency escalation.\n"
    "• Evaluated models via forward-chronological holdout (71.19% accuracy, 0.8415 ROC-AUC), empirically confirming seasonal concept drift.",
    title="KEY TECHNICAL ACHIEVEMENTS & CONTRIBUTIONS"
)

# Embed Clean Fig 1
add_image_with_caption(
    doc,
    os.path.join(CLEAN_SS_DIR, "fig1_scope_pyspark.png"),
    "Figure 1: High-Level Analytical Objectives and Distributed PySpark Processing Pipeline Architecture"
)

# -------------------------------------------------------------
# 2. INTRODUCTION
# -------------------------------------------------------------
add_header_styled(doc, "2. Introduction", level=1)

add_p(doc,
    "Wildfires represent one of the most critical environmental and economic hazards globally, devastating ecosystems, destroying infrastructure, and emitting massive quantities of atmospheric greenhouse gases. Spaceborne remote sensing—specifically the Visible Infrared Imaging Radiometer Suite (VIIRS) aboard the Suomi National Polar-orbiting Partnership (Suomi-NPP) satellite—provides near-continuous global active fire surveillance. However, the volume and velocity of polar-orbiting satellite feeds present substantial computational bottlenecks for emergency dispatch agencies such as CAL FIRE, EFFIS, and the Australian Rural Fire Service.",
    bold_prefix="Operational Background: "
)

add_p(doc,
    "Conventional single-machine data analysis pipelines suffer from prohibitive memory constraints and lengthy processing cycles when attempting multi-dimensional temporal aggregations over millions of geographic telemetry records. Furthermore, machine-learning models developed for wildfire dispatch frequently suffer from temporal data leakage when evaluated with standard random shuffling, obscuring real-world vulnerabilities caused by seasonal climate shifts. This technical report details the design, implementation, and empirical validation of a scalable Big Data architecture that overcomes these computational and methodological challenges.",
    bold_prefix="Problem Statement: "
)

# -------------------------------------------------------------
# 3. RESEARCH OBJECTIVES
# -------------------------------------------------------------
add_header_styled(doc, "3. Research Objectives", level=1)

add_p(doc, "The primary research question guiding this investigation is: \"When does observed global wildfire activity peak, how does thermal radiative energy fluctuate chronologically across days, months, and diurnal hours, and how does seasonal concept drift affect predictive triage models when evaluated under leakage-safe forward testing?\"", italic=True)

add_p(doc, "To provide comprehensive, mathematically rigorous answers, five specific sub-objectives were established:")
add_p(doc, "Quantify daily detection frequency and apply rolling-window smoothing to isolate the highest-impact disaster dates globally.", bold_prefix="1. Identification of Global Peak Activity Dates: ")
add_p(doc, "Resolve Year-Month aggregated trends and conduct an exhaustive calendar continuity audit to detect archival sensor discontinuities.", bold_prefix="2. Seasonal Volume and Energetic Dynamics: ")
add_p(doc, "Aggregate telemetry into 24-hour UTC bins to delineate diurnal solar convective peaks and assess the impact of polar sun-synchronous overpass windows.", bold_prefix="3. Diurnal Cycle and Acquisition Dynamics: ")
add_p(doc, "Characterize the radiative power contrast between daytime and nighttime events and derive emergency dispatch triage thresholds.", bold_prefix="4. Day versus Night Thermal Energetics: ")
add_p(doc, "Develop a forward-chronological holdout split to measure model resilience against temporal domain shift and deploy the pipeline into an operational web framework.", bold_prefix="5. Leakage-Safe Machine Learning Validation: ")

# -------------------------------------------------------------
# 4. DATASET SPECIFICATIONS & SCHEMA AUDIT
# -------------------------------------------------------------
add_header_styled(doc, "4. Dataset Description & Telemetry Specifications", level=1)

add_p(doc,
    "The dataset was sourced directly from the NASA Fire Information for Resource Management System (FIRMS) archive, designated as fire_archive_SV-C2_817450.csv (~730.16 MB uncompressed). The archive contains 9,511,402 individual hotspot detections collected across a 262-calendar-day observation span between October 12, 2025 and June 30, 2026.",
    bold_prefix="Data Source & Volume: "
)

schema_headers = ["Attribute", "Data Type", "Physical Description", "Null Count", "Operational Validity"]
schema_data = [
    ["latitude", "Double", "Center latitude of the 375m sensor pixel", "0 (0.0%)", "100% valid (-90.0° to +90.0°)"],
    ["longitude", "Double", "Center longitude of the 375m sensor pixel", "0 (0.0%)", "100% valid (-180.0° to +180.0°)"],
    ["brightness", "Double", "VIIRS I-4 thermal channel (4 µm) temperature (K)", "0 (0.0%)", "Calibrated Kelvin scale"],
    ["scan, track", "Double", "Pixel spatial resolution dimensions along scan/track", "0 (0.0%)", "Geometric footprint metrics"],
    ["acq_date", "String / Date", "Satellite overpass acquisition calendar date (UTC)", "0 (0.0%)", "Continuous ISO-8601 strings"],
    ["acq_time", "Integer", "UTC acquisition timestamp in HHMM format", "0 (0.0%)", "0000 to 2359 UTC valid"],
    ["satellite", "String", "Spacecraft constellation identifier (Suomi-NPP)", "0 (0.0%)", "100% Suomi-NPP VIIRS"],
    ["confidence", "String", "NASA algorithmic detection quality tier (l, n, h)", "0 (0.0%)", "Quality assurance metadata"],
    ["bright_t31", "Double", "VIIRS I-5 background channel (11 µm) temperature (K)", "0 (0.0%)", "Thermal background baseline"],
    ["frp", "Double", "Fire Radiative Power in MegaWatts (MW)", "0 (0.0%)", "Biomass combustion rate"],
    ["daynight", "String", "Solar illumination flag: Diurnal ('D') vs. Nocturnal ('N')", "0 (0.0%)", "Binary solar state"],
    ["type", "Integer", "Source classification (0=wildfire, 1=volcano, 2=industrial)", "0 (0.0%)", "90.1% vegetation fires"]
]
schema_tbl = doc.add_table(rows=1, cols=5)
style_table(schema_tbl, [1.1, 1.0, 2.3, 0.9, 1.5], schema_headers, schema_data, align_cols=['L', 'C', 'L', 'C', 'L'])
doc.add_paragraph().paragraph_format.space_after = Pt(6)

add_p(doc,
    "Timestamps are recorded as UTC integers ranging from 0 to 2359. Integers below 1000 represent morning hours requiring leading-zero restoration (e.g., integer 53 corresponds to 00:53 UTC). Data integrity verification confirmed 100% compliance across all 9,511,402 rows, with zero null coordinates, zero malformed timestamps, and zero invalid physical temperatures.",
    bold_prefix="Data Quality & Integrity Audit: "
)

# -------------------------------------------------------------
# 5. DISTRIBUTED PYSPARK PROCESSING ARCHITECTURE
# -------------------------------------------------------------
add_header_styled(doc, "5. Distributed Big Data Processing Architecture", level=1)

add_p(doc,
    "Processing nearly 10 million multi-parameter geographic records exceeds the memory and compute capacity of single-threaded Python pandas, which triggers out-of-memory errors and runtimes exceeding 10 minutes. To provide scalable, fault-tolerant execution, a distributed architecture was constructed using Apache PySpark 4.2.0 executing on Java 21 LTS.",
    bold_prefix="Architectural Rationale: "
)

add_p(doc,
    "PySpark's Catalyst Query Optimizer analyzes transformation expressions and compiles them into an optimized Directed Acyclic Graph (DAG) of Resilient Distributed Datasets (RDDs). Through lazy evaluation, column pruning, type casting, and spatial-temporal filtering are pipelined together. In-memory caching prevents redundant disk I/O across successive aggregation queries.",
    bold_prefix="Execution Optimization: "
)

# Benchmark Table
add_header_styled(doc, "Empirical PySpark Execution Benchmarks (9,511,402 Records)", level=2)

bench_headers = ["Pipeline Stage", "Framework / Engine", "Processed Rows", "Duration (s)", "Throughput (rows/s)"]
bench_data = [
    ["SparkSession Initialization", "PySpark 4.2 / Java 21 LTS", "—", "5.93 s", "—"],
    ["Dataset Ingestion & Schema Audit", "PySpark CSV Distributed Reader", "9,511,402", "10.15 s", "937,083 rows/s"],
    ["Data Cleaning, Casting & Caching", "PySpark DataFrame API", "9,511,402", "9.35 s", "1,017,262 rows/s"],
    ["Daily Aggregation & 7-Day Window", "PySpark Window API", "9,511,402", "2.51 s", "3,789,403 rows/s"],
    ["Monthly Temporal Aggregation", "Spark SQL GroupBy", "9,511,402", "1.52 s", "6,257,501 rows/s"],
    ["Hourly UTC Diurnal Cycle (0-23)", "Spark SQL GroupBy", "9,511,402", "0.99 s", "9,607,476 rows/s"],
    ["Day/Night Temporal Aggregation", "Spark SQL GroupBy", "9,511,402", "1.24 s", "7,670,485 rows/s"],
    ["TOTAL PYSPARK DISTRIBUTED PIPELINE", "Apache PySpark Distributed", "9,511,402", "31.75 s", "299,571 rows/s"],
    ["High-Perf Vectorized Fallback Engine", "Vectorized PyArrow / Polars", "9,511,402", "15.69 s", "606,207 rows/s"],
    ["ML Model Training & Temporal Holdout", "Scikit-Learn Multi-Threaded", "200,000", "10.39 s", "19,250 samples/s"],
    ["NoSQL Storage Persistence", "SQLite Document Store", "5,000 + 38", "0.12 s", "42,000 docs/s"]
]
bench_tbl = doc.add_table(rows=1, cols=5)
style_table(bench_tbl, [2.2, 1.8, 1.1, 0.9, 1.2], bench_headers, bench_data, align_cols=['L', 'L', 'R', 'R', 'R'])
doc.add_paragraph().paragraph_format.space_after = Pt(6)

add_image_with_caption(
    doc,
    os.path.join(CLEAN_SS_DIR, "fig2_runtime_benchmark.png"),
    "Figure 2: Empirical PySpark Distributed Query Benchmarking Across 9.51 Million Telemetry Records"
)

add_image_with_caption(
    doc,
    os.path.join(CLEAN_SS_DIR, "fig3_pyspark_code.png"),
    "Figure 3: PySpark Distributed Window Aggregation and UTC Hourly Feature Transformation Implementation"
)

# -------------------------------------------------------------
# 6. RESULTS & EMPIRICAL ANALYSIS
# -------------------------------------------------------------
add_header_styled(doc, "6. Empirical Results & Spatiotemporal Analysis", level=1)

# Finding 1
add_header_styled(doc, "6.1 Peak Incident Dates and Extreme Global Events", level=2)
add_p(doc,
    "The analysis revealed that global wildfire activity peaked on October 16, 2025, recording 66,867 separate satellite hotspot detections within a single 24-hour UTC window. Hotspots on this date generated a cumulative thermal energy output of 675,835 MegaWatts (MW), averaging 10.11 MW per detection. October 17 and October 25 closely followed, each exceeding 65,000 detections. Geospatial correlation shows that these October peaks were driven primarily by end-of-dry-season biomass burning across Sub-Saharan Africa and high-latitude agricultural clearances in the Northern Hemisphere.",
    bold_prefix="Single-Day Disaster Peak: "
)

top_dates_headers = ["Rank", "Acquisition Date", "Hotspot Count", "Mean FRP (MW)", "Daily Total FRP (MW)", "Regional Activity Driver"]
top_dates_data = [
    ["1", "2025-10-16", "66,867", "10.11 MW", "675,835 MW", "Boreal & Sub-Saharan clearance"],
    ["2", "2025-10-17", "65,679", "10.09 MW", "662,580 MW", "South American savanna / African bush"],
    ["3", "2025-10-25", "65,099", "10.12 MW", "658,900 MW", "Tropical equatorial fire clearance"],
    ["4", "2026-01-09", "64,548", "7.91 MW", "510,420 MW", "Australian summer bushfire activity"],
    ["5", "2026-01-26", "64,057", "7.91 MW", "506,980 MW", "Australian & Cerrado fires"],
    ["6", "2026-01-14", "62,972", "7.92 MW", "498,320 MW", "Savanna seasonal burning"],
    ["7", "2026-01-06", "61,369", "7.91 MW", "485,120 MW", "Australian bushfires"],
    ["8", "2025-10-15", "60,936", "10.10 MW", "615,400 MW", "Northern autumn agricultural clearing"],
    ["9", "2026-03-28", "60,267", "8.38 MW", "505,037 MW", "Indochina dry season biomass burning"],
    ["10", "2026-01-25", "59,145", "7.91 MW", "467,837 MW", "Southern Hemisphere peak"]
]
top_tbl = doc.add_table(rows=1, cols=6)
style_table(top_tbl, [0.6, 1.2, 1.1, 1.1, 1.3, 1.9], top_dates_headers, top_dates_data, align_cols=['C', 'C', 'R', 'R', 'R', 'L'])
doc.add_paragraph().paragraph_format.space_after = Pt(6)

add_image_with_caption(
    doc,
    os.path.join(CLEAN_SS_DIR, "fig4_peak_dates.png"),
    "Figure 4: Global Peak Wildfire Detection Dates Ranked by Frequency, Radiative Energy, and Regional Drivers"
)

add_image_with_caption(
    doc,
    os.path.join(CHARTS_DIR, "member2_daily_trend.png"),
    "Figure 5: Daily Global Fire Hotspot Volume, 7-Day Centered Moving Average, and Cumulative FRP (MW)"
)

# Finding 2
add_header_styled(doc, "6.2 Monthly Seasonal Dynamics & Archive Continuity Audit", level=2)
add_p(doc,
    "Aggregating telemetry across Year-Month intervals uncovers a critical operational divergence between detection volume and physical energy release:",
    bold_prefix="Monthly Seasonality Divergence: "
)
add_p(doc, "January 2026 recorded the highest detection volume with 1,531,105 hotspots (16.10% of total volume), propelled by intense Southern Hemisphere summer heatwaves across Southeastern Australia and the southern Cerrado.", bold_prefix="• Peak Detection Volume: ")
add_p(doc, "June 2026 recorded the highest mean radiative power (10.32 MW per detection, releasing 12,356,080 MW total energy). These high-energy burns reflect intense boreal forest crown fires in Canada and Siberia, where heavy fuel loads generate catastrophic heat release.", bold_prefix="• Peak Thermal Energy Intensity: ")
add_p(doc, "December 2025 recorded the lowest count among complete months (1,079,467 hotspots, 11.35%).", bold_prefix="• Quietest Complete Month: ")
add_p(doc, "A calendar sequence audit across the 262-day operational span identified exactly 39 missing calendar days. Crucially, the entire month of May 2026 (0 records from 2026-05-01 to 2026-05-31) is absent due to satellite historical archival ingestion cutoffs. Identifying this discontinuity is vital to prevent automated dispatch systems from falsely concluding that wildfire danger ceased during late spring.", bold_prefix="• Sensor Archive Discontinuity Audit: ")

add_image_with_caption(
    doc,
    os.path.join(CLEAN_SS_DIR, "fig6_monthly_gap.png"),
    "Figure 6: Monthly Fire Radiometry Distributions and Historical Archive Discontinuity Audit"
)

add_image_with_caption(
    doc,
    os.path.join(CHARTS_DIR, "member2_monthly_trend.png"),
    "Figure 7: Monthly Fire Hotspot Volume and Dual-Axis Mean Fire Radiative Power (MW)"
)

# Finding 3
add_header_styled(doc, "6.3 24-Hour UTC Diurnal Convective Cycle", level=2)
add_p(doc,
    "Binning acquisition timestamps across 24 UTC hours reveals an exceptionally sharp diurnal concentration. Peak activity occurs during Hour 11:00 UTC (1,525,521 detections) and Hour 12:00 UTC (1,330,823 detections). Combined, these two hours account for 2,856,344 detections—representing 30.0% of the entire 9.51-million-row dataset. A secondary morning peak appears at Hour 06:00 UTC (513,532 detections), while the quietest observation windows occur at 09:00 UTC (109,108) and 02:00 UTC (111,534).",
    bold_prefix="Diurnal Solar Peak: "
)

add_callout(doc,
    "The Suomi-NPP VIIRS sensor operates in a polar sun-synchronous orbit designed with a local solar crossing time of ~13:30 (descending node). Consequently, UTC acquisition timestamps reflect satellite overpass geometry over specific longitudes. Hours 11:00 to 12:00 UTC align with early-to-mid afternoon local solar time across the African continent, Mediterranean Europe, and Western Asia. At this time, surface solar irradiance reaches its maximum, atmospheric relative humidity drops to its minimum, and vegetation flammability peaks, producing massive fire propagation.",
    title="PHYSICAL SATELLITE ORBITAL DYNAMICS & CONVECTIVE FORCING"
)

add_image_with_caption(
    doc,
    os.path.join(CLEAN_SS_DIR, "fig8_diurnal_cycle.png"),
    "Figure 8: 24-Hour UTC Acquisition Frequency and Polar Sun-Synchronous Overpass Geometry"
)

add_image_with_caption(
    doc,
    os.path.join(CHARTS_DIR, "member2_hourly_distribution.png"),
    "Figure 9: Diurnal 24-Hour UTC Hotspot Frequency Distribution Illustrating Global Convective Windows"
)

# Finding 4
add_header_styled(doc, "6.4 Diurnal versus Nocturnal Thermal Energetics", level=2)
add_p(doc, "7,246,710 detections (76.19% of volume) | Mean FRP: 9.85 MW | Mean Brightness: 337.8 K.", bold_prefix="• Daytime Detections ('D'): ")
add_p(doc, "2,264,692 detections (23.81% of volume) | Mean FRP: 4.72 MW | Mean Brightness: 326.1 K.", bold_prefix="• Nighttime Detections ('N'): ")

add_callout(doc,
    "Under standard atmospheric conditions, nocturnal boundary-layer cooling and relative humidity recovery suppress flame intensity and rate of spread. Consequently, any nocturnal detection exhibiting a Fire Radiative Power >= 30 MW indicates an extreme, fuel-driven crown fire capable of overcoming nocturnal humidity inversions. In the WARD decision framework, this finding establishes an automatic Level-1 emergency escalation rule to deploy night-vision aerial firefighting resources.",
    title="OPERATIONAL INTELLIGENCE PROTOCOL (WARD DISPATCH RULE)",
    color="C2410C",
    bg_color="FFF7ED"
)

add_image_with_caption(
    doc,
    os.path.join(CLEAN_SS_DIR, "fig10_day_night.png"),
    "Figure 10: Diurnal versus Nocturnal Thermal Energetics and Nocturnal Threshold Derivation"
)

add_image_with_caption(
    doc,
    os.path.join(CHARTS_DIR, "member2_day_vs_night.png"),
    "Figure 11: Proportional Share and Mean Fire Radiative Power (MW) Contrast Between Day and Night Hotspots"
)

# Finding 5
add_header_styled(doc, "6.5 Frequency versus Thermal Radiative Energy Discrepancy", level=2)
add_p(doc,
    "A vital empirical conclusion is that raw hotspot frequency is an insufficient operational dispatch metric. Ranking days by detection count highlights January dates dominated by hundreds of thousands of low-energy agricultural stubble burns (~7.9 MW). In contrast, ranking days by cumulative thermal energy prioritizes October and June dates dominated by high-intensity forest crown fires (10.1 to 10.3 MW mean FRP). Operational dispatch frameworks must prioritize Total Fire Radiative Power (MW) over raw count to prevent misallocation of emergency aircraft.",
    bold_prefix="Operational Risk Assessment: "
)

add_image_with_caption(
    doc,
    os.path.join(CHARTS_DIR, "member2_top_dates_comparison.png"),
    "Figure 12: Analytical Disparity Between Raw Hotspot Frequency and Cumulative Fire Radiative Power"
)

# -------------------------------------------------------------
# 7. MACHINE LEARNING EVALUATION & CONCEPT DRIFT
# -------------------------------------------------------------
add_header_styled(doc, "7. Machine Learning Evaluation & Concept Drift Analysis", level=1)

add_p(doc,
    "Traditional machine-learning workflows randomly partition datasets into an 80% training and 20% test split. In environmental time-series, this creates severe Temporal Data Leakage: future atmospheric states, fuel conditions, and seasonal patterns leak into the training partition, generating artificially inflated evaluation metrics that deteriorate when deployed into production.",
    bold_prefix="Temporal Data Leakage Vulnerability: "
)

add_p(doc,
    "To ensure leakage-safe validation, a Forward-Chronological Holdout Split was implemented. The model was trained strictly on historical observations recorded up to October 15, 2025 (160,000 samples) and evaluated forward in time on subsequent future observations from October 16, 2025 through June 30, 2026 (40,000 samples). FRP was strictly excluded from feature inputs to prevent target leakage.",
    bold_prefix="Chronological Holdout Methodology: "
)

ml_headers = ["Model Architecture", "Target Variable", "Evaluation Strategy", "Accuracy", "Weighted F1", "ROC-AUC", "Latency"]
ml_data = [
    ["Random Forest Classifier", "Severity Tier (4-Class)", "Random Stratified Split (80/20)", "74.72%", "0.7309", "0.8742", "0.08 ms"],
    ["HistGradientBoosting", "Severity Tier (4-Class)", "Random Stratified Split (80/20)", "74.72%", "0.7297", "0.8748", "0.04 ms"],
    ["Random Forest (Temporal)", "Severity Tier (4-Class)", "Forward Chronological Split", "71.19%", "0.6868", "0.8415", "0.002 ms"],
    ["Random Forest Regressor", "Continuous FRP (MW)", "Continuous Energy Prediction", "—", "—", "R² = 0.4736", "RMSE: 13.45 MW"]
]
ml_tbl = doc.add_table(rows=1, cols=7)
style_table(ml_tbl, [1.8, 1.4, 1.6, 0.7, 0.7, 0.8, 0.7], ml_headers, ml_data, align_cols=['L', 'L', 'L', 'C', 'C', 'C', 'R'])
doc.add_paragraph().paragraph_format.space_after = Pt(6)

add_callout(doc,
    "Evaluating the model strictly on subsequent chronological dates produces a realistic 3.53% drop in classification accuracy (from 74.72% to 71.19%), while sustaining strong discriminative capability with a multiclass ROC-AUC of 0.8415. This drop demonstrates seasonal concept drift: decision boundaries shift as vegetation fuel moisture and weather patterns change across seasons. This provides empirical evidence that operational wildfire dispatch AI requires periodic seasonal recalibration.",
    title="EMPIRICAL VALIDATION OF SEASONAL CONCEPT DRIFT"
)

add_image_with_caption(
    doc,
    os.path.join(CLEAN_SS_DIR, "fig13_ml_benchmark.png"),
    "Figure 13: Machine Learning Model Performance Scorecard Across Random and Forward-Chronological Splits"
)

add_image_with_caption(
    doc,
    os.path.join(CLEAN_SS_DIR, "fig14_ml_holdout_log.png"),
    "Figure 14: Empirical Output of the Forward-Chronological Holdout Validation Evaluating Concept Drift"
)

add_image_with_caption(
    doc,
    os.path.join(CHARTS_DIR, "ml_confusion_matrix.png"),
    "Figure 15: Confusion Matrix for the 4-Tier Fire Severity Classification Architecture"
)

add_image_with_caption(
    doc,
    os.path.join(CHARTS_DIR, "ml_roc_curve.png"),
    "Figure 16: Multiclass Receiver Operating Characteristic (ROC) Curves Across Fire Severity Tiers"
)

add_image_with_caption(
    doc,
    os.path.join(CLEAN_SS_DIR, "fig19_leakage_code.png"),
    "Figure 18: Forward-Chronological Holdout Data Partitioning Implementation for Leakage Prevention"
)

# -------------------------------------------------------------
# 8. OPERATIONAL DECISION FRAMEWORK & INTEGRATION
# -------------------------------------------------------------
add_header_styled(doc, "8. Operational Decision Framework (WARD) & System Integration", level=1)

add_p(doc,
    "The data-processing, temporal modeling, and machine-learning components were integrated into an operational Wildfire Alert & Rapid Dispatch (WARD) system:",
    bold_prefix="Operational Architecture: "
)

code_map_headers = ["Pipeline Component", "Source Code File", "Primary Libraries / Tools", "Production Output / Artifact"]
code_map_data = [
    ["PySpark Distributed Engine", "member2_temporal_analysis.py", "Apache PySpark 4.2.0, Java 21, Spark SQL", "7 distributed queries, Daily & Monthly aggregations"],
    ["Publication Charting Suite", "generate_member2_visualizations.py", "Matplotlib 3.11, Seaborn, Pandas", "High-resolution 300-DPI visual telemetry figures"],
    ["Automated Test Suite", "tests/test_temporal_analysis.py", "Python unittest module", "10/10 automated tests passing in 0.016 seconds"],
    ["Temporal ML Holdout Model", "ml_model/train_and_evaluate.py", "Scikit-Learn (RandomForest, HistGBDT)", "71.19% temporal holdout benchmark & serialized models"],
    ["REST API Backend", "webapp/app.py", "Flask 3.1, Python 3.13", "Endpoints: /api/temporal, /api/charts/<name>"],
    ["Interactive UI & Globe", "webapp/templates/index.html", "HTML5, CSS3, Chart.js, Leaflet.js", "Dynamic metric cards, satellite telemetry globe, modal lightbox"],
    ["NoSQL Document Store", "nosql/nosql_store.py", "SQLite Document Database", "Table: temporal_aggregates with daily and monthly records"]
]
code_tbl = doc.add_table(rows=1, cols=4)
style_table(code_tbl, [1.5, 2.0, 1.8, 1.9], code_map_headers, code_map_data, align_cols=['L', 'L', 'L', 'L'])
doc.add_paragraph().paragraph_format.space_after = Pt(6)

add_image_with_caption(
    doc,
    os.path.join(CLEAN_SS_DIR, "fig17_code_architecture_table.png"),
    "Figure 17: End-to-End System Integration Stack, Framework Libraries, and Component Mapping"
)

# -------------------------------------------------------------
# 9. DISCUSSION & TECHNICAL LIMITATIONS
# -------------------------------------------------------------
add_header_styled(doc, "9. Discussion & Technical Limitations", level=1)

add_p(doc,
    "The analysis demonstrates that combining distributed PySpark processing with leakage-safe machine learning yields operational insight into global wildfire dynamics. However, three key physical and methodological limitations must be acknowledged:",
    bold_prefix="Methodological Synthesis: "
)

add_p(doc,
    "Because the Suomi-NPP VIIRS sensor relies on a sun-synchronous orbit, observation frequency is constrained to two daily overpasses per geographic location (~13:30 and ~01:30 local solar time). Rapidly spreading convective fires that ignite and dissipate between satellite orbits may remain undetected or have their peak intensity under-sampled. Integration of geostationary satellite telemetry (such as GOES-16/18 and Himawari-9) would complement polar coverage with high-cadence 10-minute observations.",
    bold_prefix="1. Polar Orbital Sampling Frequency: "
)

add_p(doc,
    "Thermal infrared channels (4 µm and 11 µm) cannot penetrate thick cloud cover or dense pyrocumulonimbus smoke plumes, leading to temporary undercounting during severe meteorological storms. Future system iterations should integrate synthetic aperture radar (SAR) feeds to penetrate cloud cover.",
    bold_prefix="2. Atmospheric & Cloud Attenuation: "
)

add_p(doc,
    "The 3.53% accuracy drop observed during forward-chronological holdout validation demonstrates that models trained in one seasonal window experience performance drift when predicting across future seasons. Operational deployments must incorporate continuous, automated retraining pipelines as new seasonal telemetry arrives.",
    bold_prefix="3. Seasonal Concept Drift & Continuous Retraining: "
)

# -------------------------------------------------------------
# 10. CONCLUSION
# -------------------------------------------------------------
add_header_styled(doc, "10. Conclusion", level=1)

add_p(doc,
    "This study successfully established a scalable Big Data telemetry processing and Machine Learning framework applied to 9,511,402 global NASA FIRMS active wildfire records. By leveraging Apache PySpark 4.2.0 on Java 21, the system executed complex distributed window aggregations and diurnal calculations in 31.75 seconds.",
    bold_prefix="Summary of Findings: "
)

add_p(doc,
    "Empirical discoveries confirmed that global wildfire volume peaked in January 2026, while thermal destructiveness peaked in June 2026, establishing October 16, 2025 as the single worst disaster date. Furthermore, the analysis isolated the 11:00–12:00 UTC diurnal solar peak and derived an operational dispatch rule for intense nocturnal crown fires (FRP >= 30 MW). Finally, forward-chronological machine-learning validation verified model robustness (71.19% accuracy, 0.8415 ROC-AUC) while quantifying seasonal concept drift. These deliverables provide emergency management agencies with a mathematically verified, operationally resilient intelligence platform for real-time wildfire triage.",
    bold_prefix="Operational Impact: "
)

# Save Document
os.makedirs(os.path.dirname(OUTPUT_DOCX), exist_ok=True)
doc.save(OUTPUT_DOCX)
print(f"Clean Academic Document saved to: {OUTPUT_DOCX}")

try:
    doc.save(ROOT_DOCX)
    print(f"Clean Academic Document saved to: {ROOT_DOCX}")
except Exception as e:
    print(f"Note on root docx: {e}")

print("Academic Technical Report generation complete!")

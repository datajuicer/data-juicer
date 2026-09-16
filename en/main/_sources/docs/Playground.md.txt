# Web Playground

The Web Playground is Data-Juicer's interactive visualization tool. It lets you drag sliders to adjust filter thresholds, preview the effect in real time, and quickly converge on a satisfying data recipe—all in your browser.

> **Note:** The Playground requires a source installation of Data-Juicer.

---

## Launch

Run from the project root:

```bash
streamlit run app.py
```

Your browser opens the Playground interface automatically.

---

## Workflow

A typical session has four steps:

### 1. Parse Configuration

Specify a recipe (the UI pre-fills a sample), upload your own YAML, or override parameters. The interface shows parsed config alongside the raw YAML.

### 2. Analyze original data

Run the analyzer on your dataset—summary stats and distribution plots appear directly in the UI. Equivalent to running `dj-analyze`, but with inline results.

### 3. Process data

Run the full recipe. The UI shows side-by-side statistics of the original and processed data.

### 4. Tune Filter operators

This is the Playground's core feature:

- **Drag sliders** to adjust each Filter's threshold
- The page reports the **discard ratio** in real time
- Histograms show the cutoff line
- Specific **kept vs. discarded samples** are listed
- Stacked bar charts show the combined effect of all Filters
- Vocabulary diversity analysis
- **Download kept/discarded data as JSONL**

---

## Export your recipe

Once satisfied with the thresholds, copy the tuned parameters back into your recipe YAML. This "analyze → process → tune → export" loop is the core workflow for polishing data recipes with Data-Juicer.

---

## Next steps

- For command-line analyzer usage, see [Data Analysis Guide](AnalyzeData.md)
- Ready to run at scale? See [Quickstart §4](tutorial/QuickStart.md#4-run-the-pipeline)

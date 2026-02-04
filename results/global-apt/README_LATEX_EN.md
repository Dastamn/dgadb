# LaTeX Integration Guide - Global-APT Results

Quick guide to integrate the results table, evaluation pipeline, and commentary into your Overleaf paper.

## Files Included

- **`results_table.tex`** - Complete LaTeX code with:
  - Results table (all algorithms)
  - Evaluation pipeline diagram (TikZ)
  - Human-written commentary

## Quick Start

### 1. Copy the Code

Open `results_table.tex` and copy its content directly into your LaTeX document where you want the results section.

### 2. Required Packages

Add these to your document preamble (before `\begin{document}`):

```latex
\usepackage{booktabs}      % Professional tables
\usepackage{tikz}          % For pipeline diagram
\usetikzlibrary{positioning, arrows.meta}
```

### 3. That's It!

The file is self-contained and ready to use. Just paste it in your document.

## What's Included

### Results Table

A clean, professional table comparing all 5 algorithms:
- GraphSAGE (best performer)
- NetWalk
- SLADE
- GAT
- GeneralDYG

Metrics shown: ROC-AUC, Precision, Recall, F1-Score

### Evaluation Pipeline

A TikZ diagram showing the complete evaluation workflow:
- Data preparation (JSON → Parquet)
- Temporal split (70/15/15)
- Training/Validation/Test
- Algorithm execution
- Metrics calculation

### Commentary

Natural, human-written commentary explaining:
- Why GraphSAGE performs best
- Trade-offs between recall and precision
- Issues with GeneralDYG
- Importance of threshold optimization

## Example Usage in Your Paper

```latex
\documentclass{article}
\usepackage{booktabs}
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta}

\begin{document}

\section{Results}

% Paste content from results_table.tex here
\input{results_table}  % or copy-paste directly

\section{Discussion}
% Your discussion section...

\end{document}
```

## Customization

### Adjust Table Size

If the table is too wide, make it smaller:

```latex
\begin{table}[htbp]
\centering
\caption{...}
\label{tab:results}
\footnotesize  % or \tiny for very small
\begin{tabular}{lcccc}
% ... table content
```

### Change Pipeline Colors

Edit the fill colors in the TikZ diagram:

```latex
fill=blue!20   % Light blue
fill=green!20  % Light green
fill=red!20    % Light red
```

### Modify Commentary

The commentary is written in a natural, conversational style. Feel free to adjust it to match your paper's tone - make it more formal if needed, or keep the casual style if that fits.

## Visualizing in Overleaf

1. **Compile the document** - Click "Recompile" in Overleaf
2. **Check the PDF** - The table and diagram should appear
3. **Adjust if needed** - If the pipeline diagram looks off, tweak the node positions

## Tips

- The table uses `booktabs` for clean, professional lines
- Best results (GraphSAGE) are highlighted with `\textbf{}`
- The pipeline diagram is scalable - it will adjust to your page width
- All labels (`\label{}`) are set up for cross-referencing with `\ref{}`

## Troubleshooting

**Table too wide?**
- Use `\small` or `\footnotesize` before `\begin{tabular}`
- Or reduce column spacing

**Pipeline diagram cut off?**
- Adjust `node distance` values in TikZ
- Or use `\resizebox{0.9\textwidth}{!}{...}` around the tikzpicture

**Missing packages error?**
- Make sure `booktabs` and `tikz` are in your preamble
- Add `\usetikzlibrary{positioning, arrows.meta}` for TikZ

## File Structure

```
results/global-apt/
├── results_table.tex          ← Copy this into your paper
├── README_LATEX_EN.md         ← This file
└── results_all_algorithms_*.json  ← Raw results data
```

That's it! Just copy-paste and you're good to go.

# Repository Project Analysis Instructions

## Purpose

Analyze the repository and produce a clear, concise explanation of the project using the broad CS229-style outline below. The answer should **not** read like a formal paper. Instead, it should give the important ideas that directly answer each point.

The goal is for a reader to quickly understand:

- what the project does,
- why it matters,
- what data and methods it uses,
- how the experiments were run,
- what the results show,
- what is missing or could be improved.

## Required Rule: Analyze the Repository First

Before answering any category, inspect the repository directly. Do **not** answer from assumptions, memory, or generic machine-learning knowledge.

Analyze as much of the repo as possible, including:

- `README.md` and documentation files,
- project reports, notes, or design documents,
- source-code folders,
- data-loading scripts,
- preprocessing and feature-engineering code,
- model or algorithm implementation files,
- training scripts,
- evaluation scripts,
- configuration files,
- notebooks,
- experiment logs,
- result tables, figures, plots, or saved outputs,
- tests, examples, and command-line entry points,
- dependency files such as `requirements.txt`, `pyproject.toml`, `environment.yml`, or `package.json`.

When giving an answer, point to the relevant files, functions, scripts, configs, or outputs that support the claim. If something cannot be found in the repo, clearly write: **Not found in repo** and explain what information would be needed.

## Output Style

Use short sections and bullet points. Do not write long paragraphs unless a concept needs explanation.

For each category:

- answer the listed questions directly,
- include only the most important ideas,
- mention the repo evidence used,
- identify missing, unclear, or weak parts,
- avoid filler or generic descriptions,
- use equations only when they clarify the method, metric, or constraint,
- describe the project as implemented in the repo, not as it ideally should be.

## Required Structure

The final response should be organized under the broad outline categories below:

```markdown
# Project Analysis

## 1. Introduction
### 1.1 Project Summary
### 1.2 Problem and Importance
### 1.3 Motivation and Background

## 2. Related Work
### 2.1 Related Work and Existing Approaches

## 3. Dataset and Features
### 3.1 Dataset and Data Pipeline
### 3.2 Features

## 4. Methods
### 4.1 Methods and Algorithms
### 4.2 Training Procedure

## 5. Experiments / Results / Discussion
### 5.1 Evaluation Metrics
### 5.2 Experiments
### 5.3 Results
### 5.4 Discussion and Error Analysis
### 5.5 Visualizations and Tables

## 6. Conclusion / Future Work
### 6.1 Conclusion
### 6.2 Future Work

## 7. Contributions
### 7.1 Contributions

## 8. Citations and References
### 8.1 Citations and References

## Final Checklist for the Repo Analyzer
```

The subcategories must all be answered, but they should be grouped under the broader outline categories shown above.

---

# Categories to Answer

## 1. Introduction

### 1.1 Project Summary

Give a quick overview of the project.

Answer:

- What is the project title or inferred project name?
- What problem is the repo trying to solve?
- What is the core idea in one to three sentences?
- What are the main inputs?
- What are the main outputs?
- What type of model, algorithm, or system is used?
- What files in the repo show this most clearly?

### 1.2 Problem and Importance

Explain the problem and why it matters.

Answer:

- What real-world or technical problem is being addressed?
- Why is this problem important?
- Who would care about solving it?
- What goes wrong if the problem is not solved well?
- What exactly does the system take as input?
- What exactly does the system output?
- Is the task classification, regression, forecasting, optimization, control, ranking, clustering, generation, or something else?
- What assumptions about the problem are made in the repo?

### 1.3 Motivation and Background

Explain enough background for a reader to understand why the project exists.

Answer:

- What motivated the project?
- What domain context is needed to understand it?
- What is the baseline workflow without this project?
- Is the project replacing manual work, improving an existing model, enforcing constraints, reducing cost, improving accuracy, improving reliability, or something else?
- What background terms, equations, or system components should be defined?

---

## 2. Related Work

### 2.1 Related Work and Existing Approaches

Summarize how this project relates to previous work or existing methods.

Answer:

- What prior papers, tools, algorithms, or systems are mentioned in the repo?
- What categories of approaches exist for this problem?
- What are the strengths and weaknesses of those approaches?
- How is this repo similar to previous approaches?
- How is this repo different?
- What seems to be the current standard or state-of-the-art approach?
- Does the repo include at least five relevant references? If not, say what reference areas are missing.
- If the repo does not include related work, infer only the likely categories and mark them as **needs external citation**.

---

## 3. Dataset and Features

### 3.1 Dataset and Data Pipeline

Describe the data used by the project.

Answer:

- What dataset or datasets are used?
- Where does the data come from?
- How many examples, rows, time steps, regions, files, or samples are used?
- How is the data split into training, validation, and test sets?
- What is the time range of the data, if applicable?
- What is the unit of observation, such as hourly load, image, patient, text sequence, market region, node, edge, or transaction?
- What preprocessing is performed?
- Are missing values handled? If so, how?
- Is normalization, scaling, encoding, discretization, filtering, or aggregation used?
- Are there any data leakage risks?
- What files implement the data pipeline?

### 3.2 Features

Explain what information the model uses to make predictions or decisions.

Answer:

- What raw variables are used?
- What engineered features are created?
- Are time, seasonality, lag variables, rolling statistics, geography, region labels, weather, prices, constraints, or external variables used?
- Are features selected manually or automatically?
- Are features transformed using methods such as PCA, Fourier transforms, embeddings, one-hot encoding, polynomial expansion, or normalization?
- Which features seem most important?
- Are any important features missing?
- What files show the feature construction process?

---

## 4. Methods

### 4.1 Methods and Algorithms

Explain the model, algorithm, or system architecture.

Answer:

- What algorithm or algorithms are implemented?
- How does each algorithm work at a high level?
- What objective function, loss function, reward function, or optimization problem is used?
- What constraints are included, if any?
- Are the constraints hard constraints, soft penalties, post-processing rules, projections, filters, or learned behavior?
- What parameters or learned weights does the model use?
- What mathematical notation is necessary to understand the method?
- What scripts or modules implement the method?
- If multiple methods are used, explain how they connect.
- If the repo has a constraint layer, explain where it sits in the pipeline and how it modifies or validates model outputs.

### 4.2 Training Procedure

Explain how the model is trained or fit.

Answer:

- What training script or notebook is used?
- What target variable is the model trained to predict or optimize?
- What loss function or training objective is minimized or maximized?
- What optimizer or solver is used?
- What hyperparameters are chosen?
- How are hyperparameters selected?
- Is cross-validation used? If yes, how many folds or what split method?
- Are random seeds set?
- Is training reproducible from the repo?
- What commands would a reader run to train the model?

---

## 5. Experiments / Results / Discussion

### 5.1 Evaluation Metrics

Explain how performance is measured.

Answer:

- What are the primary metrics?
- Why are those metrics appropriate for the task?
- Are there secondary metrics?
- For regression or forecasting, are metrics such as MAE, MSE, RMSE, MAPE, bias, or average error used?
- For classification, are metrics such as accuracy, precision, recall, F1, AUC, AUPRC, or confusion matrix used?
- For optimization or constraint problems, are cost, feasibility, violations, penalties, runtime, or robustness measured?
- Are metrics computed overall and by subgroup, region, time period, or operating condition?
- Are equations for the metrics included or needed?
- What files calculate the metrics?

### 5.2 Experiments

Describe what experiments were run.

Answer:

- What models, baselines, or variants are compared?
- What hyperparameters or design choices are tested?
- Is there an ablation study?
- Are there sensitivity tests?
- Are there stress tests or edge cases?
- Are train/validation/test results reported separately?
- Are experiments repeated across seeds, regions, time periods, or folds?
- What command, script, notebook, or output file contains each experiment?

### 5.3 Results

Summarize the quantitative and qualitative results.

Answer:

- What are the main numerical results?
- Which model or method performs best?
- How large is the improvement over baseline?
- Are there tables, plots, logs, or figures showing the results?
- Are there qualitative examples of success and failure?
- Where does the algorithm fail?
- Does the project discuss why certain methods worked or failed?
- Are the results strong enough to support the project claims?
- Are any results missing, unclear, or not reproducible?

### 5.4 Discussion and Error Analysis

Interpret the results instead of just listing them.

Answer:

- What do the results mean?
- Why did the best method likely work best?
- Why did weaker methods fail?
- Is the model overfitting or underfitting?
- What evidence supports that conclusion?
- Are errors concentrated in certain inputs, time periods, regions, classes, or scenarios?
- Are there unrealistic assumptions or limitations?
- Are the constraints too weak, too strong, or appropriately calibrated?
- What tradeoffs appear between accuracy, cost, feasibility, runtime, and interpretability?

### 5.5 Visualizations and Tables

Identify what figures or tables should be included or discussed.

Answer:

- What plots, tables, or figures already exist in the repo?
- What do they show?
- Are axes, units, legends, labels, and titles clear?
- Are both quantitative and qualitative results shown?
- Are there missing visualizations that would make the project easier to understand?
- For forecasting or regression, should there be predicted vs. actual plots, residual plots, error distributions, or time-series plots?
- For classification, should there be confusion matrices, ROC curves, or precision-recall curves?
- For optimization or constraints, should there be violation counts, cost breakdowns, or feasibility plots?

---

## 6. Conclusion / Future Work

### 6.1 Conclusion

Give the key takeaway.

Answer:

- What did the project accomplish?
- What is the strongest result?
- Which method was best?
- What is the main lesson from the experiments?
- Did the repo successfully solve the original problem?
- What claims are supported by evidence?
- What claims are not fully supported?

### 6.2 Future Work

List the most important improvements.

Answer:

- What would improve the project with more time?
- What additional data would help?
- What stronger baselines should be tested?
- What modeling improvements should be tried?
- What evaluation gaps should be fixed?
- What engineering work is needed for deployment or scaling?
- What assumptions should be relaxed?
- What parts of the repo should be refactored or documented better?

---

## 7. Contributions

### 7.1 Contributions

Explain who did what, if the repo includes that information.

Answer:

- Does the repo identify team members?
- What did each person contribute?
- Are contributions separated by data, modeling, experiments, writing, infrastructure, or analysis?
- If contribution information is missing, write **Not found in repo**.

---

## 8. Citations and References

### 8.1 Citations and References

Check whether the repo properly cites outside sources.

Answer:

- What papers, datasets, libraries, or tools are cited?
- Are dataset sources cited?
- Are algorithms or baseline methods cited?
- Are formulas, metrics, or domain-specific constants cited?
- Are citations complete enough for a reader to find the sources?
- Are any important claims missing citations?
- If the repo includes BibTeX, references, or citation files, identify them.

---

## Final Checklist for the Repo Analyzer

Before finalizing the answer, verify that:

- every major claim is supported by repo evidence,
- every subcategory above is answered,
- missing information is labeled clearly,
- the input and output of the system are explicitly stated,
- the dataset, features, methods, metrics, experiments, and results are separated clearly,
- the answer is grouped under the broad outline categories,
- the answer is written as important ideas, not as a formal paper,
- the response helps a new reader understand the project quickly.

Customer Churn & Lifecycle Analysis — Pokémon Go

Overview
This project analyses behavioural and transaction data from 50,000 active Pokémon Go players across two seasons (Summer and Fall) to understand customer lifecycle patterns, model customer value, and predict churn among paying users.
The analysis covers four interconnected areas: customer profiling and segmentation, lifecycle grid construction, Customer Lifetime Value (CLV) modelling, and churn prediction using logistic regression with stepwise feature selection.

Key Results
•	50,000 active summer players analysed across 4 customer segments
•	18,149 paying users identified for CLV and churn modelling
•	Average CLV: €9.53 across paying users (12-month horizon, BG/NBD + Gamma-Gamma model)
•	Social Raiders identified as the highest-value segment by predicted CLV
•	Churn model achieved a 1.10x lift in targeting accuracy
•	Targeting the riskiest 20% of flagged players captures 22.1% of all actual churners

Project Structure
├── pokemon_go_analysis.py       # Main analysis script
├── ROC_Curve.png                # ROC curve output from churn model
├── Usage_Heatmap.png            # Game usage lifecycle grid heatmap
└── README.md

Note: Raw data files (CSV) are not included in this repository. The datasets were provided under academic licence for this assignment and are not publicly shareable.

Methodology
Q1 — Basetable Construction & Customer Profiling
•	Merged five data sources: summer/fall financial transactions, summer/fall session data, and customer demographics
•	Built a unified basetable for all active summer players (minimum 1 play session)
•	Aggregated usage metrics: total sessions, distance walked, duration, experience points, recency
•	Aggregated financial metrics: total purchases, total spend, purchase recency
•	Profiled 4 customer segments: Walker, Miscellaneous, Social Raider, Catcher

Q2 — Lifecycle Grids
Two lifecycle grids were constructed to visualise customer behaviour patterns:
•	Financial Lifecycle Grid (paying users only): Recency of purchase × Frequency of purchase tiers — using manual bins due to low purchase frequency distribution
•	Game Usage Lifecycle Grid (all active players): Recency of play × Frequency of sessions — using quantile-based binning

Q3 — Customer Lifetime Value (CLV) Modelling
CLV was modelled using the BG/NBD + Gamma-Gamma framework via the lifetimes library:
•	BetaGeoFitter (BG/NBD): Models the probability that a customer is still active and predicts future transaction frequency
•	GammaGammaFitter: Models the expected monetary value per transaction
•	Both models fitted with a penaliser coefficient of 0.1 to handle convergence issues from the skewed gaming transaction distribution
•	CLV projected over a 12-month horizon with a 10% annual discount rate
•	Model fitted on returning customers (frequency > 0) within the paying user cohort

Q4 — Churn Analysis & Prediction
Churn definition: A summer paying user who made zero purchases in the Fall season.
Modelling approach:
•	Logistic Regression with stepwise forward selection using AIC as the elimination criterion (via a custom stepwise module)
•	Features considered: Fall bonus receipt, customer type (dummified), play recency, total session duration, total distance, age, income (dummified), gender (dummified)
•	Final model evaluated using: Accuracy, AUC-ROC, Confusion Matrix, and Lift at top 20% targeting

Business evaluation:
•	The Fallbonus Effect Matrix was constructed to show churn rate differences between players who received the promotional bonus vs those who did not, broken down by customer segment


Libraries Used
 
•	pandas
•	numpy
•	statsmodels
•	matplotlib
•	seaborn
•	plotnine
•	scikit-learn
•	lifetimes
 

Install dependencies:
pip install pandas numpy statsmodels matplotlib seaborn plotnine scikit-learn lifetimes
The stepwise module used for forward feature selection was provided as part of the course materials and is not a standard PyPI package.

Context
Course: Data-Driven (Customer) Insights — MSc Data Analytics & AI
Institution: EDHEC Business School
Period: January – March 2026
Team project


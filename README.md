# USOPC Interview Project

This repository is organized as follows:

+ `dashboard/` is a directory containing an interactive Python Reflex
dashboard that can be used to interact with the data and modeling output.
+ `images/` contain images from exploratory data analysis.
+ `out/` contains generalized linear modeling output.
+ `report_project/` contains all of the materials for the two-page write-up
that explains the approach taken to the project, lessons learned, challenges
overcome, etc.
+ `report_sports_science/` contains all of the materials for a more extensive
sports science analysis that describes and interprets the results of the
analysis.
+ `LoadandWellnessData.xlsx` is the provided data.
+ `Makefile` is a Makefile that allows for automated execution of the various
parts of the analysis.
+ `REAMDE.md` is this document.
+ `analysis_weekly_workload.py` bins athlete data into athlete-weeks, aggregating
the variables of interest differently based on their nature and description.
+ `fit_model_gamma.r` is an R script that fits a gamma generalized linear
regression to the output of `analysis_weekly_workload.py`.
+ `util.py` is a utility module.


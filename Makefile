gamma_output_dirs := workload_distance_sleep_soreness workload_sleep workload_sleep_fatigue workload_sleep_fatigue_delta workload_sleep_mood workload_sleep_mood_delta workload_sleep_motivation workload_sleep_motivation_delta workload_sleep_stress workload_sleep_stress_delta

eda:
	source .venv/bin/activate; python eda.py

analysis_weekly_workload:
	source .venv/bin/activate; python analysis_weekly_workload.py

# Thanks:
# https://www.r-bloggers.com/2015/09/passing-arguments-to-an-r-script-from-command-lines/ 
fit_gamma:
	$(foreach p, $(gamma_output_dirs), Rscript fit_gamma_model.r out/analysis_weekly_workload/$(p)/; )
	Rscript fit_gamma_model.r out/analysis_weekly_workload/workload_distance_sleep_soreness/ "1.5" "4.2"
	Rscript fit_gamma_model.r out/analysis_weekly_workload/workload_sleep_mood/ "3.2" "4.5"
	Rscript fit_gamma_model.r out/analysis_weekly_workload/workload_sleep_motivation/ "2.5" "5.0"
	Rscript fit_gamma_model.r out/analysis_weekly_workload/workload_sleep_stress/ "1.0" "4.5"


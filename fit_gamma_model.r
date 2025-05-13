library(jsonlite)
library(ggplot2)
library(broom)
library(tidyverse)

# https://www.geeksforgeeks.org/how-to-read-command-line-parameters-from-an-r-script/
args_ <- commandArgs(trailingOnly=TRUE)
# args_ <- c(
# 	'out/analysis_weekly_workload/workload_sleep_fatigue/'
# )

print(args_)

# Load data.
data_train <- read.csv(paste(args_[1], 'data_train.csv', sep=''))
data_train = data_train[, -1]
data_test <- read.csv(paste(args_[1], 'data_test.csv', sep=''))
data_test = data_test[, -1]

if (length(args_) > 1) {
	# https://www.geeksforgeeks.org/how-to-convert-character-to-numeric-in-r/
	override_xlim <- c(as.numeric(args_[2]), as.numeric(args_[3]))
} else {
	override_xlim <- c(NA, NA)
}

# Extract feature names.
feature_names = colnames(data_train)
outcome <- tail(feature_names, 1)
outcome_label_text = gsub("_", " ", gsub("X100.", "100 - ", outcome))
predictors <- feature_names[1:length(feature_names) - 1]

# Pick an adjustment.
if (min(data_train[outcome]) <= 0) {
	adj <- abs(min(data_train[outcome])) + 0.5
} else {
	adj <- 0
}

# Adjust these so that we have nonzero input such that the gamma
# regression can fit.
data_train[outcome] = data_train[outcome] + adj
data_test[outcome] = data_test[outcome] + adj

# Briefly confirm what our outcome and predictors are.
print("Outcome:")
print(outcome)
print(min(data_train[outcome]))
print("Predictors:")
print(predictors)

# Build formula.
form <- paste(outcome, '~', paste(predictors, collapse=' + '))

# Basic regression.
mod_gamma <- glm(
	form,
	data=data_train,
	family=Gamma(link='log'),
	control=list(trace=FALSE)
)

# Find influential outliers and get rid of them.
im <- influence.measures(mod_gamma)
im_col_sums <- colSums(im$is.inf)
idx_inf_multiple <- which(rowSums(im$is.inf) > 1)
print("Influential outlier indeces:")
print(idx_inf_multiple)
data_train_subset <- data_train[-idx_inf_multiple, ]

# https://www.digitalocean.com/community/tutorials/get-number-of-rows-and-columns-in-r
print(paste("Dropped ", nrow(data_train) - nrow(data_train_subset), ' outliers', sep=''))

# Refit without outliers.
mod_gamma <- glm(
	form,
	data=data_train_subset,
	family=Gamma(link='log'),
	control=list(trace=FALSE)
)

# Use broom to get a tidy summary of our data.
tidy_glm_reg <- tidy(mod_gamma, conf.int = TRUE)

# Get our coefficients.
coef_ = sort(coefficients(mod_gamma)[-1])
confint_ = confint(mod_gamma)
df_coef <- data.frame(coef_)
df_coef['coef_name'] = names(coef_)

# Make predictions on the test set.
preds_test <- predict(mod_gamma, newdata=data_test)
df_pred_test <- data.frame(
	"y_test"=data_test[outcome],
	"y_pred_test"=preds_test
)
names(df_pred_test) <- c('y_test', 'y_pred_test')

#print(df_pred_test)

# Predictions.
png(paste(args_[1], 'scatter_pred_true.png', sep=''))
p <- ggplot(df_pred_test, aes(log(y_test), y_pred_test)) + geom_point()
if (length(args_) > 1) {
	print("Overriding plot x axis limits...")
	print(override_xlim)
	p + lims(x=override_xlim) + xlab(paste("True Log(", outcome_label_text, ")", sep='')) + ylab(paste("Predicted ", outcome_label_text, sep=''))
} else {
	# https://stackoverflow.com/questions/11936339/replace-specific-characters-within-strings
	p + xlab(paste("True Log(", outcome_label_text, ")", sep='')) + ylab(paste("Predicted ", outcome_label_text, sep=''))
}
dev.off()

# Coefficient plot.
# Thanks:
# https://interludeone.com/posts/2022-12-15-coef-plots/coef-plots.html
# https://stackoverflow.com/questions/63883240/controlling-dpi-of-plot-in-r/63883327
png(paste(args_[1], 'coef.png', sep=''), units="in", width=5, height=5, res=300)
tidy_glm_reg %>%
	filter(term != "(Intercept)") %>%
	# This will place the largest coefficient on top.
	mutate(term = fct_reorder(term, estimate)) %>%
	ggplot(aes(estimate, term)) +
	geom_point() +
	# Confidence intervals on coefficients.
	geom_errorbarh(aes(xmin = conf.low, xmax = conf.high)) +
	# Display where zero is for ease of interpretation.
	geom_vline(xintercept=0, lty=2) +
	labs(
		x = paste("Estimate of Coefficient Effect on Log(", outcome_label_text, ")", sep=''),
		y = NULL,
		title = NULL
	)
dev.off()


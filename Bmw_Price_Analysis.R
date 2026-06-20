# BMW Used Car Price Prediction: Statistical vs ML Models with SHAP
# Ali Bahmanyar - Kharazmi University
# Data: https://www.kaggle.com/datasets/danielkyrka/bmw-pricing-challenge

library(tidyverse)
library(lubridate)
library(moments)
library(caret)
library(glmnet)
library(randomForest)
library(xgboost)
library(Metrics)
library(shapviz)
library(patchwork)

set.seed(42)

# ── 1. Load and clean data ───────────────────────────────────────

raw <- read_csv("bmw_pricing_challenge.csv")

bmw <- raw %>%
  mutate(
    reg_yr   = year(ymd(registration_date)),
    car_age  = 2018 - reg_yr,
    n_feat   = feature_1 + feature_2 + feature_3 + feature_4 +
               feature_5 + feature_6 + feature_7 + feature_8,
    model_grp = case_when(
      str_detect(model_key, "^1") ~ "S1",
      str_detect(model_key, "^2") ~ "S2",
      str_detect(model_key, "^3") ~ "S3",
      str_detect(model_key, "^4") ~ "S4",
      str_detect(model_key, "^5") ~ "S5",
      str_detect(model_key, "^6") ~ "S6",
      str_detect(model_key, "^7") ~ "S7",
      str_detect(model_key, "^X") ~ "X",
      str_detect(model_key, "^M") ~ "M",
      str_detect(model_key, "^Z") ~ "Z",
      TRUE ~ "Other"
    ),
    log_price = log(price),
    log_mlg   = log(mileage + 1)
  )

q1 <- quantile(bmw$price, 0.25); q3 <- quantile(bmw$price, 0.75)
iqr <- q3 - q1
hi_bound  <- q3 + 3 * iqr

bmw_clean <- bmw %>%
  filter(
    mileage > 0, engine_power >= 50, price >= 1000,
    car_age >= 0, car_age <= 30,
    price <= hi_bound
  )

q1m <- quantile(bmw_clean$mileage, 0.25); q3m <- quantile(bmw_clean$mileage, 0.75)
hi_mlg <- q3m + 3 * (q3m - q1m)
bmw_clean <- bmw_clean %>% filter(mileage <= hi_mlg)

write_csv(bmw_clean, "bmw_clean.csv")


# ── 2. Prepare model matrix ──────────────────────────────────────

dat <- bmw_clean %>%
  select(price, log_price, mileage, log_mlg, engine_power, car_age,
         n_feat, feature_1:feature_8, model_grp, fuel, car_type, paint_color) %>%
  mutate(across(starts_with("feature_"), as.integer),
         across(c(model_grp, fuel, car_type, paint_color), as.factor))

idx   <- createDataPartition(dat$price, p = 0.8, list = FALSE)
train <- dat[idx, ]
test  <- dat[-idx, ]

eval_metrics <- function(y, yhat, label) {
  data.frame(model = label,
             rmse = rmse(y, yhat),
             mae  = mae(y, yhat),
             r2   = cor(y, yhat)^2)
}

fml <- log_price ~ log_mlg + engine_power + car_age + n_feat +
       model_grp + fuel + car_type + paint_color


# ── 3. Linear, Ridge, Lasso ──────────────────────────────────────

lm_fit <- lm(fml, data = train)
lm_pred <- exp(predict(lm_fit, newdata = test))
m_lm <- eval_metrics(test$price, lm_pred, "Linear")

X_tr <- model.matrix(fml, data = train)[, -1]
X_te <- model.matrix(fml, data = test)[, -1]
y_tr <- train$log_price

ridge_cv  <- cv.glmnet(X_tr, y_tr, alpha = 0, nfolds = 10)
ridge_fit <- glmnet(X_tr, y_tr, alpha = 0, lambda = ridge_cv$lambda.min)
ridge_pred <- exp(predict(ridge_fit, s = ridge_cv$lambda.min, newx = X_te))
m_ridge <- eval_metrics(test$price, as.vector(ridge_pred), "Ridge")

lasso_cv  <- cv.glmnet(X_tr, y_tr, alpha = 1, nfolds = 10)
lasso_fit <- glmnet(X_tr, y_tr, alpha = 1, lambda = lasso_cv$lambda.min)
lasso_pred <- exp(predict(lasso_fit, s = lasso_cv$lambda.min, newx = X_te))
m_lasso <- eval_metrics(test$price, as.vector(lasso_pred), "Lasso")


# ── 4. Random Forest ──────────────────────────────────────────────

rf_fit <- randomForest(fml, data = train, ntree = 500, mtry = 4, importance = TRUE)
rf_pred <- exp(predict(rf_fit, newdata = test))
m_rf <- eval_metrics(test$price, rf_pred, "RandomForest")


# ── 5. XGBoost ─────────────────────────────────────────────────────

dtrain <- xgb.DMatrix(data = X_tr, label = train$log_price)
dtest  <- xgb.DMatrix(data = X_te, label = test$log_price)

xgb_fit <- xgb.train(
  params = list(objective = "reg:squarederror", eta = 0.05,
                max_depth = 6, subsample = 0.8, colsample_bytree = 0.8),
  data = dtrain, nrounds = 500,
  evals = list(train = dtrain, test = dtest),
  verbose = 0, early_stopping_rounds = 30
)
xgb_pred <- exp(predict(xgb_fit, dtest))
m_xgb <- eval_metrics(test$price, xgb_pred, "XGBoost")


# ── 6. Results table ──────────────────────────────────────────────

results <- bind_rows(m_lm, m_ridge, m_lasso, m_rf, m_xgb) %>% arrange(rmse)
write_csv(results, "model_comparison.csv")

saveRDS(rf_fit,  "rf_model.rds")
saveRDS(xgb_fit, "xgb_model.rds")


# ── 7. SHAP analysis (XGBoost) ────────────────────────────────────

shp <- shapviz(xgb_fit, X_pred = X_te)

p_imp  <- sv_importance(shp, kind = "bar", max_display = 10)
p_bee  <- sv_importance(shp, kind = "beeswarm", max_display = 10)
ggsave("fig_shap_importance.png", p_imp, width = 8, height = 5, dpi = 150)
ggsave("fig_shap_beeswarm.png",   p_bee, width = 8, height = 6, dpi = 150)

p_dep_age  <- sv_dependence(shp, v = "car_age")
p_dep_pow  <- sv_dependence(shp, v = "engine_power")
p_dep_mlg  <- sv_dependence(shp, v = "log_mlg")
p_dep_feat <- sv_dependence(shp, v = "n_feat")
fig_dep <- (p_dep_age | p_dep_pow) / (p_dep_mlg | p_dep_feat)
ggsave("fig_shap_dependence.png", fig_dep, width = 14, height = 10, dpi = 150)

idx_max <- which.max(test$price)
idx_min <- which.min(test$price)
ggsave("fig_waterfall_high.png", sv_waterfall(shp, row_id = idx_max), width = 9, height = 6, dpi = 150)
ggsave("fig_waterfall_low.png",  sv_waterfall(shp, row_id = idx_min), width = 9, height = 6, dpi = 150)


# ── 8. Statistical vs SHAP importance comparison ──────────────────

lm_imp <- summary(lm_fit)$coefficients %>%
  as.data.frame() %>%
  rownames_to_column("var") %>%
  filter(var %in% c("log_mlg", "engine_power", "car_age", "n_feat")) %>%
  mutate(score = abs(`t value`) / max(abs(`t value`)) * 100,
         method = "Linear (|t|)") %>%
  select(var, score, method)

shap_imp <- colMeans(abs(shp$S)) %>%
  as.data.frame() %>%
  rownames_to_column("var") %>%
  rename(val = ".") %>%
  filter(var %in% c("log_mlg", "engine_power", "car_age", "n_feat")) %>%
  mutate(score = val / max(val) * 100, method = "XGBoost (SHAP)") %>%
  select(var, score, method)

cmp <- bind_rows(lm_imp, shap_imp) %>%
  mutate(var = recode(var, log_mlg = "Mileage", engine_power = "Power",
                       car_age = "Age", n_feat = "Features"))

p_cmp <- ggplot(cmp, aes(x = reorder(var, score), y = score, fill = method)) +
  geom_col(position = "dodge", width = 0.6) +
  coord_flip() +
  labs(x = NULL, y = "Normalized importance", fill = "Method") +
  theme_minimal()
ggsave("fig_importance_comparison.png", p_cmp, width = 9, height = 5, dpi = 150)


# ── 9. Residual diagnostics ────────────────────────────────────────

test$pred_xgb  <- xgb_pred
test$err       <- test$price - test$pred_xgb
test$age_group <- cut(test$car_age, breaks = c(0, 3, 7, 12, 30),
                       labels = c("0-3y", "4-7y", "8-12y", "13+y"))

p_err <- ggplot(test, aes(x = car_age, y = err)) +
  geom_point(alpha = 0.25, color = "#1A56DB") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "#E3742F") +
  geom_smooth(method = "loess", color = "#E3742F", se = TRUE) +
  labs(x = "Car age (years)", y = "Prediction error (EUR)") +
  theme_minimal()
ggsave("fig_prediction_error.png", p_err, width = 9, height = 5.5, dpi = 150)

err_by_age <- aggregate(abs(err) ~ age_group, data = test, FUN = mean)
write_csv(test %>% select(car_age, price, pred_xgb, err, age_group), "prediction_errors.csv")

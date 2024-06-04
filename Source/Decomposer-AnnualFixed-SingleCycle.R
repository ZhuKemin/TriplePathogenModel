library(forecast)
library(tidyverse)
library(lubridate)
library(ggplot2)

# 主处理函数
process_flu_data <- function(filename, pathogen) {
  # ---------------------------- 读取数据 ----------------------------
  if (pathogen == 'Flu') {
    n_annual <- 52
    low_bound <- 80
    file_path <- paste0("../Data/FluData/China_influenza_seasonality-main/cleandata/", filename, ".csv")
    flu_data <- read.csv(file_path)
    flu_data$t <- as.Date(flu_data$t, format = "%Y-%m-%d")
  }

  if (pathogen == 'RSV') {
    n_annual <- 12
    low_bound <- 18
    file_path <- paste0("../Data/RSVData/", filename, ".csv")
    flu_data <- read.csv(file_path)
    flu_data$t <- as.Date(flu_data$t, format = "%Y/%m/%d")
  }

  output_dir <- file.path(paste0("../Output/", pathogen, "-", filename))
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # ----------------------- 计算单季节性周期分解 -----------------------
  single_period_flu_ts <- msts(flu_data$data, seasonal.periods = n_annual)
  single_period_mstl_result <- mstl(single_period_flu_ts, s.window = 'periodic', iterate = 50)

  # 将单周期分解结果与日期合并
  single_period_mstl_df_with_date <- cbind(Date = flu_data$t, data.frame(single_period_mstl_result))

  # 为单周期分解结果命名和保存CSV文件
  single_period_output_csv <- paste0(output_dir, "/SingleCycle", ".csv")
  write.csv(single_period_mstl_df_with_date, single_period_output_csv, row.names = FALSE)

  # ---------------------- 搜索最优第二季节性周期 ----------------------
  period_values <- c()

  # Save each result to its own file
  cycle_output_dir <- paste0(output_dir, "/AnnualFixed-SingleCycle/")
  if (!dir.exists(cycle_output_dir)) {
    dir.create(cycle_output_dir, recursive = TRUE)
  }

  # 循环遍历周期
  half_rows <- floor((nrow(flu_data) - 1) / 2)
  for (cycle in seq(low_bound, half_rows, by = 1)) {
    flu_ts <- msts(flu_data$data, seasonal.periods = c(n_annual, cycle))
    mstl_result <- mstl(flu_ts, s.window = 'periodic', iterate = 50, t.degree = 0)
    
    # 存储当前周期的分解结果
    current_mstl_result = cbind(Date = flu_data$t, data.frame(mstl_result))
    output_csv <- paste0(cycle_output_dir, "cycle_", cycle, ".csv")
    write.csv(current_mstl_result, output_csv, row.names = FALSE)

    # 保存周期值
    period_values <- c(period_values, cycle)
  }

  cat("Processed cycles from", low_bound, "to", half_rows, "\n")
}


# 从命令行读取参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 2) {
  process_flu_data(args[1], args[2])
}
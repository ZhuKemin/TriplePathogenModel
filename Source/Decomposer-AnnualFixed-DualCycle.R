library(forecast)
library(tidyverse)
library(lubridate)
library(ggplot2)

# 考虑第一周期是一整年, 同时优化第二第三周期的方案

# 主处理函数
process_flu_data <- function(filename, pathogen) {
  # ---------------------------- 读取数据 ----------------------------
  if (pathogen == 'Flu') {
    file_path <- paste0("../Data/FluData/China_influenza_seasonality-main/cleandata/", filename, ".csv")
    flu_data <- read.csv(file_path)
    flu_data$t <- as.Date(flu_data$t, format = "%Y-%m-%d")
  }

  if (pathogen == 'RSV') {
    file_path <- paste0("../Data/RSVData/", filename, ".csv")
    flu_data <- read.csv(file_path)
    flu_data$t <- as.Date(flu_data$t, format = "%Y/%m/%d")
  }

  output_dir <- file.path(paste0("../Output/", pathogen, "-", filename))
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # ---------------------- 搜索最优第二和第三季节性周期 ----------------------
  half_rows <- floor((nrow(flu_data) - 1) / 2)
  n_annual <- if (pathogen == 'Flu') 52 else 12
  second_lower_bound <- if (pathogen == 'Flu') 52 else 12
  third_lower_bound <- if (pathogen == 'Flu') 52 else 12

  # 保存每个周期组合的分解结果
  cycle_output_dir <- paste0(output_dir, "/AnnualFixed-DualCycle/")
  if (!dir.exists(cycle_output_dir)) {
    dir.create(cycle_output_dir, recursive = TRUE)
  }

  for (second_cycle in seq(second_lower_bound, half_rows, by = 1)) {
    for (third_cycle in seq(third_lower_bound, half_rows, by = 1)) {
      flu_ts <- msts(flu_data$data, seasonal.periods = c(n_annual, second_cycle, third_cycle))
      mstl_result <- mstl(flu_ts, s.window = 'periodic', iterate = 50, t.degree = 0)
      
      # 存储当前周期组合的分解结果
      result_file_path <- paste0(cycle_output_dir, "cycle_", second_cycle, "_", third_cycle, ".csv")
      result_df <- cbind(Date = flu_data$t, data.frame(mstl_result))
      write.csv(result_df, result_file_path, row.names = FALSE)
    }
  }

  cat("Processed cycles from", second_lower_bound, "to", half_rows, "\n")
}


# 从命令行读取参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 2) {
  process_flu_data(args[1], args[2])
}
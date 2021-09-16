library(tidyverse)
library(drc)

all_data <- read_csv("data/2021-08-26_nano_bret_test.reformed.csv")
all_data$mBU <- all_data$Lum_610 / all_data$Lum_450 * 1000


#--------------------------------------------------------

constructs <- all_data$Construct %>% unique()

construct <- constructs[1]
rm("all_reformed_data")
for (construct in constructs) {
    construct_data <- all_data %>% filter(Construct == construct)
    min_time <- min(construct_data$Time)
    construct_data$Seconds <- construct_data$Time - min_time
    timepoints <- c()
    mean_corrected_mbus <- c()
    concentrations <- c()
    sds <- c()
    for (timepoint in unique(construct_data$Seconds)) {
        construct_data_timepoint <- construct_data %>% filter(Seconds == timepoint)
        control_mean_mBU <- mean(construct_data_timepoint$mBU[!construct_data_timepoint$With_618])
        control_sd_mBU <- sd(construct_data_timepoint$mBU[!construct_data_timepoint$With_618])
        construct_data_timepoint <- construct_data_timepoint %>% filter(With_618)
        for (concentration in unique(construct_data_timepoint$uM)) {
            sample_mean_mBU <- mean(construct_data_timepoint$mBU[construct_data_timepoint$uM == concentration])
            sample_sd_mBU <- sd(construct_data_timepoint$mBU[construct_data_timepoint$uM == concentration])
            sample_mean_corrected_mBU <- sample_mean_mBU - control_mean_mBU
            timepoints <- c(timepoints, timepoint)
            mean_corrected_mbus <- c(mean_corrected_mbus, sample_mean_corrected_mBU)
            concentrations <- c(concentrations, concentration)
            sds <- c(sds, sample_sd_mBU)
        }
    }
    construct_tibble <- tibble(Seconds = timepoints, uM = concentrations, mBU_corrected = mean_corrected_mbus, stdev = sds, Construct = construct) %>% arrange(Seconds, uM)
    if (!exists("all_reformed_data")) {
        all_reformed_data <- construct_tibble
    } else {
        all_reformed_data <- rbind(all_reformed_data, construct_tibble) %>% as_tibble()
    }
     p <- ggplot(construct_tibble, aes(x = uM, y = mBU_corrected, col = as.factor(Seconds / 60))) +
         theme_bw() +
         geom_point() +
         geom_line() +
         scale_x_log10(breaks = unique(construct_tibble$uM)) +
         guides(col = guide_legend(title = "Minutes")) +
         labs(title = construct)
     p
     ggsave(paste0("~/data/Dynamite/", Sys.Date(), "_", str_replace_all(construct, " ", "_"), "_nanobret_test.png"), dpi=1000)
}

rm("all_ec50")
constructs_all <- c()
timepoints <- c()
ec50s <- c()
for (construct in constructs) {
    construct_data <- all_reformed_data %>% filter(Construct == construct)
    for (timepoint in unique(construct_data$Seconds)) {
        timepoint_data <- construct_data %>% filter(Seconds == timepoint)
        drc_model <- drm(mBU_corrected ~ uM, data = timepoint_data, fct = LL.4())
        constructs_all <- c(constructs_all, construct)
        timepoints <- c(timepoints, timepoint)
        ec50s <- c(ec50s, drc_model$coefficients[4] %>% unlist() %>% as.numeric())
    }
}
all_ec50 <- tibble(Construct = constructs_all, Seconds = timepoints, ec50 = ec50s)

p <- ggplot(all_ec50 %>% filter(Seconds > 0), aes(x = Seconds/60, y = ec50 * 1000, col = Construct))+
    theme_bw() +
    geom_point() +
    geom_line() +
    labs(x = "Minutes", y = "EC50 (nM)")
p
ggsave(paste0("~/data/Dynamite/", Sys.Date(), "_nanobret_test_ec50s.png"), dpi=1000)

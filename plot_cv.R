library(dplyr)
library(ggplot2)
library(tidyr)

theme_set(theme_minimal())

cv_results <- read.csv('results_regression/cv_results_v0.csv')

cv_long <- cv_results %>% 
  pivot_longer(
    cols = split0_test_score:split11_test_score,
    names_to = c("split"),
    names_pattern = "split(.*)_test_score",
    values_to = "test_score"
  ) %>% 
  select(learner:test_score) %>% 
  mutate(
    test_score = -test_score,
    data = stringr::str_remove_all(data, 'data/'),
    data = stringr::str_remove_all(data, '.csv')
  )

cv_long %>% 
  ggplot(aes(x = test_score, fill = encoder, color = encoder)) + 
  geom_density(alpha = 0.3) +
  geom_rug() + 
  facet_wrap(~data, scales = 'free')


plot_lm <- cv_long %>% 
  filter(learner == 'lm', encoder != 'se', !grepl('house', data)) %>% 
  ggplot(aes(x = test_score, fill = encoder, color = encoder)) + 
  # geom_histogram(alpha = 0.3, bins = 5, position = 'dodge') +
  geom_density(alpha = 0.3) + 
  geom_rug() + 
  facet_wrap(~data, scales = 'free', ncol = 2) +
  xlab('Cross validation scores') + 
  ggtitle("Linear model cross-validation results", 
          "Comparison of target and quantile encodings") +
  ggsave("results_regression/lm_cv_results.png")

plot_lm_se <- cv_long %>% 
  filter(learner == 'lm', !grepl('house', data)) %>% 
  ggplot(aes(x = test_score, fill = encoder, color = encoder)) + 
  # geom_histogram(alpha = 0.3, bins = 5, position = 'dodge') +
  geom_density(alpha = 0.3) + 
  geom_rug() + 
  facet_wrap(~data, scales = 'free', ncol = 2)


plot_lgb <- cv_long %>% 
  filter(learner == 'lg', encoder != 'se', !grepl('house', data)) %>% 
  ggplot(aes(x = test_score, fill = encoder, color = encoder)) + 
  # geom_histogram(alpha = 0.3, bins = 5, position = 'dodge') +
  geom_density(alpha = 0.3) + 
  geom_rug() + 
  facet_wrap(~data, scales = 'free', ncol = 2)

plot_lgb



# Plot differences --------------------------------------------------------


score_diffs <- cv_long %>% 
  filter(encoder != 'se') %>% 
  arrange(desc(encoder)) %>% 
  group_by(learner, data, split) %>% 
  summarise(
    score_difference = first(test_score) - last(test_score)
  ) %>% 
  ungroup()

score_diffs %>% 
  filter(learner == 'lm', !grepl('house', data)) %>% 
  ggplot(aes(x = score_difference)) + 
  geom_density() + 
  geom_rug() + 
  geom_vline(aes(xintercept = 0, color = 'No difference')) +
  facet_wrap(~data, scales = 'free', ncol = 2) + 
  xlab('Cross validation difference') + 
  labs(colour = '') + 
  ggtitle("Linear model cross-validation results", 
          "Score difference between target and quantile encodings \nPositive differences indicate quantile method being better") +
  ggsave("results_regression/lm_cv_differences.png")
  
score_diffs %>% 
  filter(learner == 'lg', !grepl('house', data)) %>% 
  ggplot(aes(x = score_difference)) + 
  geom_density() + 
  geom_rug() + 
  geom_vline(aes(xintercept = 0, color = 'No difference')) +
  facet_wrap(~data, scales = 'free', ncol = 2) + 
  xlab('Cross validation difference') + 
  labs(colour = '') + 
  ggtitle("Lightgbm model cross-validation results", 
          "Score difference between target and quantile encodings \nPositive differences indicate quantile method being better") +
  ggsave("results_regression/lg_cv_differences.png")


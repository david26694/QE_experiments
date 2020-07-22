library(dplyr)
library(ggplot2)
library(tidyr)

theme_set(theme_minimal())

cv_results <- read.csv('results_regression/cv_results.csv')

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

cv_test_prep <- cv_long %>% 
  # There are some repeated rows
  group_by(learner, encoder, data, split) %>% 
  summarise_all(last) %>% 
  ungroup() %>% 
  pivot_wider(
    names_from = "encoder",
    values_from = "test_score"
    )

cv_test_prep %>% 
  filter(!grepl('house_', data)) %>% 
  group_by(learner, data) %>% 
  summarise(
    p_value = unlist(wilcox.test(
      te, 
      pe, 
      alternative = 'greater', 
      paired = TRUE,
      exact = TRUE
      )["p.value"])
  )

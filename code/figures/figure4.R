# figure4.R
# Performance by tract size: all embedding models vs. all social risk indices.
# Shows how average R2 across 40 PLACES health outcomes varies across the
# urban-to-rural gradient (tract area deciles).
#
# Usage: Rscript code/figures/figure4.R

library(tidyverse)

REPO    <- getwd()
OUTPUTS <- file.path(REPO, "outputs")
FIG_DIR <- file.path(OUTPUTS, "figures")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

# ── Model registry (mirrors Figure 3) ─────────────────────────────────────────
REAL_PATHS <- c(
  "AlphaEarth\nFoundations"  = file.path(OUTPUTS, "alphaearth_foundations", "q4_decile.csv"),
  "Prithvi-EO-2.0\nTiny-TL"  = file.path(OUTPUTS, "prithvi_tiny",          "prithvi_tiny_q4_decile.csv"),
  "Prithvi-EO-2.0\n300M-TL"  = file.path(OUTPUTS, "prithvi_300M-TL",       "q4_decile.csv"),
  "OlmoEarth-1.2\nNano"      = file.path(OUTPUTS, "olmoearth_nano_pca64",  "q4_decile.csv"),
  "Clay-v1.5"                = file.path(OUTPUTS, "clay_pca64",           "q4_decile.csv")
)

MOCK_MODELS <- c("OlmoEarth-1.2\nBase")
MOCK_SEEDS  <- setNames(c(202L), MOCK_MODELS)
MOCK_SCALES <- setNames(c(1.60), MOCK_MODELS)
MOCK_SD     <- 0.013

MODEL_ORDER <- c(
  "AlphaEarth\nFoundations",
  "Prithvi-EO-2.0\nTiny-TL",
  "Prithvi-EO-2.0\n300M-TL",
  "OlmoEarth-1.2\nNano",
  "OlmoEarth-1.2\nBase",
  "Clay-v1.5"
)

# Okabe-Ito palette -- same as Figure 3
MODEL_COLORS <- c(
  "AlphaEarth\nFoundations"  = "#0072B2",
  "Prithvi-EO-2.0\nTiny-TL"  = "#009E73",
  "Prithvi-EO-2.0\n300M-TL"  = "#D55E00",
  "OlmoEarth-1.2\nNano"      = "#E69F00",
  "OlmoEarth-1.2\nBase"      = "#CC79A7",
  "Clay-v1.5"                = "#56B4E9"
)

MODEL_LEGEND <- setNames(
  c(
    "AlphaEarth\nFoundations",
    "Prithvi-EO-2.0\nTiny-TL",
    "Prithvi-EO-2.0\n300M-TL",
    "OlmoEarth-1.2\nNano",
    "OlmoEarth-1.2\nBase *",
    "Clay-v1.5"
  ),
  MODEL_ORDER
)

# ── Load real model data ───────────────────────────────────────────────────────
real_data <- imap_dfr(as.list(REAL_PATHS), function(path, model) {
  read_csv(path, show_col_types = FALSE) %>%
    mutate(model = model, model_type = "real")
})

# Reference: average area and index values across all real model evaluation splits
reference <- real_data %>%
  group_by(decile) %>%
  summarise(
    median_aland_km2 = mean(median_aland_km2),
    r2_readi         = mean(r2_readi),
    r2_svi           = mean(r2_svi),
    r2_sdi           = mean(r2_sdi),
    .groups          = "drop"
  )

# ── Generate mock embedding model data ────────────────────────────────────────
# Scaled from Prithvi-EO-2.0 Tiny-TL baseline with added noise.
# These are illustrative placeholders only.
prithvi_tiny_base <- real_data %>%
  filter(model == "Prithvi-EO-2.0\nTiny-TL") %>%
  select(decile, r2_emb)

mock_data <- imap_dfr(as.list(MOCK_SEEDS), function(seed, model) {
  set.seed(seed)
  prithvi_tiny_base %>%
    mutate(
      r2_emb     = pmax(0.05, pmin(0.80, r2_emb * MOCK_SCALES[model] + rnorm(n(), 0, MOCK_SD))),
      model      = model,
      model_type = "mock"
    ) %>%
    left_join(reference %>% select(decile, median_aland_km2), by = "decile")
})

# ── Combine and shape ─────────────────────────────────────────────────────────
emb_data <- bind_rows(
  real_data %>% select(decile, median_aland_km2, r2_emb, model, model_type),
  mock_data
) %>%
  mutate(model = factor(model, levels = MODEL_ORDER))

index_long <- reference %>%
  pivot_longer(c(r2_readi, r2_sdi, r2_svi), names_to = "index_col", values_to = "r2") %>%
  mutate(
    index = recode(index_col, r2_readi = "ReADI", r2_sdi = "SDI", r2_svi = "SVI"),
    index = factor(index, levels = c("ReADI", "SDI", "SVI"))
  )

x_labels <- reference %>%
  mutate(lbl = case_when(
    median_aland_km2 < 1   ~ paste0(round(median_aland_km2 * 100), " ha"),
    median_aland_km2 < 10  ~ paste0(sprintf("%.1f", median_aland_km2), " km²"),
    median_aland_km2 < 100 ~ paste0(round(median_aland_km2), " km²"),
    TRUE                   ~ paste0(round(median_aland_km2 / 10) * 10, " km²")
  )) %>%
  pull(lbl)

# ── Build figure ───────────────────────────────────────────────────────────────
p <- ggplot() +
  # Social risk index reference lines: black with distinct dash patterns
  geom_line(
    data = index_long,
    aes(x = decile, y = r2, linetype = index),
    color = "black", linewidth = 0.85
  ) +
  # Embedding model lines (real: full opacity; mock: 45% opacity via alpha scale)
  geom_line(
    data = emb_data,
    aes(x = decile, y = r2_emb, color = model, alpha = model_type),
    linewidth = 1.05
  ) +
  # Real model points (filled circles)
  geom_point(
    data = filter(emb_data, model_type == "real"),
    aes(x = decile, y = r2_emb, color = model),
    size = 2.5, shape = 19
  ) +
  # Mock model points (open circles)
  geom_point(
    data = filter(emb_data, model_type == "mock"),
    aes(x = decile, y = r2_emb, color = model),
    size = 2.5, shape = 1, stroke = 0.9, alpha = 0.55
  ) +
  scale_color_manual(
    values = MODEL_COLORS,
    labels = MODEL_LEGEND,
    name   = "Geospatial Foundation Models"
  ) +
  scale_alpha_manual(
    values = c("real" = 1.0, "mock" = 0.45),
    guide  = "none"
  ) +
  scale_linetype_manual(
    values = c("ReADI" = "longdash", "SDI" = "dashed", "SVI" = "dotted"),
    name   = "Social risk index"
  ) +
  scale_x_continuous(
    breaks = 1:10,
    labels = x_labels,
    expand = expansion(add = 0.3)
  ) +
  scale_y_continuous(
    name         = expression(atop("Avg " * R^2 * "  across 40 PLACES outcomes", "(held-out states, by tract-size decile)")),
    limits       = c(0, 0.6),
    breaks       = seq(0, 0.6, by = 0.2),
    minor_breaks = c(0.1, 0.3, 0.5),
    expand       = expansion(add = 0)
  ) +
  labs(
    x       = "Median tract area  (D1 = most urban, D10 = most rural)",
    caption = "* Placeholder values; embedding model not yet processed."
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x        = element_text(angle = 30, hjust = 1, size = 9),
    legend.position       = "bottom",
    legend.box            = "horizontal",
    legend.box.just = "center",
    legend.background   = element_rect(color = "grey40", fill = "white", linewidth = 0.4),
    legend.margin       = margin(4, 6, 4, 6),
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_line(color = "grey91", linewidth = 0.3),
    panel.grid.major.x = element_line(color = "grey91"),
    plot.caption       = element_text(size = 8.5, color = "grey50"),
    plot.margin        = margin(8, 8, 8, 14)
  )

ggsave(file.path(FIG_DIR, "figure4_tract_size.png"), p, width = 11, height = 5.5, dpi = 300)
message("Saved: ", file.path(FIG_DIR, "figure4_tract_size.png"))

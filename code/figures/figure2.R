# Figure 2: per-outcome R² comparison across predictor types and embedding models.
# Requires: tidyverse, patchwork
#
# Usage (from repo root):
#   Rscript code/figures/figure2.R                     # CHECKUP (default)
#   Rscript code/figures/figure2.R --outcome DIABETES
#   Rscript code/figures/figure2.R --main              # main composite (4 outcomes)
#   Rscript code/figures/figure2.R --supp 1            # supplementary page 1
#   Rscript code/figures/figure2.R --supp 2            # supplementary page 2

library(tidyverse)
library(patchwork)

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO    <- getwd()   # run from repo root: Rscript code/figures/figure2.R
OUTPUTS <- file.path(REPO, "outputs")
FIG_DIR <- file.path(OUTPUTS, "figures")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

INDEX_CSV <- file.path(OUTPUTS, "prithvi_tiny", "prithvi_tiny_places_reg.csv")

# ── Model registry ─────────────────────────────────────────────────────────────
# Set value to NULL for models whose embeddings are not yet complete.
MODEL_SOURCES <- list(
  "AlphaEarth"         = file.path(OUTPUTS, "alphaearth_foundations", "places_reg.csv"),
  "Prithvi Tiny-TL"    = file.path(OUTPUTS, "prithvi_tiny", "prithvi_tiny_places_reg.csv"),
  "Prithvi 300M-TL"    = file.path(OUTPUTS, "prithvi_300M-TL", "places_reg.csv"),
  "OlmoEarth-1.1 Nano" = NULL,
  "OlmoEarth-1.1 Base" = NULL,
  "Clay v1.5"          = NULL
)

MOCK_SEEDS <- c("OlmoEarth-1.1 Nano" = 101L, "OlmoEarth-1.1 Base" = 202L, "Clay v1.5" = 303L)

MODEL_LABELS <- c(
  "AlphaEarth"         = "AlphaEarth\nFoundations",
  "Prithvi Tiny-TL"    = "Prithvi-EO-2.0\nTiny-TL",
  "Prithvi 300M-TL"    = "Prithvi-EO-2.0\n300M-TL",
  "OlmoEarth-1.1 Nano" = "OlmoEarth-1.1\nNano",
  "OlmoEarth-1.1 Base" = "OlmoEarth-1.1\nBase",
  "Clay v1.5"          = "Clay-v1.5"
)

# ── Outcome labels ─────────────────────────────────────────────────────────────
OUTCOME_LABELS <- c(
  ACCESS2 = "Lack of health insurance",    ARTHRITIS   = "Arthritis",
  BINGE   = "Binge drinking",              BPHIGH      = "High blood pressure",
  BPMED   = "BP medication use",           CANCER      = "Cancer (excl. skin)",
  CASTHMA = "Current asthma",              CHD         = "Coronary heart disease",
  CHECKUP = "Annual checkup",              CHOLSCREEN  = "Cholesterol screening",
  COGNITION = "Cognitive decline",         COLON_SCREEN = "Colorectal cancer screening",
  COPD    = "COPD",                        CSMOKING    = "Current smoking",
  DENTAL  = "Dental visit",               DEPRESSION  = "Depression",
  DIABETES = "Diabetes",                   DISABILITY  = "Any disability",
  EMOTIONSPT = "Emotional support",        FOODINSECU  = "Food insecurity",
  FOODSTAMP  = "Food stamp use",           GHLTH       = "Fair/poor general health",
  HEARING = "Hearing disability",          HIGHCHOL    = "High cholesterol",
  HOUSINSECU = "Housing insecurity",       INDEPLIVE   = "Independent living difficulty",
  LACKTRPT   = "Lack of transportation",   LONELINESS  = "Loneliness",
  LPA     = "Physical inactivity",         MAMMOUSE    = "Mammography use",
  MHLTH   = "Poor mental health days",     MOBILITY    = "Mobility disability",
  OBESITY = "Obesity",                     PHLTH       = "Poor physical health days",
  SELFCARE = "Self-care disability",       SHUTUTILITY = "Utility shutoff",
  SLEEP   = "Short sleep duration",        STROKE      = "Stroke",
  TEETHLOST = "Tooth loss",               VISION      = "Vision disability"
)

# ── Colorblind-safe sequential palettes (Oranges / Blues / Purples) ────────────
PALETTES <- list(
  "Indices Alone"                    = colorRampPalette(c("#FEE6CE", "#7F2704"))(100),
  "Embeddings Alone"                 = colorRampPalette(c("#DEEBF7", "#08306B"))(100),
  "Embeddings With Combined Indices" = colorRampPalette(c("#EFEDF5", "#3F007D"))(100)
)

PANEL_LEVELS <- c("Indices Alone", "Embeddings Alone", "Embeddings With Combined Indices")

# ── Fixed display orders (bottom → top within each panel) ─────────────────────
# Colors are assigned by position in these vectors, not by per-outcome R².
# This keeps a given model at the same height with the same color across all
# outcomes, so readers can compare panels without re-orienting.
INDEX_ORDER <- c("ReADI", "SVI", "SDI", "All indices")

EMB_ORDER <- c(
  "Prithvi-EO-2.0\n300M-TL",
  "Prithvi-EO-2.0\nTiny-TL",
  "OlmoEarth-1.1\nNano",
  "OlmoEarth-1.1\nBase",
  "Clay-v1.5",
  "AlphaEarth\nFoundations"
)

# Pre-compute one color per position in each order (lo=0.38 → hi=0.88 of palette).
.mk_color_map <- function(order_vec, palette_vec) {
  n   <- length(order_vec)
  idx <- round(seq(0.38 * 99 + 1, 0.88 * 99 + 1, length.out = n))
  setNames(palette_vec[idx], order_vec)
}
IDX_COLORS  <- .mk_color_map(INDEX_ORDER, PALETTES[["Indices Alone"]])
EMB_COLORS  <- .mk_color_map(EMB_ORDER,   PALETTES[["Embeddings Alone"]])
COMB_COLORS <- .mk_color_map(EMB_ORDER,   PALETTES[["Embeddings With Combined Indices"]])

# ── Data assembly ──────────────────────────────────────────────────────────────
build_data <- function(outcome) {
  # Index baseline (Prithvi split as canonical)
  idx <- read_csv(INDEX_CSV, show_col_types = FALSE) |>
    filter(outcome == !!outcome,
           model %in% c("ReADI", "SVI", "SDI", "All indices")) |>
    transmute(model, r2, panel = "Indices Alone", mocked = FALSE)

  # Real embedding models
  real <- imap_dfr(MODEL_SOURCES, function(path, name) {
    if (is.null(path)) return(NULL)
    read_csv(path, show_col_types = FALSE) |>
      filter(outcome == !!outcome,
             model %in% c("Embeddings", "All indices + Embeddings")) |>
      transmute(model_type = model, r2, model_name = name)
  })

  # Calibrate mock range from real values
  emb_vals  <- real$r2[real$model_type == "Embeddings"]
  comb_vals <- real$r2[real$model_type == "All indices + Embeddings"]
  emb_lo  <- min(emb_vals)  * 0.85;  emb_hi  <- max(emb_vals)  * 1.10
  comb_lo <- min(comb_vals) * 0.90;  comb_hi <- max(comb_vals) * 1.05

  # Mocked models
  mock_names <- names(Filter(is.null, MODEL_SOURCES))
  mock <- map_dfr(mock_names, function(name) {
    seed <- MOCK_SEEDS[[name]]
    set.seed(seed);     emb_r2  <- runif(1, emb_lo,  emb_hi)
    set.seed(seed + 1L); comb_r2 <- runif(1, comb_lo, comb_hi)
    tibble(model_type = c("Embeddings", "All indices + Embeddings"),
           r2 = c(emb_r2, comb_r2), model_name = name)
  })

  emb <- bind_rows(real, mock) |>
    mutate(
      mocked = model_name %in% mock_names,
      panel  = if_else(model_type == "Embeddings",
                       "Embeddings Alone",
                       "Embeddings With Combined Indices"),
      model  = model_name
    ) |>
    select(model, r2, panel, mocked)

  bind_rows(idx, emb) |>
    mutate(
      panel = factor(panel, levels = PANEL_LEVELS),
      label = if_else(mocked,
                      paste0(coalesce(MODEL_LABELS[model], model), " *"),
                      coalesce(MODEL_LABELS[model], model))
    )
}

# ── Fixed colors and y ordering ───────────────────────────────────────────────
# Colors and positions are determined by global order constants, not per-outcome
# R², so every model appears at the same height with the same color in every panel.
prepare_plot_data <- function(df) {
  df |>
    mutate(
      base_label = str_remove(label, " \\*$"),
      y_pos = case_when(
        panel == "Indices Alone" ~ match(base_label, INDEX_ORDER),
        TRUE                     ~ match(base_label, EMB_ORDER)
      ),
      fill_color = case_when(
        panel == "Indices Alone"                    ~ IDX_COLORS[base_label],
        panel == "Embeddings Alone"                 ~ EMB_COLORS[base_label],
        panel == "Embeddings With Combined Indices" ~ COMB_COLORS[base_label]
      ),
      y_key = paste0(as.integer(panel), "__", str_pad(y_pos, 2, pad = "0"), "__", label)
    ) |>
    arrange(y_key) |>
    mutate(y_key = factor(y_key, levels = unique(y_key)))
}

# ── Core panel builder (returns ggplot, does not save) ─────────────────────────
# x_limits: c(lo, hi); NULL = auto-compute from this outcome's data.
# show_y:   show y-axis bar labels (set FALSE for non-leftmost columns in composites).
# show_x_lbl: show x-axis title (set FALSE for non-bottom rows in supplementary).
# compact:  tight sizing for 4×5 supplementary layout.
make_panel <- function(
    outcome,
    x_limits   = NULL,
    show_y     = TRUE,
    show_x_lbl = TRUE,
    compact    = FALSE
) {
  d <- build_data(outcome) |> prepare_plot_data()

  outcome_label <- unname(OUTCOME_LABELS[outcome])
  if (is.na(outcome_label)) outcome_label <- str_to_title(str_replace_all(outcome, "_", " "))

  label_map <- set_names(d$label, as.character(d$y_key))

  if (is.null(x_limits)) {
    x_min <- min(d$r2, 0) - 0.02
    x_max <- max(d$r2) * 1.08
  } else {
    x_min <- x_limits[1]
    x_max <- x_limits[2]
  }

  # All size/spacing parameters scale with compact setting
  bs      <- if (compact) 6    else 10
  bar_w   <- if (compact) 0.60 else 0.68
  bar_lw  <- if (compact) 0.20 else 0.30
  strip_s <- if (compact) 5.5  else 9.5
  yax_s   <- if (compact) 4.5  else 8.5
  xax_s   <- if (compact) 4.5  else 8.0
  xttl_s  <- if (compact) 5.0  else 9.0
  ttl_s   <- if (compact) 6.0  else 12.0
  spc_cm  <- if (compact) 0.15 else 0.35
  str_mg  <- if (compact) 2    else 4
  mg      <- if (compact) c(2, 3, 2, 2) else c(6, 10, 6, 4)  # t, r, b, l

  # In compact mode, shorten strip labels so rotated text fits within sub-panel height.
  panel_labeller <- if (compact) {
    as_labeller(c(
      "Indices Alone"                    = "Indices",
      "Embeddings Alone"                 = "Embeddings",
      "Embeddings With Combined Indices" = "Combined"
    ))
  } else {
    label_value
  }

  # Reference line: combined-indices baseline (drawn in every sub-panel)
  all_idx_r2 <- d$r2[d$panel == "Indices Alone" & d$label == "All indices"]
  all_idx_r2 <- if (length(all_idx_r2) == 1L) all_idx_r2[1] else NA_real_

  p <- ggplot(d, aes(x = r2, y = y_key, fill = fill_color)) +
    geom_col(width = bar_w, color = "#666666", linewidth = bar_lw) +
    geom_vline(xintercept = 0, color = "#555555", linewidth = 0.45, linetype = "dashed") +
    {if (!is.na(all_idx_r2))
      geom_vline(xintercept = all_idx_r2, color = "#B05000",
                 linewidth = 0.8, linetype = "dotted")} +
    facet_grid(panel ~ ., scales = "free_y", space = "free_y", switch = "y",
               labeller = panel_labeller) +
    scale_fill_identity() +
    scale_y_discrete(labels = label_map) +
    scale_x_continuous(
      limits = c(x_min, x_max),
      labels = scales::label_number(accuracy = 0.01),
      expand = expansion(mult = c(0, 0.02))
    ) +
    labs(
      title = outcome_label,
      x     = if (show_x_lbl) expression(R^2 ~ "(held-out states)") else NULL,
      y     = NULL
    ) +
    theme_minimal(base_size = bs) +
    theme(
      plot.title           = element_text(face = "bold", size = ttl_s, hjust = 0.5,
                                          margin = margin(b = if (compact) 2 else 8)),
      plot.title.position  = "plot",
      strip.text           = element_text(face = "bold", size = strip_s, hjust = 0.5,
                                          color = "black",
                                          margin = margin(t = str_mg, b = str_mg,
                                                          l = str_mg, r = str_mg)),
      strip.background     = element_rect(fill = "#f5f5f5", color = "black",
                                          linewidth = if (compact) 0.4 else 0.6),
      strip.clip           = "off",
      panel.border         = element_rect(color = "black", fill = NA,
                                          linewidth = if (compact) 0.4 else 0.6),
      panel.grid.major.y   = element_blank(),
      panel.grid.major.x   = element_line(color = "#ebebeb",
                                          linewidth = if (compact) 0.25 else 0.40),
      panel.grid.minor     = element_blank(),
      panel.spacing        = unit(spc_cm, "cm"),
      axis.text.y          = if (show_y) element_text(size = yax_s, color = "black")
                             else element_blank(),
      axis.ticks.y         = element_blank(),
      axis.text.x          = element_text(size = xax_s, color = "black"),
      axis.title.x         = element_text(size = xttl_s, color = "black",
                                          margin = margin(t = if (compact) 2 else 6)),
      plot.margin          = margin(t = mg[1], r = mg[2], b = mg[3], l = mg[4])
    )

  p
}

# ── Single-outcome figure (wraps make_panel, saves to disk) ───────────────────
figure2 <- function(outcome = "CHECKUP") {
  p <- make_panel(outcome)
  d <- build_data(outcome)

  # Proportional figure height: 0.38 in per bar row + 0.5 in per strip + margins
  n_rows   <- nrow(d)
  n_panels <- length(PANEL_LEVELS)
  fig_h <- n_rows * 0.38 + n_panels * 0.5 + 1.4

  out <- file.path(FIG_DIR, paste0("figure2_", outcome, ".png"))
  ggsave(out, p, width = 7.5, height = fig_h, dpi = 300, bg = "white")
  message("Saved → ", out)
  invisible(p)
}

# ── Global x limits across a set of outcomes ──────────────────────────────────
# Used by composite builders to enforce a shared x scale across all panels.
.global_xlim <- function(outcomes) {
  all_r2 <- unlist(lapply(outcomes, function(o) build_data(o)$r2))
  c(min(all_r2, 0) - 0.02, max(all_r2) * 1.08)
}

# ── All outcomes sorted ascending by AlphaEarth R² for a given model type ────
.sorted_outcomes <- function(sort_by = "All indices + Embeddings") {
  read_csv(
    file.path(OUTPUTS, "alphaearth_foundations", "places_reg.csv"),
    show_col_types = FALSE
  ) |>
    filter(model == sort_by) |>
    arrange(r2) |>
    pull(outcome)
}

# ── Main figure (4 outcomes at 10/30/70/90th percentile) ──────────────────────
# Outcomes selected by AlphaEarth combined R² percentile rank; 1 row × 4 cols.
figure2_main <- function() {
  all_oc <- .sorted_outcomes(sort_by = "Embeddings")
  n      <- length(all_oc)
  sel_oc <- all_oc[pmax(1L, pmin(n, round(c(0.10, 0.30, 0.70, 0.90) * n)))]

  message("Main figure outcomes: ", paste(sel_oc, collapse = ", "))

  xlim <- .global_xlim(sel_oc)
  message("Shared x range: [", round(xlim[1], 3), ", ", round(xlim[2], 3), "]")

  plots <- imap(sel_oc, function(oc, i) {
    make_panel(oc, x_limits = xlim, show_y = (i == 1L), show_x_lbl = TRUE, compact = FALSE)
  })

  composite <- wrap_plots(plots, nrow = 1) +
    plot_annotation(
      caption = "* placeholder — embedding run not yet complete",
      theme   = theme(
        plot.caption = element_text(size = 7.5, color = "#888888", face = "italic",
                                    hjust = 0, margin = margin(t = 4))
      )
    )

  d_sample <- build_data(sel_oc[1])
  fig_h    <- nrow(d_sample) * 0.38 + length(PANEL_LEVELS) * 0.5 + 1.4

  out <- file.path(FIG_DIR, "figure2_main.png")
  ggsave(out, composite, width = 10, height = fig_h, dpi = 300, bg = "white")
  message("Saved → ", out)
  invisible(composite)
}

# ── Supplementary figures (20 outcomes, 4 cols × 5 rows, letter paper) ────────
# page=1 → outcomes 1–20 (lowest AlphaEarth combined R²)
# page=2 → outcomes 21–40
figure2_supp <- function(page = 1) {
  all_oc   <- .sorted_outcomes()
  start    <- (page - 1L) * 20L + 1L
  end      <- min(start + 19L, length(all_oc))
  these_oc <- all_oc[start:end]

  message("Supp page ", page, " outcomes (", start, "–", end, "): ",
          paste(these_oc, collapse = ", "))

  xlim <- .global_xlim(these_oc)
  message("Shared x range: [", round(xlim[1], 3), ", ", round(xlim[2], 3), "]")

  n_cols      <- 4L
  n_rows_grid <- ceiling(length(these_oc) / n_cols)

  plots <- imap(these_oc, function(oc, i) {
    col_i <- ((i - 1L) %% n_cols) + 1L
    row_i <- ((i - 1L) %/% n_cols) + 1L
    make_panel(
      oc,
      x_limits   = xlim,
      show_y     = (col_i == 1L),
      show_x_lbl = (row_i == n_rows_grid),
      compact    = TRUE
    )
  })

  composite <- wrap_plots(plots, nrow = n_rows_grid, ncol = n_cols) +
    plot_annotation(
      caption = "* placeholder — embedding run not yet complete",
      theme   = theme(
        plot.caption = element_text(size = 6, color = "#888888", face = "italic",
                                    hjust = 0, margin = margin(t = 3))
      )
    )

  out <- file.path(FIG_DIR, paste0("figure2_supp_", page, ".png"))
  ggsave(out, composite, width = 8.5, height = 11, dpi = 300, bg = "white")
  message("Saved → ", out)
  invisible(composite)
}

# ── CLI ────────────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)

if ("--main" %in% args) {
  figure2_main()
} else if ("--supp" %in% args) {
  supp_i <- which(args == "--supp") + 1L
  pg     <- if (length(args) >= supp_i) as.integer(args[supp_i]) else 1L
  figure2_supp(pg)
} else {
  outcome <- "CHECKUP"
  flag_i  <- which(args == "--outcome")
  if (length(flag_i) > 0 && length(args) >= flag_i + 1L) outcome <- args[flag_i + 1L]
  figure2(outcome)
}

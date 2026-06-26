# Figure 3: Incremental ΔR² by PLACES outcome and embedding model.
# Each bar shows the absolute additional variance in a PLACES outcome explained
# by the embeddings beyond what a social risk index alone captures.
#
# Usage (from repo root):
#   Rscript code/figures/figure3.R                  # ReADI single panel (default)
#   Rscript code/figures/figure3.R --index SDI
#   Rscript code/figures/figure3.R --index SVI
#   Rscript code/figures/figure3.R --all            # all three indices, shared order

library(tidyverse)
library(patchwork)

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO    <- getwd()
OUTPUTS <- file.path(REPO, "outputs")
FIG_DIR <- file.path(OUTPUTS, "figures")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

# ── Model registry ─────────────────────────────────────────────────────────────
MODEL_SOURCES <- list(
  "AlphaEarth"         = file.path(OUTPUTS, "alphaearth_foundations", "places_residual_by_index.csv"),
  "Prithvi Tiny-TL"    = file.path(OUTPUTS, "prithvi_tiny",           "prithvi_tiny_places_residual_by_index.csv"),
  "Prithvi 300M-TL"    = file.path(OUTPUTS, "prithvi_300M-TL",        "places_residual_by_index.csv"),
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

MODEL_ORDER <- c(
  "AlphaEarth\nFoundations",
  "Prithvi-EO-2.0\nTiny-TL",
  "Prithvi-EO-2.0\n300M-TL",
  "OlmoEarth-1.1\nNano",
  "OlmoEarth-1.1\nBase",
  "Clay-v1.5"
)

# ── Okabe-Ito colorblind-safe palette ─────────────────────────────────────────
MODEL_COLORS <- c(
  "AlphaEarth\nFoundations" = "#0072B2",
  "Prithvi-EO-2.0\nTiny-TL" = "#009E73",
  "Prithvi-EO-2.0\n300M-TL" = "#D55E00",
  "OlmoEarth-1.1\nNano"     = "#E69F00",
  "OlmoEarth-1.1\nBase"     = "#CC79A7",
  "Clay-v1.5"               = "#56B4E9"
)

# ── Outcome labels ─────────────────────────────────────────────────────────────
OUTCOME_LABELS <- c(
  ACCESS2    = "Lack of health insurance",    ARTHRITIS    = "Arthritis",
  BINGE      = "Binge drinking",              BPHIGH       = "High blood pressure",
  BPMED      = "BP medication use",           CANCER       = "Cancer (excl. skin)",
  CASTHMA    = "Current asthma",              CHD          = "Coronary heart disease",
  CHECKUP    = "Annual checkup",              CHOLSCREEN   = "Cholesterol screening",
  COGNITION  = "Cognitive decline",           COLON_SCREEN = "Colorectal cancer screening",
  COPD       = "COPD",                        CSMOKING     = "Current smoking",
  DENTAL     = "Dental visit",               DEPRESSION   = "Depression",
  DIABETES   = "Diabetes",                   DISABILITY   = "Any disability",
  EMOTIONSPT = "Emotional support",           FOODINSECU   = "Food insecurity",
  FOODSTAMP  = "Food stamp use",              GHLTH        = "Fair/poor general health",
  HEARING    = "Hearing disability",          HIGHCHOL     = "High cholesterol",
  HOUSINSECU = "Housing insecurity",          INDEPLIVE    = "Independent living difficulty",
  LACKTRPT   = "Lack of transportation",      LONELINESS   = "Loneliness",
  LPA        = "Physical inactivity",         MAMMOUSE     = "Mammography use",
  MHLTH      = "Poor mental health days",     MOBILITY     = "Mobility disability",
  OBESITY    = "Obesity",                     PHLTH        = "Poor physical health days",
  SELFCARE   = "Self-care disability",        SHUTUTILITY  = "Utility shutoff",
  SLEEP      = "Short sleep duration",        STROKE       = "Stroke",
  TEETHLOST  = "Tooth loss",                 VISION       = "Vision disability"
)


# ── Canonical outcome order (ReADI × AlphaEarth, ascending so top = highest) ──
canonical_order <- function() {
  read_csv(MODEL_SOURCES[["AlphaEarth"]], show_col_types = FALSE) |>
    filter(index == "ReADI") |>
    arrange(additional_var) |>
    pull(outcome)
}


# ── Data assembly ──────────────────────────────────────────────────────────────
# outcome_order_asc: character vector of outcome codes, ascending ΔR²
# (lowest first → appears at bottom of chart; highest last → appears at top).
# When NULL, computed from this index's AlphaEarth data.
build_data <- function(index, outcome_order_asc = NULL) {
  mock_names <- names(Filter(is.null, MODEL_SOURCES))

  real <- imap_dfr(MODEL_SOURCES, function(path, name) {
    if (is.null(path)) return(NULL)
    read_csv(path, show_col_types = FALSE) |>
      filter(index == !!index) |>
      transmute(
        outcome,
        additional_var,
        model_key   = name,
        model_label = MODEL_LABELS[[name]],
        mocked      = FALSE
      )
  })

  if (is.null(outcome_order_asc)) {
    outcome_order_asc <- real |>
      filter(model_key == "AlphaEarth") |>
      arrange(additional_var) |>
      pull(outcome)
  }

  # Keep only the outcomes we intend to display
  real    <- real |> filter(outcome %in% outcome_order_asc)

  mock_lo <- min(real$additional_var) * 0.85
  mock_hi <- max(real$additional_var) * 1.05

  mock <- map_dfr(mock_names, function(name) {
    seed <- MOCK_SEEDS[[name]]
    set.seed(seed)
    tibble(
      outcome        = outcome_order_asc,
      additional_var = runif(length(outcome_order_asc), mock_lo, mock_hi),
      model_key      = name,
      model_label    = MODEL_LABELS[[name]],
      mocked         = TRUE
    )
  })

  bind_rows(real, mock) |>
    mutate(
      outcome_label = coalesce(OUTCOME_LABELS[outcome], outcome),
      outcome_label = factor(outcome_label,
                             levels = coalesce(OUTCOME_LABELS[outcome_order_asc],
                                               outcome_order_asc)),
      model_label   = factor(model_label, levels = MODEL_ORDER)
    )
}


# ── Core panel builder ─────────────────────────────────────────────────────────
# outcome_order_asc: pre-computed order to pass through to build_data
# x_limits: shared c(lo, hi); NULL = auto from this panel's data
# show_y: whether to draw outcome labels (FALSE for non-leftmost panels)
# show_x_lbl: whether to draw the x-axis title (set TRUE for one panel only)
# compact: TRUE = dense layout for supplementary; FALSE = full size for main paper
make_panel <- function(
    index,
    outcome_order_asc = NULL,
    x_limits          = NULL,
    show_y            = TRUE,
    show_x_lbl        = TRUE,
    compact           = FALSE
) {
  d <- build_data(index, outcome_order_asc)

  if (is.null(x_limits)) {
    x_max <- max(d$additional_var) * 1.08
    x_min <- min(min(d$additional_var), 0) - 0.01
  } else {
    x_min <- x_limits[1]
    x_max <- x_limits[2]
  }

  # Size tokens — match figure2.R compact/full convention
  ttl_s  <- if (compact) 10   else 14
  yax_s  <- if (compact) 8.5  else 12
  xax_s  <- if (compact) 8.0  else 11
  xttl_s <- if (compact) 9.0  else 12
  lgd_s  <- if (compact) 8    else 11
  bs     <- if (compact) 10   else 13
  lw     <- if (compact) 0.40 else 0.55

  y_labels <- if (compact) {
    function(x) x
  } else {
    function(x) str_wrap(x, width = 12)
  }

  ggplot(d, aes(
    x     = additional_var,
    y     = outcome_label,
    fill  = model_label,
    alpha = mocked
  )) +
    geom_col(
      position = position_dodge2(width = 0.85, padding = 0.15, reverse = TRUE),
      color    = NA,
      width    = 0.85
    ) +
    geom_vline(xintercept = 0, color = "#555555", linewidth = 0.45,
               linetype = "dashed") +
    scale_fill_manual(
      values = MODEL_COLORS,
      labels = setNames(MODEL_ORDER, MODEL_ORDER),
      guide  = guide_legend(
        title        = NULL,
        ncol         = 3,
        byrow        = TRUE,
        override.aes = list(alpha = c(0.88, 0.88, 0.88, 0.38, 0.38, 0.38))
      )
    ) +
    scale_alpha_manual(
      values = c("FALSE" = 0.88, "TRUE" = 0.38),
      guide  = "none"
    ) +
    scale_x_continuous(
      limits = c(x_min, x_max),
      labels = scales::label_number(accuracy = 0.01),
      expand = expansion(mult = c(0, 0.02))
    ) +
    scale_y_discrete(labels = y_labels) +
    labs(
      title = index,
      x     = if (show_x_lbl) expression(Delta * R^2 ~ "(additional variance beyond index alone)") else NULL,
      y     = NULL
    ) +
    theme_minimal(base_size = bs) +
    theme(
      plot.title          = element_text(face = "bold", size = ttl_s, hjust = 0.5,
                                         margin = margin(b = 8)),
      plot.title.position = "plot",
      legend.text         = element_text(size = lgd_s, lineheight = 0.85),
      legend.key.size     = unit(if (compact) 0.45 else 0.55, "cm"),
      legend.margin       = margin(t = 4),
      panel.border        = element_rect(color = "black", fill = NA, linewidth = 0.6),
      panel.grid.major.y  = element_blank(),
      panel.grid.major.x  = element_line(color = "#ebebeb", linewidth = lw),
      panel.grid.minor    = element_blank(),
      axis.text.y         = if (show_y) element_text(size = yax_s, color = "black",
                                                      lineheight = 0.90)
                            else element_blank(),
      axis.ticks.y        = element_blank(),
      axis.text.x         = element_text(size = xax_s, color = "black",
                                         angle = 90, hjust = 1, vjust = 0.5),
      axis.title.x        = element_text(size = xttl_s, color = "black",
                                         margin = margin(t = 6)),
      plot.margin         = margin(t = 6, r = 10, b = 6, l = 4)
    )
}


# ── Figure height helper ───────────────────────────────────────────────────────
.fig_height <- function(n_outcomes = length(OUTCOME_LABELS)) {
  n_models <- length(MODEL_SOURCES)
  n_outcomes * n_models * 0.18 + n_outcomes * 0.06 + 1.8
}

# ── Decile outcome selector ────────────────────────────────────────────────────
# Returns 11 outcome codes at the 0th, 10th, …, 100th percentile positions of
# order_asc. Ties in ΔR² are resolved by the upstream sort order (stable sort),
# so the selection is deterministic without an explicit seed.
select_percentile_outcomes <- function(order_asc) {
  n     <- length(order_asc)
  probs <- seq(0, 1, by = 0.1)
  pos   <- unique(pmax(1L, pmin(n, round(probs * (n - 1L)) + 1L)))
  order_asc[pos]
}


# ── Single-index figure ────────────────────────────────────────────────────────
figure3 <- function(index = "ReADI") {
  p   <- make_panel(index)
  out <- file.path(FIG_DIR, paste0("figure3_", index, ".png"))
  ggsave(out, p, width = 9, height = .fig_height(), dpi = 300, bg = "white")
  message("Saved → ", out)
  invisible(p)
}

# ── Main-paper figure: 11 outcomes at decile positions ────────────────────────
figure3_main <- function() {
  indices  <- c("ReADI", "SDI", "SVI")
  order    <- canonical_order()
  sel      <- select_percentile_outcomes(order)

  message("Main figure outcomes (", length(sel), "): ", paste(sel, collapse = ", "))

  all_vals <- map_dfr(indices, ~ build_data(.x, sel))$additional_var
  xlim     <- c(min(all_vals, 0) - 0.01, max(all_vals) * 1.08)

  plots <- imap(indices, function(idx, i) {
    make_panel(idx, outcome_order_asc = sel, x_limits = xlim,
               show_y = (i == 1L), show_x_lbl = (i == 2L))
  })

  composite <- wrap_plots(plots, nrow = 1) +
    plot_layout(guides = "collect") +
    plot_annotation(
      caption = "* placeholder — embedding run not yet complete",
      theme   = theme(
        plot.caption = element_text(size = 7.5, color = "#888888",
                                    face = "italic", hjust = 0,
                                    margin = margin(t = 4))
      )
    ) &
    theme(legend.position = "bottom")

  out <- file.path(FIG_DIR, "figure3_main.png")
  ggsave(out, composite, width = 7.5, height = 7.0, dpi = 300, bg = "white")
  message("Saved → ", out)
  invisible(composite)
}


# ── Three-index composite figure ──────────────────────────────────────────────
figure3_all <- function() {
  indices <- c("ReADI", "SDI", "SVI")
  order   <- canonical_order()

  # Global x range so all panels share the same scale
  all_vals <- map_dfr(indices, ~ build_data(.x, order))$additional_var
  xlim <- c(min(all_vals, 0) - 0.01, max(all_vals) * 1.08)

  plots <- imap(indices, function(idx, i) {
    make_panel(
      idx,
      outcome_order_asc = order,
      x_limits          = xlim,
      show_y            = (i == 1L),
      compact           = TRUE
    )
  })

  composite <- wrap_plots(plots, nrow = 1) +
    plot_layout(guides = "collect") +
    plot_annotation(
      caption = "* placeholder — embedding run not yet complete",
      theme   = theme(
        plot.caption = element_text(size = 7.5, color = "#888888",
                                    face = "italic", hjust = 0,
                                    margin = margin(t = 4))
      )
    ) &
    theme(legend.position = "bottom")

  out <- file.path(FIG_DIR, "figure3_all.png")
  ggsave(out, composite, width = 18, height = .fig_height(), dpi = 300, bg = "white")
  message("Saved → ", out)
  invisible(composite)
}


# ── CLI ────────────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)

if ("--all" %in% args) {
  figure3_all()
} else if ("--main" %in% args) {
  figure3_main()
} else {
  index  <- "ReADI"
  flag_i <- which(args == "--index")
  if (length(flag_i) > 0 && length(args) >= flag_i + 1L) index <- args[flag_i + 1L]
  figure3(index)
}

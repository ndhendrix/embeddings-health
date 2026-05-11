# Literature Review Guidance: Spatial Embeddings + Neighborhood Disadvantage + ACS Residuals

## Your project in one sentence
You're testing whether **satellite-derived spatial embeddings** (likely AlphaEarth or similar) explain residual variance in health outcomes that remains *after* adjusting for ACS-based neighborhood disadvantage indices (ADI, SDI, SVI, etc.) — and framing this for an epidemiology audience that needs to be brought up to speed on what embeddings even are, why the ACS data they currently use is politically fragile, and how this approach might or might not advance the field.

---

## Part 1: Audit of your 15 current Zotero sources

I grouped your existing library into the rhetorical role each item plays in an introduction. **Keep**, **demote** (cite once, briefly), or **cut** judgments are mine — push back where you disagree.

### Group A — Methods/ML background (broad)

| # | Citation | Role | Verdict |
|---|---|---|---|
| 1 | Woodman & Mangoni 2023, *Aging Clin Exp Res* — ML algorithms in geriatric medicine | General ML review | **CUT or demote.** Off-topic. Geriatrics-focused, doesn't touch spatial methods or epidemiology of place. If you want a single broad ML-in-medicine cite, swap for something on representation learning specifically. |
| 6 | Stewart, Robinson, Banerjee 2025 — Geospatial Machine Learning Libraries (arXiv) | Tooling overview | **Demote.** A one-line cite at most. It's a libraries paper, not a substantive methodological argument. Fine in methods, not introduction. |

### Group B — Embedding-adjacent / exposome methods (your novelty argument)

| # | Citation | Role | Verdict |
|---|---|---|---|
| 3 | Luan & Daoyu 2025, *Front Public Health* — Deep learning for high-resolution exposome mapping | Closest to your method | **KEEP & elevate.** This is exposome-flavored deep learning. Strong cite for the "we can use learned representations of place to extend exposure assessment" argument. |
| 5 | Brown, Brumby, Guzder-Williams et al. 2022, *Sci Data* — Dynamic World 10m land cover | Data product | **KEEP if you use Dynamic World**, otherwise demote. It's a land-cover product paper, not an embedding paper. Note that the same first author (Christopher Brown) is on the AlphaEarth Foundations paper — that's the one you actually want to cite for embeddings (see new additions). |

### Group C — ACS / Census measurement (your data source)

| # | Citation | Role | Verdict |
|---|---|---|---|
| 7 | Liang, Nau, Xie et al. 2020, *Permanente J* — ACS in a large integrated healthcare org (Kaiser) | ACS use case in health | **KEEP.** Concrete example of ACS-derived variables being used at scale in a health system. Useful for the "this is how epidemiology currently does it" framing. |
| 8 | Spielman, Folch, Nagle 2014, *Applied Geography* — Patterns of uncertainty in ACS | ACS limitations (statistical) | **KEEP — central.** This is your "ACS has real measurement error problems already" cite, well before any political defunding. Pair with Singh 2003 and Butler et al. to show ACS-derived indices are doing a lot of inferential work on shaky ground. |
| 9 | Census Bureau — Mandatory vs. Voluntary Methods page | Source for voluntary-response risks | **KEEP.** Useful primary-source citation when you describe what would happen under voluntary ACS. |

### Group D — Policy / defunding context

| # | Citation | Role | Verdict |
|---|---|---|---|
| 10 | Rep. Jack Bergman press release 2024 — "Bolster Individual Liberty" bill (re: ACS) | Political pressure | **KEEP but supplement.** A single congressional press release is thin. Add the Section 605 of FY2026 House CJS appropriations language (which would make ACS voluntary and restrict non-response follow-up) and at least one journalistic/policy-analyst source. See new additions below. |

### Group E — Pediatric asthma / sociomarkers

| # | Citation | Role | Verdict |
|---|---|---|---|
| 2 | Shin, Mahajan, Akbilgiç et al. 2018, *NPJ Digit Med* — Sociomarkers and biomarkers for pediatric asthma | Example of social-environment + biomarker modeling | **KEEP only if pediatric asthma or sociomarker framing is part of your motivation.** If your outcome is mortality or chronic disease in adults, this is tangential — cut. |

### Group F — Urban health framing

| # | Citation | Role | Verdict |
|---|---|---|---|
| 4 | Garber, Benmarhnia, de Nazelle et al. 2025, *F1000Research* — Epidemiologic case for urban health | Field-level framing | **KEEP.** Recent, programmatic, frames urban environments as exposures/effect modifiers. Good for the "epidemiology already cares about place — we're proposing a better way to measure it" pivot. |

### Group G — Deprivation indices (the residuals you're trying to improve on)

| # | Citation | Role | Verdict |
|---|---|---|---|
| 11 | Gladish, Phillips, Rehkopf 2026, *JAMA Net Open* — Neighborhood Atlas / reproducible ADI | Critique of ADI implementation | **KEEP — central.** Use this to motivate "even the standard ADI has measurement problems; we need richer features of place." |
| 12 | Stanford Social Deprivation and Vulnerability Indices dataset | Data source | **KEEP as a data citation**, not an intro cite. |
| 13 | Rey, Jougla, Fouillet et al. 2009, *BMC Public Health* — French deprivation index and mortality | Historical deprivation-index work | **DEMOTE.** Old, non-US. One-line cite for "this approach is international and longstanding" or cut entirely. |
| 14 | Singh 2003, *AJPH* — Area deprivation and widening US mortality inequalities | Foundational ADI paper | **KEEP — central.** This is the foundational US ADI/mortality paper. You need this. |
| 15 | Butler, Petterson, Phillips et al. 2013, *HSR* — Social Deprivation Index | Foundational SDI paper | **KEEP — central.** Companion to Singh; if you'll use SDI or motivate it, this is the cite. |

### Suggested cuts (the "stretch" pile)
- **Woodman & Mangoni 2023 (geriatric ML review)** — wrong domain.
- **Rey et al. 2009 (French deprivation)** — old, non-US, doesn't add unique evidence.
- **Shin et al. 2018 (pediatric asthma sociomarkers)** — only keep if asthma/pediatric framing matters to your specific outcomes.
- **Stewart et al. 2025 (Geospatial ML libraries)** — keep for methods section only, not intro.

That trims you from 15 → 11 items in the intro, with 4 of them brought in only briefly.

---

## Part 2: Papers MISSING from your library — high priority additions

These are the gaps an epidemiology reviewer will notice. Listed in order of urgency for your introduction.

### 2A. The foundational embedding paper you HAVE to cite

**Brown, C. F., Kazmierski, M. R., Pasquarella, V. J., et al. (2025). AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data. arXiv:2507.22291.**
- This is the actual technical paper for AlphaEarth — 64-dim embeddings per 10m pixel per year since 2017, globally, in Google Earth Engine.
- If your spatial embeddings come from AlphaEarth (which the surrounding literature strongly suggests), you must cite this as the method paper.
- Same lead author as your Dynamic World citation (#5), so it's a natural chain.

### 2B. Foundational deep-learning-of-built-environment-for-health papers (your direct intellectual lineage)

These are the studies an epidemiology reviewer expects to see, because they did variants of what you're doing with older image-based features instead of pre-trained embeddings:

1. **Maharana, A., & Nsoesie, E. O. (2018). Use of Deep Learning to Examine the Association of the Built Environment With Prevalence of Neighborhood Adult Obesity. *JAMA Network Open*, 1(4), e181535. doi:10.1001/jamanetworkopen.2018.1535**
   - 96 supporting/mentioning citations. The canonical "extract built-environment features from satellite imagery via CNN, predict obesity" paper. They explicitly note CNN features captured information beyond SES.
   - **You absolutely need this** — it's the precedent you're building on.

2. **Chen, Z., Zhang, T., Dazard, J.-E., et al. (2025). AI-Enhanced Analysis of Built Environment Imagery and Neighborhood Obesity in US Cities. *JAMA Network Open*, 8(9), e2534612. doi:10.1001/jamanetworkopen.2025.34612**
   - Marginal R² for fixed effects jumped from **0.632 to 0.745** when image features were added to a DSE + SDOH model.
   - This is essentially your study design (residual-explanation framing, ACS-based covariates as the baseline, image features added on top). You need to acknowledge it and explain how your approach with *embeddings* (not raw images run through a CNN) differs and adds value: faster, more generalizable, doesn't require training a CNN per outcome, pre-trained foundation model encodes multimodal data (radar + optical + climate + canopy).

3. **Phan, L., Yu, W., Keralis, J. M., et al. (2020). Google Street View Derived Built Environment Indicators and Associations with State-Level Obesity, Physical Activity, and Chronic Disease Mortality in the United States. *IJERPH*, 17(10), 3659.**
   - Street-view, not satellite, but same conceptual move: extract features → associate with chronic disease + mortality.

4. **Levy, J., Lebeaux, R. M., Hoen, A. G., et al. (2021). Using Satellite Images and Deep Learning to Identify Associations Between County-Level Mortality and Residential Neighborhood Features Proximal to Schools. *Frontiers in Public Health*, 9, 766707.**
   - Pearson r=0.72 predicting county mortality from satellite images. Closer to your mortality framing than Maharana & Nsoesie.

5. **Yeh, C., Perez, A., Driscoll, A., et al. (2020). Using publicly available satellite imagery and deep learning to understand economic well-being in Africa. *Nature Communications*, 11(1).**
   - 367+ citations. Foundational "we can predict socioeconomic outcomes from satellite imagery" paper. Use as one of 2–3 cites for the broader argument, especially the bit where they note their method "matches or exceeds benchmarks for in-country performance from geostatistical models used to predict health outcomes."

### 2C. AlphaEarth / embedding applications (recent, supports the "this is becoming a thing" argument)

- **Alvarez, C. I., Vaca, C. E. (2025). Machine Learning for Urban Air Quality Prediction Using Google AlphaEarth Foundations Satellite Embeddings: A Case Study of Quito, Ecuador. *Remote Sensing*, 17(20), 3472. doi:10.3390/rs17203472**
  - Demonstrates AlphaEarth embeddings predict an environmental health exposure (NO₂, SO₂, PM₂.₅) with R² up to 0.71. **Crucial proof-of-concept that AlphaEarth captures health-relevant environmental information**, even though this paper isn't about disease outcomes directly.
- **Yue, X., Zhao, Z., Hu, K. (2026). Estimating Economic Activity from Satellite Embeddings. *Applied Sciences*, 16(2), 582.**
  - Shows AlphaEarth embeddings predict economic activity well, useful for your "embeddings encode socioeconomic structure that may otherwise live in ACS variables" argument.
- **Pettersson, M. B., & Daoud, A. (2025). Leveraging Compact Satellite Embeddings and Graph Neural Networks for Large-Scale Poverty Mapping. arXiv:2511.01408.**
  - AlphaEarth + DHS surveys for wealth prediction across Sub-Saharan Africa. Relevant for "embeddings can substitute for / augment household surveys" — a direct analog to the ACS-defunding worry.

### 2D. Residual variance / unmeasured neighborhood effects (your methodological motivation)

The "after adjusting for ADI, there's still unexplained variance" framing has a literature you should anchor in:

- **Goel, N., Hernández, A., Thompson, C., et al. (2023). Neighborhood Disadvantage and Breast Cancer–Specific Survival. *JAMA Network Open*, 6(4), e238908.**
  - Explicitly argues that ADI effects on cancer survival persist after controlling for individual SES, NCCN-guideline treatment, etc., and the residual disparity "suggests unaccounted mechanisms, including unmeasured social determinants of health." This is the exact rhetorical setup you want: "ADI leaves residual variance; what explains it?"
- **Kim, B.-R., Yannatos, I., Blam, K., et al. (2024). Neighborhood disadvantage reduces cognitive reserve independent of neuropathologic change. *Alzheimer's & Dementia*, 20(4), 2707–2718.**
  - Same logic in a different outcome: ADI predicts cognitive function even after controlling for the obvious biological mediators. Reinforces the "unmeasured contextual factor" story you're trying to address.

### 2E. The ACS defunding policy context (current, urgent — this is your motivation hook)

Your Bergman 2024 press release is one data point. You need broader policy context. Citable, credible items:

1. **CBPP report (2025).** "Federal Data Are Disappearing as Statistical Agencies Face Budget Cuts and Political Pressure." Center on Budget and Policy Priorities. https://www.cbpp.org/research/poverty-and-inequality/federal-data-are-disappearing-as-statistical-agencies-face-budget
   - Documents the broader pattern: USDA food security survey defunded, SIPP cuts, ACS political pressure.
2. **Section 605, FY2026 House CJS Appropriations Bill.** Would make ACS response voluntary and restrict non-response follow-up across all Census Bureau surveys. As of December 2025, this language was still alive in the bill.
3. **The Census Project (Nov–Dec 2025 updates).** https://thecensusproject.org/ — multiple organizational letters from 67+ orgs warning about ACS funding and Section 605.
4. **NPR (Jan 2025).** "Funding uncertainty threatens U.S. economic data, federal statistics." https://www.npr.org/2025/01/24/nx-s1-5250264 — quotes former BLS Commissioner Erica Groshen: "we're getting close to the bone now."
5. **Trump administration's March 2026 proposed Census changes** — slash 2026 Operational Test sites from 6 to 2, cut respondent pool 75%, swap census form for ACS, add citizenship question. Particularly important if you want to argue that even nominally-preserved ACS data quality is at near-term risk.

You can use these in a brief paragraph: *Even setting aside long-running concerns about ACS margins of error (Spielman et al., 2014) and the reproducibility of derived indices (Gladish et al., 2026), the ACS faces unprecedented political headwinds — including FY2026 appropriations language that would make response voluntary, projected to substantially reduce response quality and increase costs (Census Bureau, n.d.). Methods that can supplement or partially substitute for ACS-derived neighborhood measures are therefore not merely methodological refinements but a hedge against the possibility that a core epidemiologic data source becomes substantially degraded.*

---

## Part 3: Suggested structure for the introduction (for an epidemiology audience)

I'd organize the intro into **five short sections**. This is the order in which each topic should appear, and which sources belong where.

### §1. Place as exposure: why epidemiology measures neighborhoods
*~150 words.* The standard story: neighborhood-level deprivation predicts mortality, chronic disease, cancer survival, cognitive decline.
- **Anchor cites:** Singh 2003 (#14), Butler 2013 (#15), Garber/Benmarhnia 2025 (#4), Goel 2023 (new), Kim 2024 (new).

### §2. The ACS and its derived indices: a vulnerable backbone
*~200 words.* Most US neighborhood disadvantage measures (ADI, SDI, SVI) are derived from ACS variables. ACS already has well-documented uncertainty problems (Spielman 2014), implementation errors in ADI itself have been recently shown (Gladish 2026), AND the survey is now politically vulnerable.
- **Anchor cites:** Spielman 2014 (#8), Liang 2020 (#7), Gladish 2026 (#11), Singh 2003 (#14), Butler 2013 (#15) — plus the new policy sources above (CBPP, NPR, Section 605, Bergman bill #10).

### §3. The unexplained-variance problem
*~150 words.* Even with these indices, residual neighborhood-level variance in health outcomes persists. Multiple recent studies acknowledge ADI effects on outcomes remain after individual-level adjustment, attributing this to "unmeasured social determinants" or "unaccounted mechanisms."
- **Anchor cites:** Goel 2023 (new), Kim 2024 (new).

### §4. What spatial embeddings are, and what they've done so far in health-adjacent research
*~250 words.* **This is where you teach the epidemiology audience.** Don't assume they know what embeddings are. Frame it as: a foundation model trained on globally available multimodal satellite data (optical, radar, climate, canopy structure) produces a low-dimensional vector for each location that compresses everything visible from space about that place into ~64 dimensions. Unlike ADI, which compresses ~17 social/economic variables, embeddings compress the *physical environment* including land cover, urban form, vegetation, water, infrastructure, etc.

Then cite the precedent that pre-AlphaEarth deep learning of satellite/street imagery already correlates with health:
- Maharana & Nsoesie 2018 (obesity, new)
- Chen et al. 2025 (obesity, marginal R² 0.632 → 0.745 with image features, new)
- Phan et al. 2020 (chronic disease mortality, new)
- Levy et al. 2021 (county mortality, new)
- Yeh et al. 2020 (economic well-being benchmark, new)

Then cite AlphaEarth specifically and recent applications:
- Brown et al. 2025 AlphaEarth paper (new)
- Alvarez et al. 2025 (air pollution in Quito with AlphaEarth, new)
- Luan & Daoyu 2025 (exposome deep learning, #3)

### §5. Study aim & rationale
*~100 words.* Test whether AlphaEarth (or your chosen) spatial embeddings explain residual variance in [your outcome] after adjusting for ADI/SDI/whatever your baseline neighborhood disadvantage measures are. The framing isn't "embeddings replace ACS"; it's:
1. Embeddings may capture aspects of the local environment that ACS variables cannot (built form, vegetation, urban morphology) and that have biological pathways to health (physical activity, exposures, social interaction).
2. As a side benefit, embeddings are globally available and updated annually without requiring household survey response, providing some insurance against ACS degradation.
3. Where embeddings explain residuals, they generate hypotheses for future targeted exposure assessment (e.g., specific land-cover or built-environment patterns to study mechanistically).

You should be cautious to *not* overclaim that embeddings can replace ACS — the Chen 2025 paper is a good example to cite here, noting their image features added to but did not replace SDOH variables.

---

## Part 4: Things to address that you may not have considered

These are issues an alert epidemiology reviewer will raise. Knock them down preemptively.

### 4A. "Aren't you just rediscovering urbanicity?"
Embeddings strongly correlate with urban form, which correlates with everything from population density to walkability to pollution. A reviewer will ask: are your embeddings explaining residual variance, or just re-encoding "is this a dense urban area" in a more elaborate way? Pre-empt by stating you'll adjust for urbanicity / RUCA codes / population density and/or examine embedding components stratified by urbanicity.

### 4B. "Black box / interpretability"
Embeddings are unintelligible vectors. Epidemiologists are trained to ask: *what part of the environment* is doing the work? Discuss what you'll do for interpretability — SHAP on embedding components, examining which embeddings cluster with which kinds of imagery, or post-hoc segmentation of the underlying images for high-residual-explaining locations. Cite Chen 2025 for Grad-CAM as one approach.

### 4C. "Spatial autocorrelation / Tobler's Law"
Pixels near each other have similar embeddings AND similar health outcomes. Some of the "explained residual" could be spatial autocorrelation rather than environmental signal. You'll need spatial-error models, conditional autoregressive models, or geographic cross-validation. The reviewer will check.

### 4D. "ACS measurement error vs. embedding measurement error"
ACS has known MoEs; embeddings have their own systematic errors (cloud cover, sensor changes, temporal lags). Acknowledge this — Vinge et al. 2025 (Earth Observation embeddings benchmark) is a useful reference.

### 4E. "Why these embeddings, not raw imagery + your own CNN?"
Argue: pre-trained foundation embeddings are (a) reproducible — anyone can pull them from GEE; (b) globally consistent; (c) avoid the per-study CNN training that has plagued past work (Levy 2021, Maharana 2018) with overfitting concerns; (d) integrate multimodal data (radar + optical + climate) the average epi research group could not.

---

## Part 5: Quick-reference recommended citation list (final form)

### Keep from your library (11):
1. Singh 2003 (#14)
2. Butler 2013 (#15)
3. Gladish 2026 (#11)
4. Spielman 2014 (#8)
5. Liang 2020 (#7)
6. Census Bureau Mandatory/Voluntary (#9)
7. Bergman press release (#10)
8. Garber/Benmarhnia 2025 (#4)
9. Luan & Daoyu 2025 (#3)
10. Brown et al. 2022 Dynamic World (#5) — if used as data
11. Stanford Indices dataset (#12) — as data citation

### Drop/demote (4):
- Woodman & Mangoni 2023 (#1) — cut
- Rey 2009 (#13) — cut
- Shin 2018 (#2) — cut unless asthma-relevant
- Stewart 2025 (#6) — methods section only

### Add (12 high-priority):
- Brown et al. 2025 AlphaEarth (arXiv:2507.22291)
- Maharana & Nsoesie 2018 (JAMA Net Open)
- Chen et al. 2025 (JAMA Net Open)
- Phan et al. 2020 (IJERPH)
- Levy et al. 2021 (Front Public Health)
- Yeh et al. 2020 (Nat Commun)
- Alvarez et al. 2025 (Remote Sensing — AlphaEarth + air quality)
- Pettersson & Daoud 2025 (AlphaEarth + poverty mapping)
- Goel 2023 (JAMA Net Open — breast cancer + ADI residual)
- Kim 2024 (Alzheimer's & Dementia — ADI residual)
- CBPP 2025 federal data report
- NPR Jan 2025 funding-uncertainty article

Final intro citation count: ~22 items — appropriate for a substantive epi introduction.

---

## Part 6: Things to ask yourself before writing

1. **What's your outcome and unit of analysis?** Mortality? A specific chronic disease? Census-tract or block-group? This determines which precedent papers (Maharana = census tract obesity; Levy = county mortality; Chen = census tract obesity) sit closest to you.
2. **Which spatial embeddings specifically?** AlphaEarth is the safe assumption, but if you're using SatCLIP, GeoCLIP, Tile2Vec, or training your own, the literature shifts.
3. **What's your "previously constructed model"?** Is it ADI alone, SDI alone, or a multivariable model with ADI + individual-level covariates? The residuals from each tell different stories.
4. **Are you doing residual prediction (your embeddings predict the residual) or marginal-R² improvement (your embeddings as added covariates)?** Different framings, slightly different lit reviews — the Chen 2025 marginal-R² approach is methodologically the cleanest precedent.

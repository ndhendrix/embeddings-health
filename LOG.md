# Decision Log for `embeddings-health`

## Vision
The goal of this project is to explore the role of spatial embeddings in epidemiology. 

We disagregate this into two research questions:
1. How can you integrate these embeddings into tract level analyses to augment social risk indices? This would be a practice-based paper focused on more completely capturing social risk in epi analyses. We’d create a correlation matrix and do some regression-based studies, both with embeddings alone and regressing embeddings on residuals from the regression of ReADI ~ tract-level health outcomes.
2. What concepts are these embeddings capturing that social risk indices exclude? This would be looking at internal, neuron-level representations of concepts from open-weight satellite embedding models to see if we can identify concepts that overlap with and/or add to the domains captured by area-based social risk indices. 

## Log

### 26-05-13
1. Corrected "jam codes" issue. ACS uses negative values with repeated numbers (e.g., -222222222) as codes for missing or suppressed data. This was causing extreme correlations for some fields. 
2. Converted SDI and SVI to percentiles prior to use in regressions so that they're treated like ReADI

### 26-05-11
A minimal version of paper 1 has been implemented using ReADI and CDC PLACES data. Next steps are:
1. Add individual-level analysis using secure server
2. Add SDI and SVI to all analyses
3. Currently uses a mix of 2022 and 2023 data from PLACES -- find complete 2022 data if possible

### 26-05-08
1. Paper 1: Guidance for integrating spatial embeddings in epi analyses and proof that it's worth doing so
	1. What is the correlation between satellite embeddings and the data elements from ACS that make up major area-based social risk indices? (see table of elements here: https://www.perplexity.ai/search/please-create-a-table-of-the-e-eCRKF361Ro._BO971qPZ3w)
	2. Compare performance of penalized regression on embeddings at tract level to social risk indices at tract level *using CDC Places data*
		1. NOTE: Include tract area to avoid Modifiable Areal Unit Problem -- the effect of scale on aggregation of subunits -- i.e., rural units might have 1000 embeddings, while urban units might have 10.
	3. Compare performance of penalized regression on embeddings at tract level to social risk indices + age / gender / raceeth at patient level *using AFC data*
	4. For each of 2 and 3, do penalized regression on residual of social risk index regression to see what share of the unexplained variation from social risk indices can be captured by spatial embeddings
	5. Assess heterogeneity of embedding performance across cities / regions / tract sizes
	6. Could we sell this based on threats to federal data availability? 
2. Paper 2: Mechanistic interpretability of spatial embeddings
	1. What concepts are being captured by embeddings that are being missed by area-based social risk indices? 
	2. Approach: Use the Ahsan & Wallace approach to identify what neurons are most associated with high correlation with residuals from index ~ outcome regression
	3. Perform ablation studies to validate relationship
3. Paper 3: Longitudinal focus on embeddings
	1. Can future health be predicted by changes in embeddings? 
	2. Use a lagged change in embeddings to see if they are associated with differences in health at the tract level
	3. The frequency of embedding updates is a strength compared to the more brittle risk indices
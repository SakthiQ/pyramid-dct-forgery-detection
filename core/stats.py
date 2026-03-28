import numpy as np # type: ignore
from scipy.stats import chi2 # type: ignore

def chi_square_test(score_maps: list) -> tuple:
    """
    Perform a Chi-Square test across score maps from different pyramid 
    levels to determine if variation indicates forgery.
    
    Parameters
    ----------
    score_maps : list
        List of 2D np.ndarray score maps from different pyramid levels.
        
    Returns
    -------
    tuple
        (chi_square_value, p_value) corresponding to the distribution test.
    """
    num_levels = len(score_maps)
    if num_levels <= 1:
        return 0.0, 1.0

    # 1. Compute mean score per level
    level_means = np.array([np.mean(sm) for sm in score_maps])
    
    # 2. Compute global mean
    global_mean = np.mean(level_means)
    
    # 3. Calculate deviation for each level
    deviations = level_means - global_mean
    
    # We interpret 'variance' globally across all mapping limits to compute 
    # stable deviations.
    all_scores = np.concatenate([sm.ravel() for sm in score_maps])
    variance = np.var(all_scores)
    
    # Handle small variance safely
    epsilon = 1e-8
    variance = max(float(variance), epsilon)
    
    # 4. Compute chi-square statistic
    chi_square_value = np.sum((deviations ** 2) / variance)
    
    # 5. Compute p-value
    degrees_of_freedom = num_levels - 1
    p_value = 1.0 - chi2.cdf(chi_square_value, df=degrees_of_freedom)
    
    return float(chi_square_value), float(p_value)

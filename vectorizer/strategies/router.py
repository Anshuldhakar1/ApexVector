"""Strategy router for dispatching regions to appropriate vectorization strategies."""
import logging
from typing import List, Callable, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import numpy as np

from vectorizer.types import Region, VectorRegion, RegionKind, AdaptiveConfig, VectorizationError
from vectorizer.strategies.flat import vectorize_flat
from vectorizer.strategies.gradient import vectorize_gradient
from vectorizer.strategies.edge import vectorize_edge
from vectorizer.strategies.detail import vectorize_detail
from vectorizer.debug_utils import audit_regions_at_svg

logger = logging.getLogger(__name__)


def vectorize_all_regions(
    regions: List[Region],
    image: np.ndarray,
    config: AdaptiveConfig,
    parallel: bool = False
) -> List[VectorRegion]:
    """
    Vectorize all regions using appropriate strategies.
    
    Routes each region to the best strategy based on its kind,
    with parallel processing support.
    
    Args:
        regions: List of classified regions
        image: Original image
        config: Adaptive configuration
        parallel: Whether to use parallel processing
        
    Returns:
        List of vectorized regions
    """
    if not regions:
        return []
    
    # Determine number of workers
    if config.parallel_workers == -1:
        max_workers = min(os.cpu_count() or 1, len(regions))
    else:
        max_workers = min(config.parallel_workers, len(regions))
    
    vector_regions = []
    
    if parallel and max_workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_region = {}
            for region in regions:
                future = executor.submit(
                    _vectorize_single_region,
                    region,
                    image,
                    config
                )
                future_to_region[future] = region
            
            # Collect results
            for future in as_completed(future_to_region):
                try:
                    vector_region = future.result()
                    if vector_region:
                        vector_regions.append(vector_region)
                except Exception as e:
                    # Log error and continue with other regions
                    print(f"Error vectorizing region: {e}")
                    # Try fallback
                    region = future_to_region[future]
                    try:
                        vector_region = vectorize_flat(region, image, config.max_bezier_error)
                        vector_regions.append(vector_region)
                    except Exception:
                        pass
    else:
        # Sequential processing with index tracking
        for idx, region in enumerate(regions):
            try:
                vector_region = _vectorize_single_region(region, image, config, region_idx=idx)
                if vector_region:
                    vector_regions.append(vector_region)
            except Exception as e:
                logger.warning(f"Error vectorizing region {idx}: {e}")
                # Fallback to flat
                try:
                    vector_region = vectorize_flat(region, image, config.max_bezier_error, region_idx=idx)
                    vector_regions.append(vector_region)
                except Exception as fallback_e:
                    logger.error(f"Fallback also failed for region {idx}: {fallback_e}")
    
    # Audit regions at SVG export stage
    audit_stats = audit_regions_at_svg(regions, vector_regions, phase="vectorization")
    
    # Phase 4.2: Assert no regions are dropped
    if audit_stats["dropped_regions"] > 0:
        logger.error(
            f"CRITICAL: {audit_stats['dropped_regions']} regions were dropped during vectorization!"
        )
    
    # Assert that output count equals input count (with possible exceptions for empty regions)
    input_count = len([r for r in regions if r.mask is not None and np.any(r.mask)])
    output_count = len(vector_regions)
    
    if output_count != input_count:
        logger.error(
            f"ASSERTION FAILED: Input regions with valid masks: {input_count}, "
            f"Output vector regions: {output_count}. "
            f"Difference: {input_count - output_count}"
        )
    else:
        logger.info(f"Region count assertion passed: {input_count} regions in = {output_count} regions out")
    
    return vector_regions


def _vectorize_single_region(
    region: Region,
    image: np.ndarray,
    config: AdaptiveConfig,
    region_idx: int = -1
) -> VectorRegion:
    """Vectorize a single region using appropriate strategy."""
    # Get region kind (fallback to FLAT if not set)
    kind = getattr(region, 'kind', RegionKind.FLAT)
    
    # Route to appropriate strategy
    if kind == RegionKind.FLAT:
        return vectorize_flat(region, image, config.max_bezier_error, region_idx=region_idx)
    
    elif kind == RegionKind.GRADIENT:
        try:
            return vectorize_gradient(region, image, config.max_bezier_error)
        except Exception as e:
            # Fallback to flat if gradient fails
            logger.warning(f"Gradient strategy failed for region {region_idx}, falling back to flat: {e}")
            return vectorize_flat(region, image, config.max_bezier_error, region_idx=region_idx)
    
    elif kind == RegionKind.EDGE:
        try:
            # Use tighter error for edges
            return vectorize_edge(region, image, config.max_bezier_error * 0.5)
        except Exception as e:
            logger.warning(f"Edge strategy failed for region {region_idx}, falling back to flat: {e}")
            return vectorize_flat(region, image, config.max_bezier_error, region_idx=region_idx)
    
    elif kind == RegionKind.DETAIL:
        try:
            return vectorize_detail(region, image, config.max_mesh_triangles, config.max_bezier_error)
        except Exception as e:
            logger.warning(f"Detail strategy failed for region {region_idx}, falling back to flat: {e}")
            return vectorize_flat(region, image, config.max_bezier_error, region_idx=region_idx)
    
    else:
        # Unknown kind - use flat as default
        return vectorize_flat(region, image, config.max_bezier_error, region_idx=region_idx)


def get_strategy_for_kind(kind: RegionKind) -> Callable:
    """Get the vectorization function for a given region kind."""
    strategies = {
        RegionKind.FLAT: vectorize_flat,
        RegionKind.GRADIENT: vectorize_gradient,
        RegionKind.EDGE: vectorize_edge,
        RegionKind.DETAIL: vectorize_detail,
    }
    
    return strategies.get(kind, vectorize_flat)

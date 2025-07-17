# =============================================================================
# IMPORTS
# =============================================================================
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logger = logging.getLogger(__name__)

# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================
class EffectSchedulerError(Exception):
    """Base exception for EffectScheduler errors."""
    pass


class InvalidEffectError(EffectSchedulerError):
    """Raised when an invalid effect is encountered."""
    pass


class InvalidMetricError(EffectSchedulerError):
    """Raised when invalid metric values are provided."""
    pass


class ParameterValidationError(EffectSchedulerError):
    """Raised when parameter validation fails."""
    pass


# =============================================================================
# MAIN CLASS
# =============================================================================
class EffectScheduler:
    """
    Scheduler for audio effects with adaptive selection based on performance metrics.
    
    This scheduler manages effect probabilities and selects effects based on their
    Bit Error Rate (BER) and mean Intersection over Union (mIoU) performance.
    It uses an exponential moving average to track metrics and adapts selection
    probabilities to favor better-performing effects.
    
    Attributes:
        effect_params: Dictionary mapping effect names to their parameter configurations
        beta: Smoothing factor for exponential moving average (0 < beta < 1)
        ber_threshold: Success threshold for BER (lower is better)
        miou_threshold: Success threshold for mIoU (higher is better)
    """
    
    def __init__(
        self,
        effect_params: Dict[str, Dict[str, Any]],
        beta: float = 0.9,
        ber_threshold: float = 0.001,
        miou_threshold: float = 0.95
    ) -> None:
        """
        Initialize the EffectScheduler with effect parameters and thresholds.
        
        Args:
            effect_params: Dictionary mapping effect names to their parameter configurations.
                          Each effect can have parameters with 'choices' for random selection.
            beta: Smoothing factor for exponential moving average (0 < beta < 1).
                  Higher values give more weight to historical data.
            ber_threshold: Threshold for considering BER as successful (0 <= threshold <= 1).
            miou_threshold: Threshold for considering mIoU as successful (0 <= threshold <= 1).
        
        Raises:
            ValueError: If beta is not in range (0, 1) or thresholds are invalid.
            ParameterValidationError: If effect parameters fail validation.
        """
        # Validate initialization parameters
        if not 0 < beta < 1:
            raise ValueError(f"Beta must be in range (0, 1), got {beta}")
        if not 0 <= ber_threshold <= 1:
            raise ValueError(f"BER threshold must be in range [0, 1], got {ber_threshold}")
        if not 0 <= miou_threshold <= 1:
            raise ValueError(f"mIoU threshold must be in range [0, 1], got {miou_threshold}")
        
        try:
            # Validate effect parameters early
            self._validate_effect_params(effect_params)
        except Exception as e:
            logger.error(f"Effect parameter validation failed: {str(e)}", exc_info=True)
            raise ParameterValidationError(f"Invalid effect parameters: {str(e)}")
        
        self.effect_params = effect_params
        self.beta = beta
        self.ber_threshold = ber_threshold
        self.miou_threshold = miou_threshold
        
        # Initialize effect probabilities uniformly
        num_effects = len(effect_params)
        self.effect_probabilities: Dict[str, float] = {
            effect_name: 1.0 / num_effects for effect_name in effect_params.keys()
        }
        
        # Track effect usage statistics
        self.effect_usage_stats: Dict[str, int] = {
            name: 0 for name in effect_params.keys()
        }
        self.total_effects: int = 0
        
        # Initialize exponential moving average metrics
        self.effect_metrics_history: Dict[str, Dict[str, Optional[float]]] = {
            effect_name: {'ber': None, 'miou': None} 
            for effect_name in effect_params.keys()
        }
        
        self.current_effect_name: Optional[str] = None
        
        # Track parameter-specific success rates
        self.parameter_success_rates: Dict[str, Dict[Tuple[str, Any], List[bool]]] = {}
        
        # For cycling through effects over batches
        self.effect_list: List[str] = list(effect_params.keys())
        self.effect_ptr: int = 0  # Pointer for effect assignments
        
        # Track parameter-specific metrics
        self.parameter_metrics_history: Dict[str, Dict[Any, Dict[str, Any]]] = {
            effect_name: {} for effect_name in effect_params.keys()
        }
        
        # Track metric history for analysis
        self.metric_history: Dict[str, Dict[str, Any]] = {
            effect_name: {
                'overall': {'ber': [], 'miou': []},
                'params': {}  # Parameter-specific histories
            } for effect_name in effect_params.keys()
        }
        
        logger.info(f"EffectScheduler initialized with {num_effects} effects")
    
    # =============================================================================
    # PUBLIC METHODS - EFFECT SELECTION
    # =============================================================================
    
    def select_all_effects(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Return all available effects with resolved parameters.
        
        This method is used when applying all effects to every batch, useful for
        comprehensive testing or when effect diversity is prioritized over
        performance-based selection.
        
        Returns:
            List of tuples containing (effect_name, resolved_parameters) for all
            available effects. Parameters with 'choices' will be randomly selected.
        
        Raises:
            EffectSchedulerError: If parameter resolution fails for any effect.
        """
        effects: List[Tuple[str, Dict[str, Any]]] = []
        
        try:
            # Process all effects in order
            for effect_name in self.effect_params.keys():
                raw_params = self.effect_params.get(effect_name, {})
                self.current_effect_name = effect_name
                
                # Resolve parameters with weighted selection
                effect_params = self._resolve_effect_params(raw_params)
                effects.append((effect_name, effect_params))
                
                # Update usage statistics
                self.effect_usage_stats[effect_name] += 1
                self.total_effects += 1
                
            logger.debug(f"Selected all {len(effects)} effects for application")
            return effects
            
        except Exception as e:
            logger.error(f"Failed to select all effects: {str(e)}", exc_info=True)
            raise EffectSchedulerError(f"Effect selection failed: {str(e)}")
    
    def select_effects(self, num_effects: int = 3) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Select effects based on their performance-weighted probabilities.
        
        This method uses the current effect probabilities (which are adapted based
        on performance) to randomly select a specified number of effects. The same
        effect can be selected multiple times if it has high probability.
        
        Args:
            num_effects: Number of effects to select. Will be capped at the total
                        number of available effects if larger.
        
        Returns:
            List of tuples containing (effect_name, resolved_parameters) for the
            selected effects. Parameters are resolved based on success rates.
        
        Raises:
            ValueError: If num_effects is not positive.
            EffectSchedulerError: If effect selection or parameter resolution fails.
        """
        if num_effects <= 0:
            raise ValueError(f"Number of effects must be positive, got {num_effects}")
        
        effects: List[Tuple[str, Dict[str, Any]]] = []
        
        try:
            # Get effect names and their probabilities
            effect_names = list(self.effect_probabilities.keys())
            probabilities = [self.effect_probabilities[name] for name in effect_names]
            
            # Normalize probabilities to ensure they sum to 1
            prob_sum = sum(probabilities)
            if prob_sum > 0:
                probabilities = [p / prob_sum for p in probabilities]
            else:
                # Fallback to uniform distribution if all probabilities are 0
                logger.warning("All effect probabilities are 0, using uniform distribution")
                probabilities = [1.0 / len(effect_names) for _ in effect_names]
            
            # Sample effects based on probabilities
            selected_names = np.random.choice(
                effect_names,
                size=min(num_effects, len(effect_names)),
                replace=True,  # Allow selecting the same effect multiple times
                p=probabilities
            )
            
            # Process each selected effect
            for effect_name in selected_names:
                raw_params = self.effect_params.get(effect_name, {})
                self.current_effect_name = effect_name
                
                # Resolve parameters with success-rate weighting
                effect_params = self._resolve_effect_params(raw_params)
                effects.append((effect_name, effect_params))
                
                # Update usage statistics
                self.effect_usage_stats[effect_name] += 1
                self.total_effects += 1
            
            logger.debug(f"Selected {len(effects)} effects based on probabilities")
            return effects
            
        except Exception as e:
            logger.error(f"Failed to select effects: {str(e)}", exc_info=True)
            raise EffectSchedulerError(f"Effect selection failed: {str(e)}")
    
    # =============================================================================
    # PUBLIC METHODS - METRICS AND STATISTICS
    # =============================================================================
    
    def get_effect_probabilities(self) -> Dict[str, float]:
        """
        Get current effect selection probabilities.
        
        Returns:
            Dictionary mapping effect names to their current selection probabilities.
            Probabilities sum to 1.0 and reflect the adaptive weighting based on
            performance metrics.
        """
        return self.effect_probabilities.copy()
    
    def get_effect_statistics(self) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Get comprehensive statistics for all effects.
        
        Returns:
            Dictionary mapping effect names to their statistics including:
            - usage_percentage: Percentage of total selections
            - ema_ber: Exponential moving average of BER (if available)
            - ema_miou: Exponential moving average of mIoU (if available)
            - avg_ber: Simple average of all BER measurements (if available)
            - avg_miou: Simple average of all mIoU measurements (if available)
            - selection_count: Total number of times selected
        """
        stats: Dict[str, Dict[str, Optional[float]]] = {}
        
        try:
            for effect_name in self.effect_params.keys():
                # Get current metrics
                metrics = self.effect_metrics_history[effect_name]
                
                # Calculate usage percentage
                usage_pct = (
                    (self.effect_usage_stats[effect_name] / self.total_effects * 100)
                    if self.total_effects > 0 else 0.0
                )
                
                # Calculate averages from history
                history = self.metric_history[effect_name]['overall']
                avg_ber = np.mean(history['ber']) if history['ber'] else None
                avg_miou = np.mean(history['miou']) if history['miou'] else None
                
                stats[effect_name] = {
                    'usage_percentage': usage_pct,
                    'ema_ber': metrics['ber'],
                    'ema_miou': metrics['miou'],
                    'avg_ber': avg_ber,
                    'avg_miou': avg_miou,
                    'selection_count': self.effect_usage_stats[effect_name]
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get effect statistics: {str(e)}", exc_info=True)
            return {}
    
    def update_effect_metrics(
        self,
        effect_name: str,
        effect_params: Dict[str, Any],
        localized_ber: float,
        miou: float
    ) -> None:
        """
        Update performance metrics for a specific effect and parameter combination.
        
        This method updates both the overall effect metrics and parameter-specific
        metrics using exponential moving averages. It also tracks success rates
        for individual parameter values to enable adaptive parameter selection.
        
        Args:
            effect_name: Name of the effect to update metrics for.
            effect_params: Dictionary of parameter values used for this measurement.
            localized_ber: Bit Error Rate measurement (0 <= BER <= 1, lower is better).
            miou: Mean Intersection over Union measurement (0 <= mIoU <= 1, higher is better).
        
        Raises:
            InvalidEffectError: If effect_name is not recognized.
            InvalidMetricError: If metric values are outside valid range [0, 1].
        """
        # Validate inputs
        if effect_name not in self.effect_params:
            raise InvalidEffectError(f"Unknown effect: '{effect_name}'")
        
        if not 0 <= localized_ber <= 1:
            raise InvalidMetricError(
                f"BER must be in range [0, 1], got {localized_ber}"
            )
        
        if not 0 <= miou <= 1:
            raise InvalidMetricError(
                f"mIoU must be in range [0, 1], got {miou}"
            )
        
        try:
            beta = self.beta
            
            # Update overall effect metrics using exponential moving average
            metrics = self.effect_metrics_history.setdefault(
                effect_name, {'ber': None, 'miou': None}
            )
            
            # Initialize or update BER
            if metrics['ber'] is None:
                # First observation - initialize with current value
                self.effect_metrics_history[effect_name]['ber'] = localized_ber
            else:
                # Update exponential moving average
                self.effect_metrics_history[effect_name]['ber'] = (
                    beta * metrics['ber'] + (1 - beta) * localized_ber
                )
            
            # Initialize or update mIoU
            if metrics['miou'] is None:
                # First observation - initialize with current value
                self.effect_metrics_history[effect_name]['miou'] = miou
            else:
                # Update exponential moving average
                self.effect_metrics_history[effect_name]['miou'] = (
                    beta * metrics['miou'] + (1 - beta) * miou
                )
            
            # Store raw metrics in history for analysis
            effect_history = self.metric_history[effect_name]
            effect_history['overall']['ber'].append(localized_ber)
            effect_history['overall']['miou'].append(miou)
            
            # Store parameter-specific history
            param_key = self.make_hashable(effect_params)
            if param_key not in effect_history['params']:
                effect_history['params'][param_key] = {'ber': [], 'miou': []}
            
            effect_history['params'][param_key]['ber'].append(localized_ber)
            effect_history['params'][param_key]['miou'].append(miou)
            
            # Determine if this measurement represents a "success"
            is_success = (
                localized_ber <= self.ber_threshold and 
                miou >= self.miou_threshold
            )
            
            # Track success rates for individual parameter values
            for param_name, param_value in effect_params.items():
                param_value_hashable = self.make_hashable(param_value)
                param_tuple = (param_name, param_value_hashable)
                
                # Initialize tracking structures if needed
                if effect_name not in self.parameter_success_rates:
                    self.parameter_success_rates[effect_name] = {}
                if param_tuple not in self.parameter_success_rates[effect_name]:
                    self.parameter_success_rates[effect_name][param_tuple] = []
                
                # Record success status
                self.parameter_success_rates[effect_name][param_tuple].append(is_success)
            
            # Update parameter-specific metrics
            param_metrics = self.parameter_metrics_history[effect_name].setdefault(
                param_key, {'ber': None, 'miou': None, 'count': 0}
            )
            
            if param_metrics['ber'] is None:
                param_metrics['ber'] = localized_ber
                param_metrics['miou'] = miou
            else:
                # Update exponential moving averages for this parameter combination
                param_metrics['ber'] = beta * param_metrics['ber'] + (1 - beta) * localized_ber
                param_metrics['miou'] = beta * param_metrics['miou'] + (1 - beta) * miou
            
            param_metrics['count'] += 1
            
            logger.debug(
                f"Updated metrics for {effect_name}: BER={localized_ber:.4f}, "
                f"mIoU={miou:.4f}, success={is_success}"
            )
            
        except Exception as e:
            logger.error(f"Failed to update effect metrics: {str(e)}", exc_info=True)
            raise
    
    def adapt_effect_probabilities(self) -> None:
        """
        Adapt effect selection probabilities based on performance metrics.
        
        This method recalculates effect probabilities using a reward-based system
        where effects with lower BER and higher mIoU receive higher selection
        probabilities. The adaptation uses exponential smoothing to prevent
        sudden changes and maintain stability.
        
        The reward calculation gives 80% weight to BER performance (inverted since
        lower is better) and 20% weight to mIoU performance.
        
        Raises:
            EffectSchedulerError: If probability adaptation fails.
        """
        try:
            effect_scores: Dict[str, float] = {}
            # Smoothing factor prevents sudden probability changes
            smoothing_factor = 0.8
            
            # Calculate performance scores for each effect
            for effect_name, param_metrics in self.parameter_metrics_history.items():
                if not param_metrics:
                    # No metrics available yet - assign neutral score
                    effect_scores[effect_name] = 0.0
                    continue
                
                # Calculate average score across all parameter combinations
                param_scores: List[float] = []
                for metrics in param_metrics.values():
                    if metrics['ber'] is not None and metrics['miou'] is not None:
                        # Calculate reward: lower BER is better (invert), higher mIoU is better
                        # Weight BER more heavily as it's typically more critical
                        reward = 0.8 * (1 - metrics['ber']) + 0.2 * metrics['miou']
                        param_scores.append(reward)
                
                if param_scores:
                    effect_scores[effect_name] = np.mean(param_scores)
                else:
                    effect_scores[effect_name] = 0.0
            
            # Convert scores to probabilities using softmax
            effect_names = list(effect_scores.keys())
            scores = np.array([effect_scores[name] for name in effect_names])
            
            if np.all(scores == 0):
                # All scores are zero - maintain uniform distribution
                new_probabilities = np.ones_like(scores) / len(scores)
                logger.debug("All effect scores are zero, maintaining uniform distribution")
            else:
                # Apply softmax with temperature scaling for numerical stability
                temperature = 1.0
                scores_stable = scores - np.max(scores)  # Prevent overflow
                exp_scores = np.exp(scores_stable / temperature)
                new_probabilities = exp_scores / np.sum(exp_scores)
            
            # Apply exponential smoothing to prevent sudden changes
            for effect_name, new_prob in zip(effect_names, new_probabilities):
                old_prob = self.effect_probabilities[effect_name]
                smoothed_prob = (
                    smoothing_factor * old_prob + 
                    (1 - smoothing_factor) * new_prob
                )
                self.effect_probabilities[effect_name] = smoothed_prob
            
            # Ensure probabilities sum to 1.0
            self._normalize_probabilities()
            
            logger.debug("Effect probabilities adapted based on performance metrics")
            
        except Exception as e:
            logger.error(f"Failed to adapt effect probabilities: {str(e)}", exc_info=True)
            raise EffectSchedulerError(f"Probability adaptation failed: {str(e)}")
    
    def log_adaptive_behavior(self, logger_func: Optional[Any] = None) -> None:
        """
        Log comprehensive adaptive behavior statistics.
        
        This method outputs detailed information about the current state of the
        scheduler including effect probabilities, performance metrics, and usage
        statistics. Useful for monitoring and debugging adaptive behavior.
        
        Args:
            logger_func: Optional logging function to use. If None, uses print().
                        The function should accept a single string argument.
        """
        if logger_func is None:
            logger_func = print
        
        try:
            logger_func("\n" + "=" * 60)
            logger_func("EFFECT SCHEDULER ADAPTIVE BEHAVIOR")
            logger_func("=" * 60)
            
            # Log current effect probabilities
            logger_func("\nEffect Selection Probabilities:")
            for effect, prob in sorted(
                self.effect_probabilities.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                logger_func(f"  {effect}: {prob:.4f}")
            
            # Log detailed effect statistics
            stats = self.get_effect_statistics()
            logger_func("\nEffect Performance Statistics:")
            for effect, effect_stats in sorted(stats.items()):
                logger_func(f"\n  {effect}:")
                logger_func(f"    Usage: {effect_stats['usage_percentage']:.1f}%")
                
                if effect_stats['ema_ber'] is not None:
                    logger_func(f"    EMA BER: {effect_stats['ema_ber']:.4f}")
                if effect_stats['ema_miou'] is not None:
                    logger_func(f"    EMA mIoU: {effect_stats['ema_miou']:.4f}")
                if effect_stats['avg_ber'] is not None:
                    logger_func(f"    Avg BER: {effect_stats['avg_ber']:.4f}")
                if effect_stats['avg_miou'] is not None:
                    logger_func(f"    Avg mIoU: {effect_stats['avg_miou']:.4f}")
            
            logger_func("=" * 60 + "\n")
            
        except Exception as e:
            logger.error(f"Failed to log adaptive behavior: {str(e)}", exc_info=True)
    
    # =============================================================================
    # PRIVATE METHODS - PARAMETER HANDLING
    # =============================================================================
    
    def _validate_effect_params(self, effect_params: Dict[str, Dict[str, Any]]) -> None:
        """
        Validate effect parameters with special handling for bandpass filter.
        
        This method ensures that effect parameters are properly structured and
        that special constraints (like bandpass filter frequency ordering) are
        satisfiable.
        
        Args:
            effect_params: Dictionary of effect parameters to validate.
        
        Raises:
            ParameterValidationError: If validation fails.
        """
        try:
            # Special validation for bandpass filter
            if 'bandpass_filter' in effect_params:
                bp_params = effect_params['bandpass_filter']
                
                # Check if both frequency parameters exist
                if 'cutoff_freq_low' in bp_params and 'cutoff_freq_high' in bp_params:
                    low_config = bp_params.get('cutoff_freq_low', {})
                    high_config = bp_params.get('cutoff_freq_high', {})
                    
                    low_choices = low_config.get('choices', [])
                    high_choices = high_config.get('choices', [])
                    
                    if low_choices and high_choices:
                        # Ensure at least one valid combination exists where low < high
                        valid_combo_exists = False
                        for low_freq in low_choices:
                            for high_freq in high_choices:
                                if low_freq < high_freq:
                                    valid_combo_exists = True
                                    break
                            if valid_combo_exists:
                                break
                        
                        if not valid_combo_exists:
                            raise ParameterValidationError(
                                f"Bandpass filter has no valid frequency combinations. "
                                f"Low frequencies {low_choices} must have at least one "
                                f"value less than high frequencies {high_choices}"
                            )
                        
                        logger.debug("Bandpass filter parameters validated successfully")
            
        except ParameterValidationError:
            raise
        except Exception as e:
            logger.error(f"Parameter validation error: {str(e)}", exc_info=True)
            raise ParameterValidationError(f"Failed to validate parameters: {str(e)}")
    
    def _resolve_effect_params(self, raw_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve effect parameters by selecting from choices based on success rates.
        
        For parameters with 'choices', this method uses historical success rates
        to weight the selection probability. Parameters that have led to more
        successful outcomes (low BER, high mIoU) are more likely to be selected.
        
        Args:
            raw_params: Raw parameter configuration which may contain 'choices'.
        
        Returns:
            Resolved parameters with specific values selected from choices.
        
        Raises:
            ParameterValidationError: If parameter resolution fails.
        """
        resolved_params: Dict[str, Any] = {}
        
        try:
            for param_key, param_config in raw_params.items():
                if isinstance(param_config, dict) and 'choices' in param_config:
                    choices_list = param_config['choices']
                    
                    if not choices_list:
                        logger.warning(f"Empty choices list for parameter {param_key}")
                        continue
                    
                    # Calculate weights based on historical success rates
                    weights: List[float] = []
                    for choice in choices_list:
                        choice_hashable = self.make_hashable(choice)
                        param_tuple = (param_key, choice_hashable)
                        
                        # Get success history for this parameter value
                        success_history = (
                            self.parameter_success_rates
                            .get(self.current_effect_name, {})
                            .get(param_tuple, [])
                        )
                        
                        # Calculate success rate with default of 0.5 for unknown values
                        if success_history:
                            success_rate = sum(success_history) / len(success_history)
                        else:
                            success_rate = 0.5  # Neutral weight for unexplored values
                        
                        # Add small constant to avoid zero weights
                        weight = success_rate + 0.1
                        weights.append(weight)
                    
                    # Normalize weights and select based on probabilities
                    total_weight = sum(weights)
                    if total_weight > 0:
                        probabilities = [w / total_weight for w in weights]
                        choice_idx = np.random.choice(len(choices_list), p=probabilities)
                    else:
                        # Fallback to uniform random selection
                        choice_idx = np.random.randint(len(choices_list))
                    
                    resolved_params[param_key] = choices_list[choice_idx]
                    
                else:
                    # No 'choices' - use parameter value as-is
                    resolved_params[param_key] = param_config
            
            # Special handling for bandpass filter to ensure freq_low < freq_high
            if self.current_effect_name == 'bandpass_filter':
                self._validate_bandpass_frequencies(resolved_params)
            
            return resolved_params
            
        except Exception as e:
            logger.error(f"Failed to resolve parameters: {str(e)}", exc_info=True)
            raise ParameterValidationError(f"Parameter resolution failed: {str(e)}")
    
    def _validate_bandpass_frequencies(self, params: Dict[str, Any]) -> None:
        """
        Ensure bandpass filter frequencies satisfy low < high constraint.
        
        If the constraint is violated, this method attempts to fix it by
        selecting valid alternatives from the available choices.
        
        Args:
            params: Dictionary of resolved parameters to validate and fix.
        
        Note:
            This method modifies params in-place if corrections are needed.
        """
        cutoff_low = params.get('cutoff_freq_low')
        cutoff_high = params.get('cutoff_freq_high')
        
        if cutoff_low is not None and cutoff_high is not None:
            if cutoff_low >= cutoff_high:
                logger.warning(
                    f"Invalid bandpass frequencies: low={cutoff_low}, "
                    f"high={cutoff_high}. Attempting to fix..."
                )
                
                bp_config = self.effect_params.get('bandpass_filter', {})
                
                # Try to find a valid high frequency
                high_choices = bp_config.get('cutoff_freq_high', {}).get('choices', [])
                valid_highs = [f for f in high_choices if f > cutoff_low]
                
                if valid_highs:
                    # Select a valid high frequency
                    cutoff_high = np.random.choice(valid_highs)
                else:
                    # No valid high found, try to find a valid low frequency
                    low_choices = bp_config.get('cutoff_freq_low', {}).get('choices', [])
                    valid_lows = [f for f in low_choices if f < cutoff_high]
                    
                    if valid_lows:
                        cutoff_low = np.random.choice(valid_lows)
                    else:
                        # Last resort: use min low and max high
                        logger.error(
                            "No valid bandpass frequency combination found. "
                            "Using extreme values."
                        )
                        cutoff_low = min(low_choices) if low_choices else cutoff_low
                        cutoff_high = max(high_choices) if high_choices else cutoff_high
                
                # Update the parameters
                params['cutoff_freq_low'] = cutoff_low
                params['cutoff_freq_high'] = cutoff_high
                
                logger.info(
                    f"Fixed bandpass frequencies: low={cutoff_low}, high={cutoff_high}"
                )
    
    # =============================================================================
    # PRIVATE METHODS - UTILITIES
    # =============================================================================
    
    def _normalize_probabilities(self) -> None:
        """
        Normalize effect probabilities to sum to 1.0 with numerical stability.
        
        This method handles edge cases like very small probability sums and
        ensures the final probabilities are properly normalized even in the
        presence of numerical errors.
        
        Raises:
            EffectSchedulerError: If normalization fails completely.
        """
        try:
            total = sum(self.effect_probabilities.values())
            
            # Only normalize if sum deviates from 1.0 beyond numerical precision
            if abs(total - 1.0) > 1e-6:
                # Add small epsilon to avoid division by zero
                total = max(total, 1e-10)
                
                for key in self.effect_probabilities:
                    self.effect_probabilities[key] /= total
            
            # Verify normalization succeeded
            final_total = sum(self.effect_probabilities.values())
            if abs(final_total - 1.0) > 1e-6:
                # Normalization failed - use uniform distribution as fallback
                logger.warning(
                    f"Probability normalization failed (sum={final_total}), "
                    f"using uniform distribution"
                )
                num_effects = len(self.effect_probabilities)
                for key in self.effect_probabilities:
                    self.effect_probabilities[key] = 1.0 / num_effects
                    
        except Exception as e:
            logger.error(f"Failed to normalize probabilities: {str(e)}", exc_info=True)
            raise EffectSchedulerError(f"Probability normalization failed: {str(e)}")
    
    def make_hashable(self, value: Any) -> Any:
        """
        Convert a value to a hashable representation for dictionary keys.
        
        This method recursively converts lists, tuples, dicts, and numpy arrays
        to hashable tuple representations that can be used as dictionary keys.
        
        Args:
            value: Any value to convert to hashable form.
        
        Returns:
            Hashable representation of the input value.
        """
        if isinstance(value, (list, tuple)):
            return tuple(self.make_hashable(v) for v in value)
        elif isinstance(value, dict):
            return tuple(sorted((k, self.make_hashable(v)) for k, v in value.items()))
        elif isinstance(value, np.ndarray):
            return tuple(value.tolist())
        else:
            return value


# =============================================================================
# TEST CODE
# =============================================================================
if __name__ == "__main__":
    # Configure logging for test execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Define a sample effect parameter grid for testing
    effect_param_grid = {
        'identity': {},  # No parameters
        'highpass_filter': {
            'cutoff_freq': {'choices': [100, 200, 300]}
        },
        'median_filter': {
            'kernel_size': {'choices': [3, 5, 7]}
        },
        'bandpass_filter': {
            'cutoff_freq_low': {'choices': [100, 200, 300]},
            'cutoff_freq_high': {'choices': [3000, 4000, 5000]}
        }
    }
    
    try:
        # Create scheduler instance
        scheduler = EffectScheduler(effect_param_grid, beta=0.9)
        
        # Initialize by testing each effect at least once
        print("\nInitial testing of all effects:")
        print("--------------------------------")
        
        for effect_name in effect_param_grid.keys():
            # Force selection of specific effect for initial testing
            scheduler.current_effect_name = effect_name
            effect_params = scheduler._resolve_effect_params(
                effect_param_grid[effect_name]
            )
            
            # Simulate random metrics for testing
            simulated_ber = np.random.uniform(0.1, 0.5)
            simulated_miou = np.random.uniform(0.5, 0.9)
            
            # Update metrics
            scheduler.update_effect_metrics(
                effect_name, effect_params, simulated_ber, simulated_miou
            )
            print(
                f"  {effect_name}: BER = {simulated_ber:.3f}, "
                f"mIoU = {simulated_miou:.3f}"
            )
        
        # Test effect selection and adaptation over multiple iterations
        print("\nTesting effect selection and adaptation:")
        print("----------------------------------------")
        
        batch_size = 4
        num_iterations = 5
        
        for i in range(num_iterations):
            print(f"\nIteration {i + 1}:")
            
            # Select effects for current batch
            selected_effects = scheduler.select_effects(batch_size)
            
            print("Selected effects:")
            for j, (effect_name, params) in enumerate(selected_effects):
                print(f"  Sample {j}: {effect_name} with params: {params}")
            
            # Simulate metrics for the selected effects
            print("\nSimulating metrics and updating:")
            for effect_name, params in selected_effects:
                # Simulate performance metrics with some variance
                simulated_ber = np.random.uniform(0.1, 0.5)
                simulated_miou = np.random.uniform(0.5, 0.9)
                
                # Update scheduler with new metrics
                scheduler.update_effect_metrics(
                    effect_name, params, simulated_ber, simulated_miou
                )
                print(
                    f"  {effect_name}: BER = {simulated_ber:.3f}, "
                    f"mIoU = {simulated_miou:.3f}"
                )
            
            # Adapt probabilities based on accumulated metrics
            scheduler.adapt_effect_probabilities()
            
            # Display current state
            print("\nUpdated effect probabilities:")
            probs = scheduler.get_effect_probabilities()
            for effect, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                print(f"  {effect}: {prob:.3f}")
        
        # Log comprehensive adaptive behavior summary
        print("\n" + "=" * 60)
        scheduler.log_adaptive_behavior()
        
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}", exc_info=True)
        raise
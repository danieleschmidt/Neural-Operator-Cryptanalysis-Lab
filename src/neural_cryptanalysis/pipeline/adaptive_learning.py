"""Adaptive Learning Module for Self-Healing Pipelines.

This module implements machine learning algorithms that continuously learn
from pipeline behavior to improve failure prediction and recovery strategies.

The system uses reinforcement learning and online learning techniques to
adapt to changing conditions and optimize recovery actions.
"""

import json
import logging
import numpy as np
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import threading
import time

# Mock imports for dependencies that may not be available
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import SGDRegressor
    from sklearn.metrics import mean_squared_error
except ImportError:
    # Mock implementations
    class RandomForestRegressor:
        def __init__(self, **kwargs): 
            self.feature_importances_ = []
        def fit(self, X, y): return self
        def predict(self, X): return [0.5] * len(X)
    
    class SGDRegressor:
        def __init__(self, **kwargs): pass
        def fit(self, X, y): return self
        def predict(self, X): return [0.5] * len(X)
        def partial_fit(self, X, y): return self
    
    def mean_squared_error(y_true, y_pred): 
        return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)


@dataclass
class LearningState:
    """Represents the learning state of the adaptive system."""
    model_version: int
    training_samples: int
    accuracy: float
    last_updated: datetime
    feature_importance: Dict[str, float]
    prediction_confidence: float


@dataclass
class ActionOutcome:
    """Records the outcome of a recovery action."""
    action_name: str
    timestamp: datetime
    pre_action_metrics: Dict[str, float]
    post_action_metrics: Dict[str, float]
    success: bool
    improvement_score: float
    context: Dict[str, Any]


class ReinforcementLearner:
    """Reinforcement learning for recovery action optimization."""
    
    def __init__(self, learning_rate: float = 0.1, exploration_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.action_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.logger = logging.getLogger(__name__)
    
    def choose_action(self, state: str, available_actions: List[str]) -> str:
        """Choose action using epsilon-greedy strategy."""
        import random
        
        if random.random() < self.exploration_rate:
            # Explore: choose random action
            return random.choice(available_actions)
        else:
            # Exploit: choose best known action
            q_values = {action: self.q_table[state][action] for action in available_actions}
            return max(q_values, key=q_values.get)
    
    def update_q_value(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        available_next_actions: List[str]
    ) -> None:
        """Update Q-value using Q-learning algorithm."""
        # Calculate max Q-value for next state
        max_next_q = max(
            self.q_table[next_state][next_action]
            for next_action in available_next_actions
        ) if available_next_actions else 0
        
        # Q-learning update
        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (reward + 0.9 * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        # Update action count
        self.action_counts[state][action] += 1
        
        # Decay exploration rate
        self.exploration_rate *= 0.995
        self.exploration_rate = max(0.01, self.exploration_rate)
    
    def get_action_confidence(self, state: str, action: str) -> float:
        """Get confidence in action for given state."""
        count = self.action_counts[state][action]
        if count == 0:
            return 0.0
        
        # Confidence based on number of trials and Q-value
        q_value = self.q_table[state][action]
        confidence = min(1.0, (count / 10.0) * (1.0 + q_value))
        return max(0.0, confidence)


class OnlineLearner:
    """Online learning for continuous model adaptation."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.model = SGDRegressor(learning_rate='adaptive', eta0=0.01)
        self.feature_buffer = deque(maxlen=window_size)
        self.target_buffer = deque(maxlen=window_size)
        self.is_fitted = False
        self.performance_history = deque(maxlen=50)
        self.logger = logging.getLogger(__name__)
    
    def add_sample(self, features: List[float], target: float) -> None:
        """Add new training sample."""
        self.feature_buffer.append(features)
        self.target_buffer.append(target)
        
        # Initial fit when we have enough samples
        if not self.is_fitted and len(self.feature_buffer) >= 10:
            X = list(self.feature_buffer)
            y = list(self.target_buffer)
            self.model.fit(X, y)
            self.is_fitted = True
            self.logger.info("Online learner initialized")
        
        # Incremental learning
        elif self.is_fitted:
            self.model.partial_fit([features], [target])
    
    def predict(self, features: List[float]) -> float:
        """Make prediction."""
        if not self.is_fitted:
            return 0.5  # Default prediction
        
        try:
            prediction = self.model.predict([features])[0]
            return max(0.0, min(1.0, prediction))  # Clamp to [0, 1]
        except Exception as e:
            self.logger.warning(f"Prediction error: {e}")
            return 0.5
    
    def evaluate_performance(self) -> float:
        """Evaluate model performance on recent data."""
        if not self.is_fitted or len(self.feature_buffer) < 10:
            return 0.0
        
        try:
            X = list(self.feature_buffer)[-20:]
            y_true = list(self.target_buffer)[-20:]
            y_pred = [self.predict(x) for x in X]
            
            mse = mean_squared_error(y_true, y_pred)
            performance = max(0.0, 1.0 - mse)  # Convert MSE to performance score
            
            self.performance_history.append(performance)
            return performance
            
        except Exception as e:
            self.logger.warning(f"Performance evaluation error: {e}")
            return 0.0


class PatternRecognizer:
    """Recognizes patterns in pipeline behavior and failures."""
    
    def __init__(self, pattern_window: int = 50):
        self.pattern_window = pattern_window
        self.failure_patterns = {}
        self.success_patterns = {}
        self.temporal_patterns = deque(maxlen=pattern_window)
        self.logger = logging.getLogger(__name__)
    
    def add_observation(
        self,
        metrics: Dict[str, float],
        failure_occurred: bool,
        timestamp: datetime
    ) -> None:
        """Add new observation for pattern recognition."""
        # Extract key features
        feature_vector = self._extract_features(metrics)
        
        # Add to temporal sequence
        self.temporal_patterns.append({
            'features': feature_vector,
            'failure': failure_occurred,
            'timestamp': timestamp
        })
        
        # Update pattern libraries
        pattern_key = self._vectorize_features(feature_vector)
        
        if failure_occurred:
            self.failure_patterns[pattern_key] = self.failure_patterns.get(pattern_key, 0) + 1
        else:
            self.success_patterns[pattern_key] = self.success_patterns.get(pattern_key, 0) + 1
    
    def _extract_features(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Extract categorical features from metrics."""
        features = {}
        
        # Categorize CPU usage
        cpu = metrics.get('cpu_usage', 0)
        if cpu < 30:
            features['cpu_level'] = 'low'
        elif cpu < 70:
            features['cpu_level'] = 'medium'
        else:
            features['cpu_level'] = 'high'
        
        # Categorize memory usage
        memory = metrics.get('memory_usage', 0)
        if memory < 40:
            features['memory_level'] = 'low'
        elif memory < 80:
            features['memory_level'] = 'medium'
        else:
            features['memory_level'] = 'high'
        
        # Categorize error rate
        error_rate = metrics.get('error_rate', 0)
        if error_rate < 0.01:
            features['error_level'] = 'low'
        elif error_rate < 0.05:
            features['error_level'] = 'medium'
        else:
            features['error_level'] = 'high'
        
        # Categorize response time
        response_time = metrics.get('response_time', 0)
        if response_time < 100:
            features['response_level'] = 'fast'
        elif response_time < 300:
            features['response_level'] = 'medium'
        else:
            features['response_level'] = 'slow'
        
        return features
    
    def _vectorize_features(self, features: Dict[str, str]) -> str:
        """Convert feature dict to string key."""
        return ','.join(f"{k}:{v}" for k, v in sorted(features.items()))
    
    def predict_failure_risk(self, metrics: Dict[str, float]) -> float:
        """Predict failure risk based on recognized patterns."""
        feature_vector = self._extract_features(metrics)
        pattern_key = self._vectorize_features(feature_vector)
        
        failure_count = self.failure_patterns.get(pattern_key, 0)
        success_count = self.success_patterns.get(pattern_key, 0)
        total_count = failure_count + success_count
        
        if total_count == 0:
            return 0.5  # Unknown pattern
        
        # Calculate failure probability with smoothing
        failure_probability = (failure_count + 1) / (total_count + 2)
        return failure_probability
    
    def detect_temporal_patterns(self) -> List[Dict[str, Any]]:
        """Detect temporal patterns in failures."""
        if len(self.temporal_patterns) < 10:
            return []
        
        patterns = []
        
        # Look for recurring time-based patterns
        time_deltas = []
        failure_times = [
            obs['timestamp'] for obs in self.temporal_patterns
            if obs['failure']
        ]
        
        if len(failure_times) >= 3:
            for i in range(1, len(failure_times)):
                delta = (failure_times[i] - failure_times[i-1]).total_seconds()
                time_deltas.append(delta)
            
            # Check for periodic patterns
            if time_deltas:
                avg_delta = np.mean(time_deltas)
                std_delta = np.std(time_deltas)
                
                if std_delta / avg_delta < 0.3:  # Low variance indicates periodicity
                    patterns.append({
                        'type': 'periodic_failure',
                        'period_seconds': avg_delta,
                        'confidence': 1 - (std_delta / avg_delta)
                    })
        
        return patterns


class AdaptiveLearningEngine:
    """Main adaptive learning engine that coordinates all learning components."""
    
    def __init__(
        self,
        model_save_path: Optional[Path] = None,
        learning_rate: float = 0.1,
        adaptation_interval: int = 100
    ):
        self.model_save_path = model_save_path or Path("adaptive_models")
        self.learning_rate = learning_rate
        self.adaptation_interval = adaptation_interval
        
        # Initialize learning components
        self.rl_learner = ReinforcementLearner(learning_rate=learning_rate)
        self.online_learner = OnlineLearner()
        self.pattern_recognizer = PatternRecognizer()
        
        # State tracking
        self.learning_state = LearningState(
            model_version=1,
            training_samples=0,
            accuracy=0.0,
            last_updated=datetime.now(),
            feature_importance={},
            prediction_confidence=0.0
        )
        
        # Action outcome tracking
        self.action_outcomes: List[ActionOutcome] = []
        self.context_state_mapping = {}
        
        # Threading for background learning
        self.learning_thread: Optional[threading.Thread] = None
        self.stop_learning = threading.Event()
        
        self.logger = logging.getLogger(__name__)
        
        # Load existing models if available
        self._load_models()
    
    def start_background_learning(self) -> None:
        """Start background learning thread."""
        if self.learning_thread and self.learning_thread.is_alive():
            return
        
        self.stop_learning.clear()
        self.learning_thread = threading.Thread(target=self._background_learning_loop, daemon=True)
        self.learning_thread.start()
        self.logger.info("Started background adaptive learning")
    
    def stop_background_learning(self) -> None:
        """Stop background learning thread."""
        self.stop_learning.set()
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        self.logger.info("Stopped background adaptive learning")
    
    def record_action_outcome(
        self,
        action_name: str,
        pre_metrics: Dict[str, float],
        post_metrics: Dict[str, float],
        success: bool,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record outcome of a recovery action for learning."""
        # Calculate improvement score
        improvement_score = self._calculate_improvement_score(pre_metrics, post_metrics)
        
        outcome = ActionOutcome(
            action_name=action_name,
            timestamp=datetime.now(),
            pre_action_metrics=pre_metrics.copy(),
            post_action_metrics=post_metrics.copy(),
            success=success,
            improvement_score=improvement_score,
            context=context or {}
        )
        
        self.action_outcomes.append(outcome)
        
        # Update reinforcement learning
        state = self._metrics_to_state(pre_metrics)
        next_state = self._metrics_to_state(post_metrics)
        reward = improvement_score if success else -0.5
        
        # Get available actions (simplified)
        available_actions = ['restart_service', 'scale_resources', 'clear_cache', 
                           'optimize_algorithms', 'circuit_breaker', 'graceful_degradation']
        
        self.rl_learner.update_q_value(state, action_name, reward, next_state, available_actions)
        
        # Update online learner
        features = list(pre_metrics.values())
        target = improvement_score
        self.online_learner.add_sample(features, target)
        
        # Update pattern recognizer
        self.pattern_recognizer.add_observation(
            post_metrics, not success, outcome.timestamp
        )
        
        self.learning_state.training_samples += 1
        self.logger.debug(f"Recorded action outcome: {action_name} -> {improvement_score:.3f}")
    
    def recommend_action(
        self,
        current_metrics: Dict[str, float],
        available_actions: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float]:
        """Recommend best action based on learned experience."""
        state = self._metrics_to_state(current_metrics)
        
        # Get RL recommendation
        rl_action = self.rl_learner.choose_action(state, available_actions)
        rl_confidence = self.rl_learner.get_action_confidence(state, rl_action)
        
        # Get pattern-based prediction
        failure_risk = self.pattern_recognizer.predict_failure_risk(current_metrics)
        
        # Combine recommendations
        if rl_confidence > 0.7:
            recommended_action = rl_action
            confidence = rl_confidence
        else:
            # Fallback to heuristic selection based on metrics
            recommended_action = self._heuristic_action_selection(current_metrics, available_actions)
            confidence = 0.5 + failure_risk * 0.3
        
        self.learning_state.prediction_confidence = confidence
        return recommended_action, confidence
    
    def predict_failure_probability(self, metrics: Dict[str, float]) -> float:
        """Predict probability of failure given current metrics."""
        # Pattern-based prediction
        pattern_risk = self.pattern_recognizer.predict_failure_risk(metrics)
        
        # ML-based prediction
        features = list(metrics.values())
        ml_risk = self.online_learner.predict(features)
        
        # Combine predictions
        combined_risk = (pattern_risk + ml_risk) / 2
        return combined_risk
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the learning process."""
        # Calculate action effectiveness
        action_effectiveness = {}
        for outcome in self.action_outcomes[-100:]:  # Recent outcomes
            action = outcome.action_name
            if action not in action_effectiveness:
                action_effectiveness[action] = []
            action_effectiveness[action].append(outcome.improvement_score)
        
        # Average effectiveness per action
        avg_effectiveness = {
            action: np.mean(scores)
            for action, scores in action_effectiveness.items()
        }
        
        # Detect patterns
        temporal_patterns = self.pattern_recognizer.detect_temporal_patterns()
        
        # Model performance
        model_performance = self.online_learner.evaluate_performance()
        
        return {
            'learning_state': {
                'model_version': self.learning_state.model_version,
                'training_samples': self.learning_state.training_samples,
                'accuracy': model_performance,
                'last_updated': self.learning_state.last_updated.isoformat()
            },
            'action_effectiveness': avg_effectiveness,
            'temporal_patterns': temporal_patterns,
            'total_outcomes': len(self.action_outcomes),
            'exploration_rate': self.rl_learner.exploration_rate
        }
    
    def _background_learning_loop(self) -> None:
        """Background learning loop for continuous adaptation."""
        while not self.stop_learning.is_set():
            try:
                if len(self.action_outcomes) > 0:
                    # Periodic model updates
                    if self.learning_state.training_samples % self.adaptation_interval == 0:
                        self._update_models()
                        self._save_models()
                    
                    # Evaluate and adapt
                    performance = self.online_learner.evaluate_performance()
                    self.learning_state.accuracy = performance
                    self.learning_state.last_updated = datetime.now()
                
                # Sleep before next iteration
                self.stop_learning.wait(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Background learning error: {e}")
                self.stop_learning.wait(60)
    
    def _metrics_to_state(self, metrics: Dict[str, float]) -> str:
        """Convert metrics to state string for RL."""
        # Discretize metrics into state representation
        cpu = metrics.get('cpu_usage', 0)
        memory = metrics.get('memory_usage', 0)
        error_rate = metrics.get('error_rate', 0)
        
        cpu_state = 'high' if cpu > 70 else 'medium' if cpu > 30 else 'low'
        memory_state = 'high' if memory > 80 else 'medium' if memory > 40 else 'low'
        error_state = 'high' if error_rate > 0.05 else 'medium' if error_rate > 0.01 else 'low'
        
        return f"cpu:{cpu_state},mem:{memory_state},err:{error_state}"
    
    def _calculate_improvement_score(
        self,
        pre_metrics: Dict[str, float],
        post_metrics: Dict[str, float]
    ) -> float:
        """Calculate improvement score from pre/post metrics."""
        improvements = []
        
        # CPU improvement (lower is better)
        cpu_pre = pre_metrics.get('cpu_usage', 50)
        cpu_post = post_metrics.get('cpu_usage', 50)
        cpu_improvement = max(-1, min(1, (cpu_pre - cpu_post) / 100))
        improvements.append(cpu_improvement)
        
        # Memory improvement (lower is better)
        mem_pre = pre_metrics.get('memory_usage', 50)
        mem_post = post_metrics.get('memory_usage', 50)
        mem_improvement = max(-1, min(1, (mem_pre - mem_post) / 100))
        improvements.append(mem_improvement)
        
        # Error rate improvement (lower is better)
        err_pre = pre_metrics.get('error_rate', 0.01)
        err_post = post_metrics.get('error_rate', 0.01)
        err_improvement = max(-1, min(1, (err_pre - err_post) * 10))
        improvements.append(err_improvement)
        
        # Response time improvement (lower is better)
        resp_pre = pre_metrics.get('response_time', 200)
        resp_post = post_metrics.get('response_time', 200)
        resp_improvement = max(-1, min(1, (resp_pre - resp_post) / 1000))
        improvements.append(resp_improvement)
        
        # Calculate weighted average
        score = np.mean(improvements)
        return max(0, min(1, (score + 1) / 2))  # Normalize to [0, 1]
    
    def _heuristic_action_selection(
        self,
        metrics: Dict[str, float],
        available_actions: List[str]
    ) -> str:
        """Heuristic action selection when ML confidence is low."""
        cpu = metrics.get('cpu_usage', 0)
        memory = metrics.get('memory_usage', 0)
        error_rate = metrics.get('error_rate', 0)
        response_time = metrics.get('response_time', 0)
        
        # Rule-based selection
        if cpu > 90 or memory > 95:
            return 'restart_service' if 'restart_service' in available_actions else available_actions[0]
        elif cpu > 70 or memory > 80:
            return 'scale_resources' if 'scale_resources' in available_actions else available_actions[0]
        elif error_rate > 0.1:
            return 'circuit_breaker' if 'circuit_breaker' in available_actions else available_actions[0]
        elif response_time > 1000:
            return 'optimize_algorithms' if 'optimize_algorithms' in available_actions else available_actions[0]
        else:
            return 'clear_cache' if 'clear_cache' in available_actions else available_actions[0]
    
    def _update_models(self) -> None:
        """Update ML models with recent data."""
        try:
            # Update feature importance based on recent outcomes
            recent_outcomes = self.action_outcomes[-50:]
            if recent_outcomes:
                feature_names = ['cpu_usage', 'memory_usage', 'error_rate', 'response_time']
                importance_scores = {}
                
                for feature in feature_names:
                    correlations = []
                    for outcome in recent_outcomes:
                        pre_value = outcome.pre_action_metrics.get(feature, 0)
                        improvement = outcome.improvement_score
                        correlations.append(abs(pre_value * improvement))
                    
                    importance_scores[feature] = np.mean(correlations)
                
                # Normalize importance scores
                total_importance = sum(importance_scores.values())
                if total_importance > 0:
                    self.learning_state.feature_importance = {
                        k: v / total_importance for k, v in importance_scores.items()
                    }
            
            self.learning_state.model_version += 1
            self.logger.info(f"Updated models to version {self.learning_state.model_version}")
            
        except Exception as e:
            self.logger.error(f"Model update error: {e}")
    
    def _save_models(self) -> None:
        """Save learned models to disk."""
        try:
            self.model_save_path.mkdir(parents=True, exist_ok=True)
            
            # Save RL Q-table
            rl_path = self.model_save_path / "rl_qtable.json"
            with open(rl_path, 'w') as f:
                # Convert defaultdict to regular dict for JSON serialization
                q_table_dict = {
                    state: dict(actions) for state, actions in self.rl_learner.q_table.items()
                }
                json.dump(q_table_dict, f)
            
            # Save learning state
            state_path = self.model_save_path / "learning_state.json"
            with open(state_path, 'w') as f:
                state_dict = {
                    'model_version': self.learning_state.model_version,
                    'training_samples': self.learning_state.training_samples,
                    'accuracy': self.learning_state.accuracy,
                    'last_updated': self.learning_state.last_updated.isoformat(),
                    'feature_importance': self.learning_state.feature_importance,
                    'prediction_confidence': self.learning_state.prediction_confidence
                }
                json.dump(state_dict, f, indent=2)
            
            # Save recent outcomes
            outcomes_path = self.model_save_path / "recent_outcomes.json"
            recent_outcomes = self.action_outcomes[-100:]  # Save last 100
            outcomes_data = []
            for outcome in recent_outcomes:
                outcomes_data.append({
                    'action_name': outcome.action_name,
                    'timestamp': outcome.timestamp.isoformat(),
                    'pre_action_metrics': outcome.pre_action_metrics,
                    'post_action_metrics': outcome.post_action_metrics,
                    'success': outcome.success,
                    'improvement_score': outcome.improvement_score,
                    'context': outcome.context
                })
            
            with open(outcomes_path, 'w') as f:
                json.dump(outcomes_data, f, indent=2)
            
            self.logger.info("Models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Model save error: {e}")
    
    def _load_models(self) -> None:
        """Load previously saved models."""
        try:
            if not self.model_save_path.exists():
                return
            
            # Load RL Q-table
            rl_path = self.model_save_path / "rl_qtable.json"
            if rl_path.exists():
                with open(rl_path, 'r') as f:
                    q_table_dict = json.load(f)
                    # Convert back to defaultdict
                    for state, actions in q_table_dict.items():
                        for action, value in actions.items():
                            self.rl_learner.q_table[state][action] = value
            
            # Load learning state
            state_path = self.model_save_path / "learning_state.json"
            if state_path.exists():
                with open(state_path, 'r') as f:
                    state_dict = json.load(f)
                    self.learning_state.model_version = state_dict.get('model_version', 1)
                    self.learning_state.training_samples = state_dict.get('training_samples', 0)
                    self.learning_state.accuracy = state_dict.get('accuracy', 0.0)
                    self.learning_state.feature_importance = state_dict.get('feature_importance', {})
                    self.learning_state.prediction_confidence = state_dict.get('prediction_confidence', 0.0)
                    
                    last_updated_str = state_dict.get('last_updated')
                    if last_updated_str:
                        self.learning_state.last_updated = datetime.fromisoformat(last_updated_str)
            
            self.logger.info(f"Loaded models (version {self.learning_state.model_version})")
            
        except Exception as e:
            self.logger.warning(f"Model load error: {e}")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create adaptive learning engine
    engine = AdaptiveLearningEngine()
    engine.start_background_learning()
    
    try:
        # Simulate some action outcomes
        for i in range(10):
            pre_metrics = {
                'cpu_usage': 70 + i * 2,
                'memory_usage': 60 + i,
                'error_rate': 0.01 + i * 0.001,
                'response_time': 200 + i * 10
            }
            
            post_metrics = {
                'cpu_usage': max(10, pre_metrics['cpu_usage'] - 20),
                'memory_usage': max(10, pre_metrics['memory_usage'] - 15),
                'error_rate': max(0, pre_metrics['error_rate'] - 0.005),
                'response_time': max(50, pre_metrics['response_time'] - 50)
            }
            
            engine.record_action_outcome(
                action_name='scale_resources',
                pre_metrics=pre_metrics,
                post_metrics=post_metrics,
                success=True
            )
            
            time.sleep(0.1)
        
        # Get recommendations
        current_metrics = {'cpu_usage': 85, 'memory_usage': 75, 'error_rate': 0.02, 'response_time': 300}
        available_actions = ['restart_service', 'scale_resources', 'clear_cache']
        
        action, confidence = engine.recommend_action(current_metrics, available_actions)
        print(f"Recommended action: {action} (confidence: {confidence:.2f})")
        
        # Get insights
        insights = engine.get_learning_insights()
        print("\nLearning insights:")
        print(json.dumps(insights, indent=2, default=str))
        
    finally:
        engine.stop_background_learning()
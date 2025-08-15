import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
from scipy.stats import pearsonr
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

@dataclass
class ActionCameraImpact:
    """Impact event specific to action camera scenarios"""
    time_index: int
    timestamp: float  # Time in seconds
    accel_magnitude: float
    gyro_magnitude: float
    impact_severity: str  # 'light', 'medium', 'heavy'
    impact_type: str  # 'drop', 'hit', 'crash', 'bounce'
    confidence: float
    camera_motion: str  # 'stationary', 'handheld', 'mounted', 'falling'

class ActionCameraImpactDetector:
    def __init__(self, sampling_rate=200):
        self.fs = sampling_rate
        self.nyquist = sampling_rate / 2
        
        # Action camera specific parameters
        self.setup_filters()
        self.calibrate_for_action_camera()
        
    def setup_filters(self):
        """Setup filters optimized for action camera scenarios"""
        # Action cameras often have more noise and vibration
        # Use slightly higher cutoff to avoid camera shake
        self.hp_cutoff = 8  # Hz - lower to catch camera drops
        
        # Impact frequency range for action cameras (drops, hits, crashes)
        self.impact_low, self.impact_high = 8, 60  # Hz
        
        # Option 2: Use Hz directly with fs parameter (cleaner approach)
        try:
            # Ensure frequencies are valid for this sampling rate
            max_freq = self.fs / 2.1  # Slightly below Nyquist for safety
            
            # Adjust frequencies if they're too high
            hp_cutoff = min(self.hp_cutoff, max_freq)
            impact_low = min(self.impact_low, max_freq * 0.5)
            impact_high = min(self.impact_high, max_freq)
            lp_cutoff = min(2, max_freq * 0.1)
            
            print(f"🔧 Filter setup: fs={self.fs}Hz, max_safe={max_freq:.1f}Hz")
            print(f"   Using frequencies: hp={hp_cutoff:.1f}Hz, bp={impact_low:.1f}-{impact_high:.1f}Hz, lp={lp_cutoff:.1f}Hz")
            
            # Create filters using Hz frequencies
            self.hp_b, self.hp_a = butter(4, hp_cutoff, btype='high', fs=self.fs)
            self.bp_b, self.bp_a = butter(4, [impact_low, impact_high], btype='band', fs=self.fs)
            self.lp_b, self.lp_a = butter(2, lp_cutoff, btype='low', fs=self.fs)
            
            print("✅ Filters created successfully")
            
        except Exception as e:
            print(f"❌ Filter creation failed: {e}")
            # Fallback: no filtering (pass-through)
            self.hp_b, self.hp_a = [1], [1]
            self.bp_b, self.bp_a = [1], [1]
            self.lp_b, self.lp_a = [1], [1]
            print("   Using pass-through filters (no filtering)")
        
    def calibrate_for_action_camera(self):
        """Set parameters specific to action camera use cases"""
        # Typical g-force ranges for different impact types
        self.impact_thresholds = {
            'light': 3.0,    # Light bump or set-down (3g)
            'medium': 6.0,   # Drop from waist height or firm hit (6g) 
            'heavy': 12.0    # Hard crash or drop from significant height (12g+)
        }
        
        # Gyro thresholds (rad/s) for rotational impacts
        self.gyro_thresholds = {
            'light': 5.0,    # Light tumble
            'medium': 15.0,  # Significant rotation
            'heavy': 30.0    # Violent spinning/tumbling
        }
        
        # Weights for sensor fusion (action cameras: accel often more reliable)
        self.accel_weight = 1.2
        self.gyro_weight = 0.8
        
    def analyze_camera_motion_state(self, accel_data, gyro_data, window_size=100):
        """
        Determine camera motion state (handheld, mounted, falling, etc.)
        This helps contextualize impact detection
        """
        # Low-pass filter for general motion trends
        accel_trend = filtfilt(self.lp_b, self.lp_a, np.abs(accel_data))
        gyro_trend = filtfilt(self.lp_b, self.lp_a, np.abs(gyro_data))
        
        motion_states = []
        
        for i in range(0, len(accel_data), window_size):
            end_idx = min(i + window_size, len(accel_data))
            
            accel_window = accel_trend[i:end_idx]
            gyro_window = gyro_trend[i:end_idx]
            
            accel_var = np.var(accel_window)
            gyro_var = np.var(gyro_window)
            accel_mean = np.mean(np.abs(accel_window))
            
            # Classify motion state based on variance and magnitude
            if accel_var < 0.5 and gyro_var < 2.0:
                state = 'stationary'  # Camera not moving much
            elif accel_var > 3.0 and gyro_var > 10.0:
                if accel_mean > 15.0:  # High acceleration suggests falling
                    state = 'falling'
                else:
                    state = 'handheld'  # Typical handheld shake
            elif accel_var < 1.5 and gyro_var < 5.0:
                state = 'mounted'  # Steady mounting (tripod, helmet, etc.)
            else:
                state = 'handheld'  # Default assumption
                
            motion_states.extend([state] * (end_idx - i))
        
        return motion_states
    
    def detect_drops_and_throws(self, accel_data, min_duration=0.2):
        """
        Specifically detect camera drops and throws using acceleration patterns
        """
        # Look for sustained low acceleration (free fall) followed by high impact
        gravity = 9.81
        freefall_threshold = 3.0  # m/s² - significantly less than gravity
        
        min_samples = int(min_duration * self.fs)
        
        # Calculate magnitude of acceleration
        accel_mag = np.abs(accel_data)
        
        # Find periods of low acceleration (potential free fall)
        low_accel_periods = accel_mag < freefall_threshold
        
        # Find connected regions of low acceleration
        drop_events = []
        in_freefall = False
        freefall_start = 0
        
        for i, is_low in enumerate(low_accel_periods):
            if is_low and not in_freefall:
                freefall_start = i
                in_freefall = True
            elif not is_low and in_freefall:
                freefall_duration = i - freefall_start
                if freefall_duration >= min_samples:
                    # Check for impact after freefall
                    impact_window = slice(i, min(i + int(0.1 * self.fs), len(accel_data)))
                    if len(accel_data[impact_window]) > 0:
                        max_impact = np.max(accel_mag[impact_window])
                        if max_impact > self.impact_thresholds['medium']:
                            drop_events.append({
                                'freefall_start': freefall_start,
                                'freefall_end': i,
                                'impact_index': i + np.argmax(accel_mag[impact_window]),
                                'impact_magnitude': max_impact,
                                'freefall_duration': freefall_duration / self.fs
                            })
                in_freefall = False
        
        return drop_events
    
    def classify_impact_scenario(self, accel_mag, gyro_mag, motion_state, context_window):
        """
        Classify the type of impact based on camera context
        """
        # Analyze the context around the impact
        pre_impact = context_window[:len(context_window)//2]
        post_impact = context_window[len(context_window)//2:]
        
        pre_accel_var = np.var(pre_impact)
        post_accel_var = np.var(post_impact)
        
        # Classification logic based on patterns
        if motion_state == 'falling' and accel_mag > self.impact_thresholds['medium']:
            return 'drop'
        elif gyro_mag > self.gyro_thresholds['heavy'] and accel_mag > self.impact_thresholds['heavy']:
            return 'crash'  # High rotation + high acceleration = crash/collision
        elif accel_mag > self.impact_thresholds['medium'] and pre_accel_var < 1.0:
            return 'hit'  # Sudden impact from stable state
        elif accel_mag < self.impact_thresholds['medium'] and gyro_mag > self.gyro_thresholds['light']:
            return 'bounce'  # Light impact with rotation
        else:
            return 'hit'  # Default classification
    
    def detect_action_camera_impacts(self, accel_data, gyro_data, 
                                   sensitivity='medium', 
                                   include_light_impacts=False):
        """
        Main impact detection optimized for action camera scenarios
        
        Args:
            accel_data: accelerometer readings (can be single axis or magnitude)
            gyro_data: gyroscope readings (can be single axis or magnitude) 
            sensitivity: 'low', 'medium', 'high'
            include_light_impacts: whether to detect gentle bumps/touches
        """
        
        # Convert to numpy arrays
        accel = np.array(accel_data)
        gyro = np.array(gyro_data)
        
        # Determine camera motion state
        motion_states = self.analyze_camera_motion_state(accel, gyro)
        
        # Detect specific drop events
        drop_events = self.detect_drops_and_throws(accel)
        
        # Check input data quality before filtering
        print(f"🔍 Input data validation:")
        print(f"   Accel: shape={accel.shape}, range=[{accel.min():.3f}, {accel.max():.3f}], has_nan={np.any(np.isnan(accel))}")
        print(f"   Gyro: shape={gyro.shape}, range=[{gyro.min():.3f}, {gyro.max():.3f}], has_nan={np.any(np.isnan(gyro))}")
        
        # Clean input data
        if np.any(np.isnan(accel)) or np.any(np.isinf(accel)):
            print("⚠️  Cleaning accel data (NaN/inf found)")
            accel = np.nan_to_num(accel, nan=0.0, posinf=0.0, neginf=0.0)
            
        if np.any(np.isnan(gyro)) or np.any(np.isinf(gyro)):
            print("⚠️  Cleaning gyro data (NaN/inf found)")
            gyro = np.nan_to_num(gyro, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Filter signals for impact detection with error handling
        try:
            print(f"🔧 Applying bandpass filter...")
            print(f"   Filter b coeffs: {self.bp_b[:3]}... (length={len(self.bp_b)})")
            print(f"   Filter a coeffs: {self.bp_a[:3]}... (length={len(self.bp_a)})")
            
            # Check filter stability
            from scipy.signal import tf2zpk
            zeros, poles, gain = tf2zpk(self.bp_b, self.bp_a)
            if np.any(np.abs(poles) >= 1.0):
                print("⚠️  Filter may be unstable (poles >= 1.0)")
            
            accel_filtered = filtfilt(self.bp_b, self.bp_a, accel)
            gyro_filtered = filtfilt(self.bp_b, self.bp_a, gyro)
            
            # Check filter output
            print(f"   Accel filtered: range=[{accel_filtered.min():.3f}, {accel_filtered.max():.3f}], has_nan={np.any(np.isnan(accel_filtered))}")
            print(f"   Gyro filtered: range=[{gyro_filtered.min():.3f}, {gyro_filtered.max():.3f}], has_nan={np.any(np.isnan(gyro_filtered))}")
            
            if np.any(np.isnan(accel_filtered)) or np.any(np.isnan(gyro_filtered)):
                print("❌ Filter output contains NaN - using unfiltered data")
                accel_filtered = accel
                gyro_filtered = gyro
                
        except Exception as e:
            print(f"❌ Filtering failed: {e}")
            print("   Using unfiltered data")
            accel_filtered = accel
            gyro_filtered = gyro
        
        # Smooth for better peak detection
        accel_smooth = savgol_filter(np.abs(accel_filtered), 
                                   window_length=min(11, len(accel_filtered)//10*2+1), 
                                   polyorder=2)
        gyro_smooth = savgol_filter(np.abs(gyro_filtered), 
                                  window_length=min(11, len(gyro_filtered)//10*2+1), 
                                  polyorder=2)
        
        # Adaptive thresholding based on motion state and sensitivity
        threshold_multipliers = {
            'low': {'accel': 8, 'gyro': 6},
            'medium': {'accel': 5, 'gyro': 4},
            'high': {'accel': 3, 'gyro': 2.5}
        }
        
        mult = threshold_multipliers[sensitivity]
        
        # Calculate adaptive thresholds
        accel_baseline = np.percentile(accel_smooth, 85)  # Use 85th percentile as baseline
        gyro_baseline = np.percentile(gyro_smooth, 85)
        
        accel_threshold = accel_baseline + mult['accel'] * np.std(accel_smooth)
        gyro_threshold = gyro_baseline + mult['gyro'] * np.std(gyro_smooth)
        
        # Find potential impacts
        accel_peaks, accel_props = find_peaks(accel_smooth, 
                                            height=accel_threshold,
                                            distance=int(0.05 * self.fs))  # 50ms minimum between peaks
        
        gyro_peaks, gyro_props = find_peaks(gyro_smooth,
                                          height=gyro_threshold, 
                                          distance=int(0.05 * self.fs))
        
        # Combine and validate impacts
        impact_events = []
        all_candidate_peaks = set(accel_peaks) | set(gyro_peaks)
        
        for peak_idx in all_candidate_peaks:
            if peak_idx >= len(accel_smooth) or peak_idx >= len(gyro_smooth):
                continue
                
            accel_mag = accel_smooth[peak_idx] 
            gyro_mag = gyro_smooth[peak_idx]
            
            # Skip light impacts if not requested
            if not include_light_impacts and accel_mag < self.impact_thresholds['light']:
                continue
            
            # Get motion state at impact time
            motion_state = motion_states[min(peak_idx, len(motion_states)-1)]
            
            # Get context window for classification
            window_size = int(0.2 * self.fs)  # 200ms context window
            start_idx = max(0, peak_idx - window_size//2)
            end_idx = min(len(accel), peak_idx + window_size//2)
            context_window = accel[start_idx:end_idx]
            
            # Classify impact scenario
            impact_type = self.classify_impact_scenario(accel_mag, gyro_mag, motion_state, context_window)
            
            # Determine severity
            if accel_mag >= self.impact_thresholds['heavy']:
                severity = 'heavy'
            elif accel_mag >= self.impact_thresholds['medium']:
                severity = 'medium'  
            else:
                severity = 'light'
            
            # Calculate confidence based on signal clarity
            noise_level = np.std(accel_smooth[max(0, peak_idx-20):peak_idx+20])
            signal_to_noise = accel_mag / (noise_level + 1e-6)
            confidence = min(1.0, signal_to_noise / 10.0)
            
            # Create impact event
            impact_event = ActionCameraImpact(
                time_index=peak_idx,
                timestamp=peak_idx / self.fs,
                accel_magnitude=accel_mag,
                gyro_magnitude=gyro_mag,
                impact_severity=severity,
                impact_type=impact_type,
                confidence=confidence,
                camera_motion=motion_state
            )
            
            impact_events.append(impact_event)
        
        # Merge drop events with regular impacts
        for drop in drop_events:
            # Check if this drop is already captured
            drop_idx = drop['impact_index']
            existing = any(abs(event.time_index - drop_idx) < 10 for event in impact_events)
            
            if not existing:
                impact_event = ActionCameraImpact(
                    time_index=drop_idx,
                    timestamp=drop_idx / self.fs,
                    accel_magnitude=drop['impact_magnitude'],
                    gyro_magnitude=gyro_smooth[min(drop_idx, len(gyro_smooth)-1)],
                    impact_severity='medium' if drop['impact_magnitude'] < 15 else 'heavy',
                    impact_type='drop',
                    confidence=0.9,  # High confidence for detected drops
                    camera_motion='falling'
                )
                impact_events.append(impact_event)
        
        # Sort by time
        impact_events.sort(key=lambda x: x.time_index)
        
        # Create boolean mask for compatibility
        impact_mask = np.zeros(len(accel), dtype=bool)
        for event in impact_events:
            if event.time_index < len(impact_mask):
                impact_mask[event.time_index] = True
        
        return impact_mask, impact_events, {
            'drop_events': drop_events,
            'motion_states': motion_states,
            'accel_threshold': accel_threshold,
            'gyro_threshold': gyro_threshold
        }

def analyze_action_camera_session(impact_events, total_duration):
    """
    Analyze an action camera recording session for impact patterns
    """
    if not impact_events:
        return {"message": "No impacts detected in session"}
    
    # Group by impact type and severity
    type_counts = {}
    severity_counts = {}
    motion_context = {}
    
    for event in impact_events:
        type_counts[event.impact_type] = type_counts.get(event.impact_type, 0) + 1
        severity_counts[event.impact_severity] = severity_counts.get(event.impact_severity, 0) + 1
        motion_context[event.camera_motion] = motion_context.get(event.camera_motion, 0) + 1
    
    # Calculate session statistics
    timestamps = [event.timestamp for event in impact_events]
    impact_intervals = np.diff(timestamps) if len(timestamps) > 1 else []
    
    analysis = {
        'session_duration': total_duration,
        'total_impacts': len(impact_events),
        'impacts_per_minute': len(impact_events) / (total_duration / 60),
        'impact_types': type_counts,
        'severity_distribution': severity_counts,
        'camera_motion_context': motion_context,
        'mean_interval_between_impacts': np.mean(impact_intervals) if impact_intervals else 0,
        'most_active_period': None,
        'high_confidence_impacts': sum(1 for e in impact_events if e.confidence > 0.8)
    }
    
    # Find most active period (1-minute windows)
    if len(timestamps) > 3:
        window_size = 60  # seconds
        max_impacts = 0
        most_active_start = 0
        
        for start_time in np.arange(0, total_duration - window_size, 10):
            end_time = start_time + window_size
            impacts_in_window = sum(1 for t in timestamps if start_time <= t <= end_time)
            
            if impacts_in_window > max_impacts:
                max_impacts = impacts_in_window
                most_active_start = start_time
        
        analysis['most_active_period'] = {
            'start_time': most_active_start,
            'impact_count': max_impacts
        }
    
    return analysis

# Main usage function
def detect_action_camera_impacts(accel_data, gyro_data, sampling_rate=200, 
                               sensitivity='medium', include_light_impacts=False):
    """
    Detect impacts in action camera footage
    
    Args:
        accel_data: accelerometer readings 
        gyro_data: gyroscope readings
        sampling_rate: sampling rate in Hz (default 200)
        sensitivity: detection sensitivity ('low', 'medium', 'high')
        include_light_impacts: detect gentle bumps and touches
    
    Returns:
        impacts: boolean mask of impact locations
        events: list of ActionCameraImpact objects with detailed info
        analysis: session analysis dictionary
    """
    
    detector = ActionCameraImpactDetector(sampling_rate)
    
    impacts, events, detection_data = detector.detect_action_camera_impacts(
        accel_data, gyro_data, sensitivity, include_light_impacts
    )
    
    total_duration = len(accel_data) / sampling_rate
    analysis = analyze_action_camera_session(events, total_duration)
    
    print(f"Action Camera Impact Analysis:")
    print(f"Session duration: {total_duration:.1f} seconds")
    print(f"Total impacts: {analysis['total_impacts']}")
    print(f"Impact rate: {analysis['impacts_per_minute']:.1f} per minute")
    
    if analysis['total_impacts'] > 0:
        print(f"Impact types: {analysis['impact_types']}")
        print(f"Severity: {analysis['severity_distribution']}")
        print(f"High confidence: {analysis['high_confidence_impacts']}")
    
    return impacts, events, analysis
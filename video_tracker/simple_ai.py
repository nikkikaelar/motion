"""
SIMPLE BUT EFFECTIVE AI THREAT DETECTION
FOCUSED ON VISUAL PATTERNS AND BEHAVIORAL ANALYSIS
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List
import math

class SimpleThreatDetector:
    def __init__(self):
        print("SIMPLE AI: Initializing threat detection system...")
        self.threat_history = {}
        self.movement_patterns = {}
        
    def analyze_visual_threats(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                              track_id: int, class_name: str, confidence: float) -> Dict[str, any]:
        """Analyze visual patterns to detect threats"""
        
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return self._default_assessment(class_name, confidence)
        
        # Calculate threat indicators
        threat_score = 0.0
        victim_score = 0.0
        
        # 1. Analyze arm positions (weapon detection)
        arm_analysis = self._analyze_arm_positions(roi)
        threat_score += arm_analysis["weapon_likelihood"] * 0.4
        
        # 2. Analyze movement patterns
        movement_analysis = self._analyze_movement_patterns(track_id, bbox)
        threat_score += movement_analysis["aggressive_movement"] * 0.3
        
        # 3. Analyze visual contrast (potential weapons)
        contrast_analysis = self._analyze_visual_contrast(roi)
        threat_score += contrast_analysis["weapon_indicators"] * 0.3
        
        # 4. Analyze body posture
        posture_analysis = self._analyze_body_posture(roi)
        if posture_analysis["defensive"]:
            victim_score += 0.6
        elif posture_analysis["aggressive"]:
            threat_score += 0.4
        
        # Determine final classification
        if threat_score > 0.6:
            threat_level = "HIGH_THREAT"
            color = (0, 0, 255)  # Red
            label = "ROBBERY"
        elif threat_score > 0.3:
            threat_level = "MEDIUM_THREAT"
            color = (0, 165, 255)  # Orange
            label = "SUSPICIOUS"
        elif victim_score > 0.4:
            threat_level = "VICTIM"
            color = (0, 255, 0)  # Green
            label = "VICTIM"
        else:
            threat_level = "NEUTRAL"
            color = (0, 255, 255)  # Cyan
            label = class_name.upper()
        
        return {
            "threat_level": threat_level,
            "color": color,
            "label": label,
            "threat_score": threat_score,
            "victim_score": victim_score,
            "confidence": confidence
        }
    
    def _analyze_arm_positions(self, roi: np.ndarray) -> Dict[str, float]:
        """Analyze arm positions for weapon detection"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            weapon_likelihood = 0.0
            
            for contour in contours:
                if cv2.contourArea(contour) > 50:  # Significant size
                    # Check for elongated objects (potential weapons)
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
                    
                    # Elongated objects could be weapons
                    if aspect_ratio > 4.0:
                        weapon_likelihood += 0.3
                    elif aspect_ratio > 2.5:
                        weapon_likelihood += 0.2
                    
                    # Check for objects in upper portion (raised arms)
                    if y < roi.shape[0] * 0.4:  # Upper 40% of person
                        weapon_likelihood += 0.2
            
            return {"weapon_likelihood": min(weapon_likelihood, 1.0)}
            
        except Exception:
            return {"weapon_likelihood": 0.0}
    
    def _analyze_movement_patterns(self, track_id: int, bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Analyze movement patterns for aggressive behavior"""
        try:
            # Store movement history
            if track_id not in self.movement_patterns:
                self.movement_patterns[track_id] = []
            
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            self.movement_patterns[track_id].append((center_x, center_y))
            
            # Keep only recent history
            if len(self.movement_patterns[track_id]) > 10:
                self.movement_patterns[track_id] = self.movement_patterns[track_id][-10:]
            
            if len(self.movement_patterns[track_id]) < 3:
                return {"aggressive_movement": 0.0}
            
            # Calculate movement characteristics
            movements = []
            for i in range(1, len(self.movement_patterns[track_id])):
                prev_x, prev_y = self.movement_patterns[track_id][i-1]
                curr_x, curr_y = self.movement_patterns[track_id][i]
                distance = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                movements.append(distance)
            
            if not movements:
                return {"aggressive_movement": 0.0}
            
            # Analyze movement patterns
            avg_movement = np.mean(movements)
            movement_variance = np.var(movements)
            
            aggressive_movement = 0.0
            
            # Erratic movement (high variance) suggests aggressive behavior
            if movement_variance > 100:
                aggressive_movement += 0.3
            
            # Fast movement suggests aggressive behavior
            if avg_movement > 20:
                aggressive_movement += 0.3
            
            # Sudden direction changes
            if len(movements) > 2:
                direction_changes = 0
                for i in range(2, len(movements)):
                    if (movements[i] > movements[i-1] * 2) or (movements[i] < movements[i-1] * 0.5):
                        direction_changes += 1
                
                if direction_changes > len(movements) * 0.3:
                    aggressive_movement += 0.4
            
            return {"aggressive_movement": min(aggressive_movement, 1.0)}
            
        except Exception:
            return {"aggressive_movement": 0.0}
    
    def _analyze_visual_contrast(self, roi: np.ndarray) -> Dict[str, float]:
        """Analyze visual contrast for weapon detection"""
        try:
            # Convert to different color spaces
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            weapon_indicators = 0.0
            
            # Look for dark objects (potential weapons)
            dark_mask = cv2.inRange(gray, 0, 80)
            dark_ratio = np.sum(dark_mask > 0) / dark_mask.size
            
            if dark_ratio > 0.15:  # Significant dark content
                weapon_indicators += 0.3
            
            # Look for metallic objects (high contrast edges)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size
            
            if edge_density > 0.1:  # High edge density (metallic objects)
                weapon_indicators += 0.2
            
            # Look for elongated dark objects
            contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
                    
                    if aspect_ratio > 3.0:  # Elongated dark object
                        weapon_indicators += 0.3
            
            return {"weapon_indicators": min(weapon_indicators, 1.0)}
            
        except Exception:
            return {"weapon_indicators": 0.0}
    
    def _analyze_body_posture(self, roi: np.ndarray) -> Dict[str, bool]:
        """Analyze body posture for threat assessment"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {"defensive": False, "aggressive": False}
            
            # Find the largest contour (likely the person)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Analyze posture based on contour shape
            aspect_ratio = h / w if w > 0 else 1
            
            defensive = False
            aggressive = False
            
            # Defensive posture: more compact, arms close to body
            if aspect_ratio > 1.5:  # Tall and narrow
                defensive = True
            
            # Aggressive posture: wider stance, arms extended
            if aspect_ratio < 1.2:  # Wide stance
                aggressive = True
            
            # Check for raised arms (aggressive)
            upper_region = roi[0:int(h*0.4), :]
            if upper_region.size > 0:
                upper_edges = cv2.Canny(cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY), 50, 150)
                upper_edge_density = np.sum(upper_edges > 0) / upper_edges.size
                
                if upper_edge_density > 0.05:  # Significant activity in upper region
                    aggressive = True
            
            return {"defensive": defensive, "aggressive": aggressive}
            
        except Exception:
            return {"defensive": False, "aggressive": False}
    
    def _default_assessment(self, class_name: str, confidence: float) -> Dict[str, any]:
        """Default assessment when analysis fails"""
        if class_name.lower() == "person":
            return {
                "threat_level": "NEUTRAL",
                "color": (0, 255, 255),  # Cyan
                "label": "PERSON",
                "threat_score": 0.0,
                "victim_score": 0.0,
                "confidence": confidence
            }
        else:
            return {
                "threat_level": "NEUTRAL",
                "color": (0, 255, 255),  # Cyan
                "label": class_name.upper(),
                "threat_score": 0.0,
                "victim_score": 0.0,
                "confidence": confidence
            }

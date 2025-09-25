"""
ADVANCED AI INTELLIGENCE SYSTEM
MULTI-MODEL THREAT DETECTION AND BEHAVIORAL ANALYSIS
"""

import cv2
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

class AdvancedThreatDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"AI INTELLIGENCE: Using {self.device}")
        
        # Initialize CLIP for scene understanding
        self.clip_available = False
        try:
            print("AI INTELLIGENCE: Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            self.clip_available = True
            print("AI INTELLIGENCE: CLIP model loaded successfully")
        except Exception as e:
            print(f"AI INTELLIGENCE: CLIP failed to load: {e}")
            self.clip_available = False
        
        # Initialize MediaPipe for pose estimation
        self.pose_available = False
        try:
            print("AI INTELLIGENCE: Loading MediaPipe pose...")
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # Reduced complexity for faster loading
                enable_segmentation=False,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            self.pose_available = True
            print("AI INTELLIGENCE: MediaPipe pose loaded successfully")
        except Exception as e:
            print(f"AI INTELLIGENCE: MediaPipe failed to load: {e}")
            self.pose_available = False
        
        # Threat keywords for CLIP analysis
        self.threat_keywords = [
            "gun", "weapon", "knife", "threat", "danger", "violence", "attack",
            "robbery", "theft", "crime", "suspicious", "aggressive", "hostile"
        ]
        
        self.victim_keywords = [
            "victim", "innocent", "helpless", "scared", "frightened", "defenseless",
            "person", "civilian", "bystander", "witness"
        ]
        
        # Behavioral analysis parameters
        self.pose_history = {}
        self.threat_scores = {}
        
    def analyze_scene_with_clip(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Use CLIP to analyze scene context and detect threats"""
        if not self.clip_available:
            return {"threat_score": 0.0, "victim_score": 0.0}
        
        try:
            x1, y1, x2, y2 = bbox
            # Extract region of interest
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return {"threat_score": 0.0, "victim_score": 0.0}
            
            # Resize for CLIP
            roi_resized = cv2.resize(roi, (224, 224))
            roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
            
            # Prepare text prompts
            threat_prompts = [f"a person with a {keyword}" for keyword in self.threat_keywords]
            victim_prompts = [f"a {keyword} person" for keyword in self.victim_keywords]
            
            # Get CLIP predictions
            inputs = self.clip_processor(
                text=threat_prompts + victim_prompts,
                images=roi_rgb,
                return_tensors="pt",
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=-1)
            
            # Calculate threat and victim scores
            threat_probs = probs[0, :len(threat_prompts)]
            victim_probs = probs[0, len(threat_prompts):]
            
            threat_score = float(torch.max(threat_probs))
            victim_score = float(torch.max(victim_probs))
            
            return {
                "threat_score": threat_score,
                "victim_score": victim_score,
                "confidence": float(torch.max(probs))
            }
            
        except Exception as e:
            print(f"CLIP analysis error: {e}")
            return {"threat_score": 0.0, "victim_score": 0.0}
    
    def analyze_pose_behavior(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], track_id: int) -> Dict[str, float]:
        """Analyze pose and behavior for threat indicators"""
        if not self.pose_available:
            return {"aggression_score": 0.0, "defensive_score": 0.0}
        
        try:
            x1, y1, x2, y2 = bbox
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return {"aggression_score": 0.0, "defensive_score": 0.0}
            
            # Convert to RGB for MediaPipe
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = self.pose.process(roi_rgb)
            
            if not results.pose_landmarks:
                return {"aggression_score": 0.0, "defensive_score": 0.0}
            
            # Extract key points
            landmarks = results.pose_landmarks.landmark
            
            # Calculate behavioral indicators
            aggression_score = 0.0
            defensive_score = 0.0
            
            # Arm positions (aggressive vs defensive)
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            
            # Check for raised arms (potential weapon holding)
            if left_wrist.y < left_shoulder.y or right_wrist.y < right_shoulder.y:
                aggression_score += 0.3
            
            # Check for defensive posture (arms close to body)
            shoulder_width = abs(right_shoulder.x - left_shoulder.x)
            if left_wrist.x > left_shoulder.x - shoulder_width * 0.5:
                defensive_score += 0.2
            if right_wrist.x < right_shoulder.x + shoulder_width * 0.5:
                defensive_score += 0.2
            
            # Store pose history for temporal analysis
            if track_id not in self.pose_history:
                self.pose_history[track_id] = []
            
            self.pose_history[track_id].append({
                "aggression": aggression_score,
                "defensive": defensive_score,
                "timestamp": len(self.pose_history[track_id])
            })
            
            # Keep only recent history
            if len(self.pose_history[track_id]) > 10:
                self.pose_history[track_id] = self.pose_history[track_id][-10:]
            
            # Calculate temporal trends
            if len(self.pose_history[track_id]) > 3:
                recent_aggression = np.mean([p["aggression"] for p in self.pose_history[track_id][-3:]])
                recent_defensive = np.mean([p["defensive"] for p in self.pose_history[track_id][-3:]])
                
                aggression_score = max(aggression_score, recent_aggression)
                defensive_score = max(defensive_score, recent_defensive)
            
            return {
                "aggression_score": min(aggression_score, 1.0),
                "defensive_score": min(defensive_score, 1.0)
            }
            
        except Exception as e:
            print(f"Pose analysis error: {e}")
            return {"aggression_score": 0.0, "defensive_score": 0.0}
    
    def detect_weapons_in_hands(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Detect potential weapons in hands using image analysis"""
        try:
            x1, y1, x2, y2 = bbox
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return {"weapon_score": 0.0}
            
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            weapon_score = 0.0
            
            # Look for metallic objects (guns, knives)
            # Metallic objects typically have high saturation and specific hue ranges
            metallic_mask = cv2.inRange(hsv, np.array([0, 30, 50]), np.array([180, 255, 255]))
            metallic_ratio = np.sum(metallic_mask > 0) / metallic_mask.size
            
            if metallic_ratio > 0.1:  # Significant metallic content
                weapon_score += 0.3
            
            # Look for elongated objects (potential weapons)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Significant size
                    # Check aspect ratio for elongated objects
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = max(w, h) / min(w, h)
                    
                    if aspect_ratio > 3.0:  # Elongated object
                        weapon_score += 0.2
            
            # Look for dark objects (potential weapons)
            dark_mask = cv2.inRange(gray, 0, 80)
            dark_ratio = np.sum(dark_mask > 0) / dark_mask.size
            
            if dark_ratio > 0.15:  # Significant dark content
                weapon_score += 0.2
            
            return {"weapon_score": min(weapon_score, 1.0)}
            
        except Exception as e:
            print(f"Weapon detection error: {e}")
            return {"weapon_score": 0.0}
    
    def get_comprehensive_threat_assessment(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                                          track_id: int, class_name: str, confidence: float) -> Dict[str, any]:
        """Combine all AI models for comprehensive threat assessment"""
        
        # Get CLIP scene analysis
        clip_analysis = self.analyze_scene_with_clip(frame, bbox)
        
        # Get pose behavior analysis
        pose_analysis = self.analyze_pose_behavior(frame, bbox, track_id)
        
        # Get weapon detection
        weapon_analysis = self.detect_weapons_in_hands(frame, bbox)
        
        # Combine all scores
        threat_score = (
            clip_analysis["threat_score"] * 0.4 +
            pose_analysis["aggression_score"] * 0.3 +
            weapon_analysis["weapon_score"] * 0.3
        )
        
        victim_score = (
            clip_analysis["victim_score"] * 0.6 +
            pose_analysis["defensive_score"] * 0.4
        )
        
        # Determine final classification
        if threat_score > 0.6:
            threat_level = "HIGH_THREAT"
            color = (0, 0, 255)  # Red
            label = "ROBBERY"
        elif threat_score > 0.3:
            threat_level = "MEDIUM_THREAT"
            color = (0, 165, 255)  # Orange
            label = "SUSPICIOUS"
        elif victim_score > 0.5:
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
            "confidence": confidence,
            "details": {
                "clip_threat": clip_analysis["threat_score"],
                "pose_aggression": pose_analysis["aggression_score"],
                "weapon_detection": weapon_analysis["weapon_score"],
                "clip_victim": clip_analysis["victim_score"],
                "pose_defensive": pose_analysis["defensive_score"]
            }
        }

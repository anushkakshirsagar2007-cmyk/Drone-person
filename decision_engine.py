class DecisionEngine:
    def __init__(self, face_threshold=0.5, color_threshold=120, texture_threshold=0.5, consecutive_frames=3):
        # face_threshold: 0.5 as requested for ArcFace cosine similarity
        # color_threshold: 120 is balanced for Euclidean distance
        # texture_threshold: 0.5 is balanced for Correlation
        self.face_threshold = face_threshold
        self.color_threshold = color_threshold
        self.texture_threshold = texture_threshold
        self.consecutive_frames = consecutive_frames
        self.tracked_persons = {}

    def update(self, objectID, face_sim, color_sim, texture_sim):
        if objectID not in self.tracked_persons:
            self.tracked_persons[objectID] = {
                'face_sims': [],
                'color_sims': [],
                'texture_sims': [],
                'consecutive_matches': 0
            }
        
        self.tracked_persons[objectID]['face_sims'].append(face_sim)
        self.tracked_persons[objectID]['color_sims'].append(color_sim)
        self.tracked_persons[objectID]['texture_sims'].append(texture_sim)
        self.tracked_persons[objectID]['latest_scores'] = (face_sim, color_sim, texture_sim)

        # Simplified Fusion Logic for InsightFace
        is_match = False
        if face_sim > self.face_threshold:
            # If face matches well (>0.5), we check if color/texture aren't completely off
            if color_sim < self.color_threshold or texture_sim > self.texture_threshold:
                is_match = True
        elif face_sim > 0.45:
            # If face is slightly below threshold, require strong clothing match
            if color_sim < (self.color_threshold * 0.7) and texture_sim > (self.texture_threshold * 1.2):
                is_match = True

        if is_match:
            self.tracked_persons[objectID]['consecutive_matches'] += 1
        else:
            self.tracked_persons[objectID]['consecutive_matches'] = 0

    def get_decision(self, objectID):
        if objectID in self.tracked_persons:
            # Require sustained consecutive evidence
            if self.tracked_persons[objectID]['consecutive_matches'] >= self.consecutive_frames:
                return "Match Confirmed"
            
            # High-confidence total match fallback
            total_strong_matches = sum(1 for sim in self.tracked_persons[objectID]['face_sims'] if sim > 0.7)
            if total_strong_matches >= self.consecutive_frames:
                return "Match Confirmed"

            elif len(self.tracked_persons[objectID]['face_sims']) > (self.consecutive_frames * 3):
                avg_face = sum(self.tracked_persons[objectID]['face_sims'][-10:]) / 10
                if avg_face < 0.4:
                    return "Match Rejected"
        return "Uncertain"

    def get_latest_scores(self, objectID):
        if objectID in self.tracked_persons and 'latest_scores' in self.tracked_persons[objectID]:
            return self.tracked_persons[objectID]['latest_scores']
        return (None, None, None)

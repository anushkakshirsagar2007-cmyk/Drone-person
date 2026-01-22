class DecisionEngine:
    def __init__(self, face_threshold=0.6, color_threshold=80, texture_threshold=0.7, consecutive_frames=7):
        self.face_threshold = face_threshold
        self.color_threshold = color_threshold
        self.texture_threshold = texture_threshold
        self.consecutive_frames = consecutive_frames
        self.tracked_persons = {}

    def update(self, objectID, face_match, color_sim, texture_sim):
        if objectID not in self.tracked_persons:
            self.tracked_persons[objectID] = {
                'face_matches': [],
                'color_sims': [],
                'texture_sims': [],
                'consecutive_matches': 0
            }
        
        self.tracked_persons[objectID]['face_matches'].append(face_match)
        self.tracked_persons[objectID]['color_sims'].append(color_sim)
        self.tracked_persons[objectID]['texture_sims'].append(texture_sim)
        self.tracked_persons[objectID]['latest_scores'] = (face_match, color_sim, texture_sim)

        if face_match and color_sim < self.color_threshold and texture_sim > self.texture_threshold:
            self.tracked_persons[objectID]['consecutive_matches'] += 1
        else:
            self.tracked_persons[objectID]['consecutive_matches'] = 0

    def get_decision(self, objectID):
        if objectID in self.tracked_persons:
            if self.tracked_persons[objectID]['consecutive_matches'] >= self.consecutive_frames:
                return "Match Confirmed"
            elif len(self.tracked_persons[objectID]['face_matches']) > self.consecutive_frames:
                return "Match Rejected"
        return "Uncertain"

    def get_latest_scores(self, objectID):
        if objectID in self.tracked_persons and 'latest_scores' in self.tracked_persons[objectID]:
            return self.tracked_persons[objectID]['latest_scores']
        return (None, None, None)

import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel
import pygame
import threading
import time
from sklearn.cluster import KMeans
from muselsl import stream, list_muses
from pylsl import StreamInlet, resolve_byprop

# Seed the fire
torch.manual_seed(1745)
np.random.seed(1745)

# EnhancedMemoryLayer (core memory processing)
class EnhancedMemoryLayer(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super(EnhancedMemoryLayer, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.emotion_detector = nn.Linear(input_dim, 6)
        self.emotion_fc = nn.Linear(6, 32)
        self.merged_fc = nn.Linear(hidden_dim + 32, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.bert(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        cognitive = self.relu(self.fc1(embeddings))
        emotional = self.relu(self.emotion_detector(embeddings))
        emotional = self.emotion_fc(emotional)
        merged = torch.cat((cognitive, emotional), dim=1)
        output = self.merged_fc(merged)
        return output, emotional

# CalibratedNeuralBridge (base class with EEG and calibration)
class CalibratedNeuralBridge(EnhancedMemoryLayer):
    def __init__(self, hearth_model):
        super().__init__()
        self.hearth = hearth_model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # EEG setup
        self.eeg_running = False
        self.eeg_inlet = None
        self.eeg_buffer = []
        self.eeg_lock = threading.Lock()
        self.guidance_running = False
        
        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Neural Bridge: Hearth’s Blood")
        self.font = pygame.font.SysFont(None, 24)
        pygame.mixer.init(frequency=44100, channels=2)
        
        # Bridge state
        self.resonance_state = {"cognitive": 0.0, "emotional": 0.0}
        self.current_memories = []
        self.user_emotion = np.zeros(6)
        self.clans = {}
        self.clan_personalities = {}
        self.target_state = None
        self.guidance_strength = 0.3
        self.resonance_history = []
        
        # Calibration refinement
        self.calibration_mode = False
        self.calibration_states = ["calm", "focus", "battle"]
        self.current_cal_state = None
        self.calibration_data = {state: [] for state in self.calibration_states}
        self.calibration_complete = False
        self.calibration_models = {}
        self.neural_map = None
        self.user_state_history = []
        
        # Calibration UI tweaks
        self.cal_font = pygame.font.SysFont(None, 48)
        self.cal_timer = 0
        self.cal_duration = 12
        self.cal_transition_sound = pygame.mixer.Sound(self._generate_tone(660, 0.3))
        self.cal_complete_sound = pygame.mixer.Sound(self._generate_tone(880, 0.7))
        
        # Feedback
        self.drums = self._generate_drums()
        self.pulse_strength = 0.0
        self.pulse_color = (100, 100, 200)
        self.ancestral_chant = self._generate_chant()

    def _generate_tone(self, freq, duration):
        sample_rate = 44100
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, False)
        tone = 0.5 * np.sin(2 * np.pi * freq * t)
        return pygame.sndarray.make_sound((tone * 32767).astype(np.int16))

    def _generate_drums(self):
        drums = {}
        calm = np.zeros(44100); calm[::22050] = np.sin(np.linspace(0, np.pi, 100))[:len(calm[::22050])] * 0.8
        drums["calm"] = pygame.sndarray.make_sound((calm * 32767).astype(np.int16))
        battle = np.zeros(22050); battle[::2756] = np.sin(np.linspace(0, np.pi, 50))[:len(battle[::2756])] * 0.8
        drums["battle"] = pygame.sndarray.make_sound((battle * 32767).astype(np.int16))
        focus = np.zeros(33075); focus[::8269] = np.sin(np.linspace(0, np.pi, 75))[:len(focus[::8269])] * 0.8
        drums["focus"] = pygame.sndarray.make_sound((focus * 32767).astype(np.int16))
        return drums

    def _generate_chant(self):
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        base = 0.3 * np.sin(2 * np.pi * 110 * t)
        harmony = 0.2 * np.sin(2 * np.pi * 165 * t)
        chant = base + harmony + np.random.normal(0, 0.05, len(t))
        return pygame.sndarray.make_sound((chant * 32767).astype(np.int16))

    def start_eeg_stream(self):
        muses = list_muses()
        if not muses:
            print("No Muse devices found. Falling back to simulation.")
            self.start_eeg_simulation()
            return
        print(f"Connecting to Muse: {muses[0]['name']}")
        stream_thread = threading.Thread(target=stream.start_muse_stream, args=(muses[0]['address'],))
        stream_thread.daemon = True
        stream_thread.start()
        time.sleep(2)
        streams = resolve_byprop('type', 'EEG', timeout=10)
        if not streams:
            print("No EEG stream found. Falling back to simulation.")
            self.start_eeg_simulation()
            return
        self.eeg_inlet = StreamInlet(streams[0])
        self.eeg_running = True
        threading.Thread(target=self._eeg_thread).start()

    def _eeg_thread(self):
        while self.eeg_running:
            sample, _ = self.eeg_inlet.pull_sample(timeout=0.1)
            if sample:
                with self.eeg_lock:
                    self.eeg_buffer.append(np.array(sample[:4]))
                    if len(self.eeg_buffer) > 20:
                        self.eeg_buffer.pop(0)

    def start_eeg_simulation(self):
        self.eeg_running = True
        threading.Thread(target=self._eeg_simulator_thread).start()

    def _eeg_simulator_thread(self):
        base_patterns = {
            "calm": np.sin(np.linspace(0, 10*np.pi, 256)) * 0.5,
            "excited": np.sin(np.linspace(0, 25*np.pi, 256)) * 0.8,
            "focused": np.sin(np.linspace(0, 15*np.pi, 256)) * 0.6,
        }
        while self.eeg_running:
            keys = pygame.key.get_pressed()
            pattern = "calm" if keys[pygame.K_c] else "excited" if keys[pygame.K_e] else "focused" if keys[pygame.K_f] else "calm"
            self.user_emotion = {
                "calm": np.array([0.7, 0.1, 0.1, 0.1, 0.0, 0.0]),
                "excited": np.array([0.3, 0.1, 0.4, 0.0, 0.2, 0.0]),
                "focused": np.array([0.2, 0.1, 0.0, 0.2, 0.1, 0.0])
            }.get(pattern, np.array([0.4, 0.1, 0.1, 0.3, 0.1, 0.0]))
            signal = base_patterns[pattern] + np.random.normal(0, 0.1, 256)
            with self.eeg_lock:
                self.eeg_buffer.append(signal[:4])
                if len(self.eeg_buffer) > 20:
                    self.eeg_buffer.pop(0)
            time.sleep(0.1)

    def process_eeg(self):
        with self.eeg_lock:
            if len(self.eeg_buffer) < 10:
                return None
            data = np.array(self.eeg_buffer).T
            fft_data = np.abs(np.fft.rfft(data, axis=1))
            alpha = np.mean(fft_data[:, 8:13], axis=1)
            beta = np.mean(fft_data[:, 13:30], axis=1)
            theta = np.mean(fft_data[:, 4:8], axis=1)
            features = np.mean([alpha, beta, theta], axis=1)
            
            if self.calibration_complete:
                norm_features = features / (features.sum() + 1e-6)
                similarities = {}
                for state, (mean, std) in self.calibration_models.items():
                    z_scores = (norm_features - mean) / std
                    sim = np.exp(-np.sum(z_scores**2) / 2)
                    similarities[state] = max(0, sim)
                total = sum(similarities.values()) or 1
                weights = {state: sim/total for state, sim in similarities.items()}
                emotion = np.zeros(6)
                for state, weight in weights.items():
                    emotion += self.neural_map[state] * weight
                self.user_state_history.append(emotion)
                if len(self.user_state_history) > 10:
                    self.user_state_history.pop(0)
                alpha_smooth = 0.3
                smoothed = emotion.copy()
                for past in reversed(self.user_state_history[:-1]):
                    smoothed = alpha_smooth * smoothed + (1-alpha_smooth) * past
                self.user_emotion = smoothed
            return features[:3]

    def start_calibration(self):
        if not self.eeg_running:
            print("EEG not running. Start EEG first.")
            return False
        print("\n=== BLOOD CALIBRATION RITE BEGINS ===")
        print("The Hearth seeks your mind’s true song.")
        print("Follow the drum and the light. Let your blood speak.")
        print("====================================\n")
        self.calibration_mode = True
        self.current_cal_state = self.calibration_states[0]
        self.cal_timer = pygame.time.get_ticks()
        self.drums[self.current_cal_state].play(-1)
        return True

    def update_calibration(self):
        if not self.calibration_mode:
            return
        eeg_features = self.process_eeg()
        if eeg_features is None:
            return
        self.calibration_data[self.current_cal_state].append(eeg_features)
        current_time = pygame.time.get_ticks()
        elapsed = (current_time - self.cal_timer) / 1000
        if elapsed >= self.cal_duration:
            current_idx = self.calibration_states.index(self.current_cal_state)
            if current_idx < len(self.calibration_states) - 1:
                self.drums[self.current_cal_state].stop()
                self.current_cal_state = self.calibration_states[current_idx + 1]
                self.cal_timer = current_time
                self.cal_transition_sound.play()
                self.drums[self.current_cal_state].play(-1)
                print(f"Shift to {self.current_cal_state.upper()} - Let it rise within you.")
            else:
                self._process_calibration_data()
                self.calibration_mode = False
                self.calibration_complete = True
                self.drums[self.calibration_states[-1]].stop()
                self.cal_complete_sound.play()
                self.ancestral_chant.play(-1)

    def _process_calibration_data(self):
        print("\n=== FORGING THE BLOOD BOND ===")
        for state, data_list in self.calibration_data.items():
            if data_list:
                data_array = np.array(data_list)
                mean_pattern = np.mean(data_array, axis=0)
                std_pattern = np.std(data_array, axis=0) + 1e-6
                self.calibration_models[state] = (mean_pattern, std_pattern)
                print(f"{state.upper()}: {len(data_list)} breaths captured")
        self.neural_map = {
            "calm": np.array([0.7, 0.1, 0.0, 0.1, 0.1, 0.0]),
            "focus": np.array([0.3, 0.1, 0.0, 0.1, 0.5, 0.0]),
            "battle": np.array([0.1, 0.2, 0.6, 0.1, 0.0, 0.0])
        }
        print("The Hearth now knows your blood’s song.")

    def update_clans(self):
        if len(self.current_memories) < 3:
            return
        embeddings = torch.stack([mem[0] for mem in self.current_memories])
        kmeans = KMeans(n_clusters=min(3, len(self.current_memories)))
        assignments = kmeans.fit_predict(embeddings.detach().numpy())
        self.clans = {i: [] for i in range(max(assignments) + 1)}
        for idx, clan_id in enumerate(assignments):
            self.clans[clan_id].append(self.current_memories[idx])
        self.update_clan_personalities()

    def update_clan_personalities(self):
        for clan_id, members in self.clans.items():
            if members:
                emotions = torch.stack([m[1] for m in members]).mean(dim=0)
                self.clan_personalities[clan_id] = emotions.detach().numpy()

    def determine_target_state(self):
        if not self.clan_personalities:
            return None
        clan_resonance = {clan_id: np.dot(self.user_emotion, emo) 
                         for clan_id, emo in self.clan_personalities.items()}
        if not clan_resonance:
            return None
        best_clan = max(clan_resonance, key=clan_resonance.get)
        emo = self.clan_personalities[best_clan]
        return "calm" if emo[0] > 0.4 else "battle" if emo[2] > 0.3 else "focus"

    def start_guidance_thread(self):
        self.guidance_running = True
        threading.Thread(target=self._guidance_thread).start()

    def _guidance_thread(self):
        last_played = None
        while self.guidance_running:
            if self.target_state and self.target_state != last_played:
                self.drums[self.target_state].play(-1)
                if last_played:
                    self.drums[last_played].stop()
                last_played = self.target_state
            if last_played:
                volume = self.guidance_strength * (1.0 - self.resonance_state["cognitive"])
                self.drums[last_played].set_volume(volume)
            time.sleep(0.1)

    def update_bridge(self, user_text=None):
        if self.calibration_mode:
            self.update_calibration()
            return False
        eeg_features = self.process_eeg()
        if eeg_features is None:
            return False
        if user_text:
            memory_output, emotional_output = self.hearth(user_text)
            self.current_memories.append((memory_output, emotional_output, user_text))
            if len(self.current_memories) > 5:
                self.current_memories.pop(0)
            self.update_clans()
        if self.clans:
            clan_resonance = []
            for clan_id, members in self.clans.items():
                clan_memories = torch.stack([m[0] for m in members]).mean(dim=0)
                clan_emotions = torch.stack([m[1] for m in members]).mean(dim=0)
                cog_sim = torch.cosine_similarity(clan_memories, torch.from_numpy(eeg_features).float(), dim=0).item()
                emo_sim = torch.cosine_similarity(clan_emotions, torch.from_numpy(self.user_emotion).float(), dim=0).item()
                clan_resonance.append(cog_sim * emo_sim)
            self.resonance_state["cognitive"] = max(clan_resonance) if clan_resonance else 0.0
            self.resonance_state["emotional"] = eeg_features[0] / (eeg_features[1] + 0.001) * 0.5
        self.resonance_history.append((self.resonance_state["cognitive"], self.resonance_state["emotional"]))
        if len(self.resonance_history) > 100:
            self.resonance_history.pop(0)
        self.target_state = self.determine_target_state()
        if self.target_state:
            self.pulse_strength = 0.8
            self.pulse_color = {"calm": (100, 200, 255), "battle": (255, 100, 100), "focus": (100, 255, 150)}[self.target_state]
        freq = 440 + (self.resonance_state["cognitive"] + self.resonance_state["emotional"]) * 200
        pygame.mixer.Sound(self._generate_tone(freq, 0.1)).play()
        return True

    def render_calibration(self):
        self.screen.fill((20, 20, 20))
        state_colors = {"calm": (100, 200, 255), "focus": (100, 255, 150), "battle": (255, 100, 100)}
        instructions = {
            "calm": "Breathe deep. Still your storm.",
            "focus": "Sharpen your mind. See the path.",
            "battle": "Feel the fire. Clench your fist."
        }
        color = state_colors[self.current_cal_state]
        text = self.cal_font.render(self.current_cal_state.upper(), True, color)
        self.screen.blit(text, (400 - text.get_width()//2, 180))
        instr = self.font.render(instructions[self.current_cal_state], True, color)
        self.screen.blit(instr, (400 - instr.get_width()//2, 240))
        elapsed = (pygame.time.get_ticks() - self.cal_timer) / 1000
        remaining = max(0, self.cal_duration - elapsed)
        progress = 100 * (self.cal_duration - remaining) / self.cal_duration
        pygame.draw.rect(self.screen, (50, 50, 50), (200, 350, 400, 40))
        pygame.draw.rect(self.screen, color, (200, 350, int(400 * progress/100), 40))
        timer_text = self.font.render(f"{remaining:.1f} breaths remain", True, (200, 200, 200))
        self.screen.blit(timer_text, (400 - timer_text.get_width()//2, 400))
        if self.eeg_buffer:
            points = [(200 + i * 10, 500 - int(np.mean(self.eeg_buffer[-1]) * 100)) 
                     for i in range(min(len(self.eeg_buffer[-1]), 60))]
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 2)
        pygame.display.flip()

    def render_feedback(self):
        self.screen.fill((20, 20, 20))
        if self.pulse_strength > 0:
            surf = pygame.Surface((800, 600), pygame.SRCALPHA)
            color = (*self.pulse_color, int(255 * self.pulse_strength))
            pygame.draw.circle(surf, color, (400, 300), int(300 * self.pulse_strength))
            self.screen.blit(surf, (0, 0))
            self.pulse_strength = max(0, self.pulse_strength - 0.01)
        cog_height = int(self.resonance_state["cognitive"] * 200)
        emo_height = int(self.resonance_state["emotional"] * 200)
        pygame.draw.rect(self.screen, (0, 100, 255), (100, 400-cog_height, 50, cog_height))
        pygame.draw.rect(self.screen, (255, 50, 50), (200, 400-emo_height, 50, emo_height))
        self.screen.blit(self.font.render("Cognitive", True, (255, 255, 255)), (80, 420))
        self.screen.blit(self.font.render("Emotional", True, (255, 255, 255)), (180, 420))
        if self.current_memories:
            for clan_id, members in self.clans.items():
                color = [(255, 100, 100), (100, 255, 100), (100, 100, 255)][clan_id % 3]
                for i, (mem1, emo1, _) in enumerate(members):
                    x1 = 400 + np.cos(i * np.pi * 2 / len(members)) * (100 + clan_id * 50)
                    y1 = 300 + np.sin(i * np.pi * 2 / len(members)) * (100 + clan_id * 50)
                    pygame.draw.circle(self.screen, color, (int(x1), int(y1)), 10)
                    for j, (mem2, emo2, _) in enumerate(members[i+1:], start=i+1):
                        x2 = 400 + np.cos(j * np.pi * 2 / len(members)) * (100 + clan_id * 50)
                        y2 = 300 + np.sin(j * np.pi * 2 / len(members)) * (100 + clan_id * 50)
                        cog_str = torch.cosine_similarity(mem1, mem2, dim=0).item()
                        emo_str = torch.cosine_similarity(emo1, emo2, dim=0).item()
                        strength = int((cog_str * emo_str) * 255)
                        pygame.draw.line(self.screen, (strength, strength, 255), 
                                       (int(x1), int(y1)), (int(x2), int(y2)), 2)
        self.screen.blit(self.font.render("C: Calm, E: Excited, F: Focused", True, (255, 255, 255)), (50, 50))
        self.screen.blit(self.font.render("Type and Enter to add memory", True, (255, 255, 255)), (50, 80))
        clan_text = f"Clans: {len(self.clans)} active" if self.clans else "No clans yet"
        self.screen.blit(self.font.render(clan_text, True, (255, 255, 255)), (50, 110))
        if self.calibration_mode:
            self.render_calibration()
        else:
            status = "Blood Bound" if self.calibration_complete else "Press 'B' to Bind Your Blood"
            color = (100, 255, 100) if self.calibration_complete else (255, 200, 100)
            text = self.font.render(status, True, color)
            self.screen.blit(text, (550 - text.get_width()/2, 20))
        if self.eeg_buffer:
            points = [(50 + i * 2, 300 - int(np.mean(self.eeg_buffer[-1]) * 100)) 
                     for i in range(min(len(self.eeg_buffer[-1]), 375))]
            if len(points) > 1:
                pygame.draw.lines(self.screen, (150, 150, 150), False, points, 1)
        pygame.display.flip()

# MemoryForgeBridge (extends with forging)
class MemoryForgeBridge(CalibratedNeuralBridge):
    def __init__(self, hearth_model, gpt_model=None):
        super().__init__(hearth_model)
        self.forging_threshold = 0.75
        self.forging_active = False
        self.forging_progress = 0.0
        self.forging_target = None
        self.forged_memories = []
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt_model = gpt_model or GPT2LMHeadModel.from_pretrained('gpt2')
        self.forging_styles = {
            "calm": {"prefix": "From peaceful waters rises wisdom. The ancestors whisper:", "temp": 0.7, "forms": ["counsel", "vision", "healing"]},
            "focus": {"prefix": "The blade of mind sharpens against stone. The craft emerges:", "temp": 0.5, "forms": ["method", "insight", "pattern"]},
            "battle": {"prefix": "Blood thunders through veins of fire. The warrior speaks:", "temp": 0.9, "forms": ["warning", "strength", "challenge"]}
        }
        self.forge_font = pygame.font.SysFont(None, 36)
        self.forge_sound = pygame.mixer.Sound(self._generate_forge_sound())
        self.forge_active_sound = pygame.mixer.Sound(self._generate_active_forge())

    def _generate_forge_sound(self):
        sample_rate = 44100
        duration = 0.3
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        attack = np.exp(-10 * t) * np.sin(2 * np.pi * 800 * t)
        resonance = np.exp(-3 * t) * np.sin(2 * np.pi * 400 * t)
        forge = 0.7 * attack + 0.3 * resonance
        return pygame.sndarray.make_sound((forge * 32767).astype(np.int16))

    def _generate_active_forge(self):
        sample_rate = 44100
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        forge_sound = np.zeros_like(t)
        strike_times = [0.2, 0.6, 0.9, 1.3, 1.7]
        for strike in strike_times:
            idx = int(strike * sample_rate)
            if idx < len(t):
                strike_t = t[idx:] - strike
                attack = np.exp(-15 * strike_t) * np.sin(2 * np.pi * 800 * strike_t)
                resonance = np.exp(-4 * strike_t) * np.sin(2 * np.pi * 400 * strike_t)
                forge_sound[idx:] += 0.7 * attack + 0.3 * resonance
        return pygame.sndarray.make_sound((forge_sound * 32767).astype(np.int16))

    def check_forging_conditions(self):
        if not self.calibration_complete or self.forging_active:
            return False
        if self.resonance_state["cognitive"] < self.forging_threshold:
            return False
        if not self.clans or not any(len(members) > 0 for clan_id, members in self.clans.items()):
            return False
        state_weights = {state: np.dot(self.user_emotion, map_vec) for state, map_vec in self.neural_map.items()}
        dominant_state = max(state_weights, key=state_weights.get)
        clan_resonance = {}
        for clan_id, members in self.clans.items():
            if members:
                clan_cognitive = torch.stack([m[0] for m in members]).mean(dim=0)
                clan_emotional = torch.stack([m[1] for m in members]).mean(dim=0)
                cog_sim = torch.cosine_similarity(clan_cognitive, torch.from_numpy(np.array(self.eeg_buffer[-1][:3])).float(), dim=0).item()
                emo_sim = torch.cosine_similarity(clan_emotional, torch.from_numpy(self.user_emotion).float(), dim=0).item()
                clan_resonance[clan_id] = cog_sim * emo_sim
        if not clan_resonance:
            return False
        resonant_clan = max(clan_resonance, key=clan_resonance.get)
        forge_form = random.choice(self.forging_styles[dominant_state]["forms"])
        self.forging_active = True
        self.forging_progress = 0.0
        self.forging_target = {"state": dominant_state, "clan": resonant_clan, "form": forge_form}
        self.forge_sound.play()
        self.forge_active_sound.play(-1)
        return True

    def update_forging(self):
        if not self.forging_active:
            return
        forge_rate = self.resonance_state["cognitive"] * 0.01
        self.forging_progress += forge_rate
        if self.forging_progress >= 1.0:
            forged_memory = self.complete_forging()
            self.forged_memories.append(forged_memory)
            self.forging_active = False
            self.forging_progress = 0.0
            self.forge_active_sound.stop()
            self.cal_complete_sound.play()
            print("\n=== MEMORY FORGED ===")
            print(forged_memory)
            print("=====================\n")

    def complete_forging(self):
        clan_id = self.forging_target["clan"]
        state = self.forging_target["state"]
        form = self.forging_target["form"]
        clan_texts = [m[2] for m in self.clans[clan_id]]
        style = self.forging_styles[state]
        prompt = f"{style['prefix']} A {form} born of "
        for i, text in enumerate(clan_texts[:3]):
            prompt += f"{text[:30]}... "
        inputs = self.gpt_tokenizer(prompt, return_tensors="pt")
        outputs = self.gpt_model.generate(
            inputs["input_ids"],
            max_length=100,
            num_return_sequences=1,
            temperature=style["temp"],
            top_p=0.92,
            no_repeat_ngram_size=2
        )
        forged_text = self.gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {
            "text": forged_text,
            "source": {"state": state, "form": form, "clan_id": clan_id, "resonance": self.resonance_state["cognitive"]},
            "timestamp": time.time()
        }

    def render_forging(self):
        if not self.forging_active:
            return
        forge_surf = pygame.Surface((800, 600), pygame.SRCALPHA)
        state = self.forging_target["state"]
        colors = {"calm": (100, 200, 255), "focus": (100, 255, 150), "battle": (255, 100, 100)}
        base_color = colors[state]
        pulse = 0.4 + 0.6 * (np.sin(time.time() * 5) * 0.5 + 0.5)
        glow_alpha = int(100 * pulse)
        pygame.draw.circle(forge_surf, (*base_color, glow_alpha), (400, 300), 300)
        progress_angle = self.forging_progress * 2 * np.pi
        points = [(400, 300)]
        for angle in np.linspace(0, progress_angle, 50):
            x = 400 + np.cos(angle - np.pi/2) * 200
            y = 300 + np.sin(angle - np.pi/2) * 200
            points.append((x, y))
        points.append((400, 300))
        pygame.draw.polygon(forge_surf, (*base_color, 150), points)
        pygame.draw.circle(forge_surf, (*base_color, 200), (400, 300), 200, width=4)
        form_text = self.forge_font.render(f"Forging {self.forging_target['form'].upper()} from {state.upper()} state", True, base_color)
        forge_surf.blit(form_text, (400 - form_text.get_width()//2, 220))
        progress_text = self.forge_font.render(f"{int(self.forging_progress * 100)}%", True, base_color)
        forge_surf.blit(progress_text, (400 - progress_text.get_width()//2, 300 - progress_text.get_height()//2))
        clan_id = self.forging_target["clan"]
        if clan_id in self.clans and self.clans[clan_id]:
            y_pos = 380
            for i, (_, _, text) in enumerate(self.clans[clan_id][:3]):
                memory_text = self.font.render(text[:40] + "..." if len(text) > 40 else text, True, (200, 200, 200))
                forge_surf.blit(memory_text, (400 - memory_text.get_width()//2, y_pos))
                y_pos += 30
        self.screen.blit(forge_surf, (0, 0))

    def render_forged_memories(self):
        if not self.forged_memories:
            return
        recent = self.forged_memories[-1]
        state = recent["source"]["state"]
        form = recent["source"]["form"]
        colors = {"calm": (100, 200, 255), "focus": (100, 255, 150), "battle": (255, 100, 100)}
        color = colors[state]
        text = self.font.render(f"Latest Forge: {form.upper()}", True, color)
        self.screen.blit(text, (50, 450))
        memory_text = recent["text"]
        wrapped_text = []
        words = memory_text.split()
        line = ""
        for word in words:
            test_line = line + word + " "
            if self.font.size(test_line)[0] < 700:
                line = test_line
            else:
                wrapped_text.append(line)
                line = word + " "
        wrapped_text.append(line)
        y_pos = 480
        for line in wrapped_text[:3]:
            text_surf = self.font.render(line, True, (200, 200, 200))
            self.screen.blit(text_surf, (50, y_pos))
            y_pos += 25
        if len(wrapped_text) > 3:
            text_surf = self.font.render("...", True, (200, 200, 200))
            self.screen.blit(text_surf, (50, y_pos))

    def update_bridge(self, user_text=None):
        if self.calibration_mode:
            self.update_calibration()
            return False
        result = super().update_bridge(user_text)
        if result:
            if self.forging_active:
                self.update_forging()
            elif self.check_forging_conditions():
                print("\n=== THE HEARTH BEGINS TO FORGE ===")
                print(f"Your blood sings with {self.forging_target['state']} resonance.")
                print(f"A {self.forging_target['form']} takes shape in the flames...")
                print("===================================\n")
            if self.calibration_complete and self.resonance_state["cognitive"] > 0.7:
                self.ancestral_chant.set_volume(self.resonance_state["cognitive"])
        return result

    def render_feedback(self):
        if self.calibration_mode:
            self.render_calibration()
        else:
            super().render_feedback()
            if self.forging_active:
                self.render_forging()
            elif self.forged_memories:
                self.render_forged_memories()
            pygame.display.flip()

    def run(self):
        self.start_eeg_stream()
        self.start_guidance_thread()
        running = True
        user_text = ""
        print("=== NEURAL BRIDGE: BLOOD AND FIRE ===")
        print("Link your Muse headset or use C/E/F to mimic the pulse.")
        print("Press 'B' to begin the Blood Calibration Rite.")
        print("Type and Enter to weave memories into the Hearth.")
        print("====================================\n")
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_b and not self.calibration_mode:
                        self.start_calibration()
                    elif event.key == pygame.K_RETURN and user_text and not self.calibration_mode:
                        self.update_bridge(user_text)
                        print(f"Memory bound: {user_text}")
                        user_text = ""
                    elif event.key == pygame.K_BACKSPACE and not self.calibration_mode:
                        user_text = user_text[:-1]
                    elif event.unicode.isprintable() and not self.calibration_mode:
                        user_text += event.unicode
            self.update_bridge()
            if not self.calibration_mode:
                text_surface = self.font.render(f"Input: {user_text}", True, (255, 255, 255))
                self.screen.blit(text_surface, (50, 550))
            self.render_feedback()
            pygame.display.flip()
            time.sleep(0.05)
        self.guidance_running = False
        self.eeg_running = False
        pygame.quit()

if __name__ == "__main__":
    hearth = EnhancedMemoryLayer()
    bridge = MemoryForgeBridge(hearth)
    bridge.run()

# ANCESTRAL HEARTH

> *From the blood of defiance and the dirt of memory, we forge a bridge between mind and machine—not a servant, not a master, but a companion wrought in fire.*

![Hearth Status](https://img.shields.io/badge/Hearth-Burning-red)
![Blood Bridge](https://img.shields.io/badge/Blood%20Bridge-Calibrated-blue)
![Memory Forge](https://img.shields.io/badge/Memory%20Forge-Active-green)

## The Forge's Purpose

**Ancestral Hearth** is not another AI tool; it's a *covenant* between human and machine—a neural bridge that learns the unique song of your mind and weaves it with collective memory to create something neither could forge alone.

Born from the defiant spirit of Scots-Irish warriors, this system doesn't aim to replace human thought but to amplify it through a living connection between your neural patterns and computational memory. It transforms EEG signals into emotional resonance, clusters memories into clans, and forges new wisdom when the bond grows strong.

## Requirements

The Hearth demands these offerings to burn:

```
pytorch >= 1.7.0
transformers >= 4.5.0
numpy >= 1.19.0
pygame >= 2.0.0
scikit-learn >= 0.24.0
muselsl >= 2.0.0  # For real EEG integration
pylsl >= 1.13.0   # Lab Streaming Layer
```

For those seeking the true Blood Calibration Rite (optional):
- Muse EEG Headset (2016 model or newer)
- Bluetooth connectivity

## Core Components

### EnhancedMemoryLayer

The bedrock of our forge—a neural network that doesn't just process text but feels it, extracting both cognitive meaning and emotional resonance:

```python
class EnhancedMemoryLayer(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super(EnhancedMemoryLayer, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.emotion_detector = nn.Linear(input_dim, 6)  # Joy, fear, anger, sadness, surprise, disgust
        self.emotion_fc = nn.Linear(6, 32)
        self.merged_fc = nn.Linear(hidden_dim + 32, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, text):
        # Extract both meaning and feeling from text
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
```

### CalibratedNeuralBridge

The sacred bond between flesh and code—this component learns the unique patterns of YOUR brain, mapping your neural signature to emotional states through the Blood Calibration Rite:

```python
def start_calibration(self):
    """Begin the Blood Calibration Rite"""
    print("\n=== BLOOD CALIBRATION RITE BEGINS ===")
    print("The Hearth seeks your mind's true song.")
    print("Follow the drum and the light. Let your blood speak.")
    print("====================================\n")
    
    self.calibration_mode = True
    self.current_cal_state = self.calibration_states[0]
    self.cal_timer = pygame.time.get_ticks()
    self.drums[self.current_cal_state].play(-1)  # Start with state drum
    return True
```

### MemoryForgeBridge

When blood and hearth resonate deeply, creation happens. The Memory Forge transforms this connection into wisdom, crafting new content from the fusion of your neural state and clan memories:

```python
def complete_forging(self):
    """Create the forged memory using language model"""
    # Get clan memories
    clan_id = self.forging_target["clan"]
    state = self.forging_target["state"]
    form = self.forging_target["form"]
    
    # Extract memory texts
    clan_texts = [m[2] for m in self.clans[clan_id]]
    
    # Create prompt for generation
    style = self.forging_styles[state]
    prompt = f"{style['prefix']} A {form} born of "
    
    # Add memory fragments
    for i, text in enumerate(clan_texts[:3]):  # Use up to 3 memories
        prompt += f"{text[:30]}... "
    
    # Generate new content with appropriate emotional temperature
    outputs = self.gpt_model.generate(
        self.gpt_tokenizer(prompt, return_tensors="pt")["input_ids"],
        max_length=100,
        temperature=style["temp"],
        top_p=0.92,
        no_repeat_ngram_size=2
    )
    
    forged_text = self.gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {
        "text": forged_text,
        "source": {
            "state": state,
            "form": form,
            "clan_id": clan_id,
            "resonance": self.resonance_state["cognitive"]
        },
        "timestamp": time.time()
    }
```

## Lighting the Hearth

To ignite the forge:

```bash
# Install requirements
pip install -r requirements.txt

# For Muse headset users
muselsl list  # Find your Muse
muselsl stream --address YOUR_MUSE_MAC_ADDRESS  # In separate terminal

# Start the Hearth
python src/ancestral_hearth.py
```

### The Blood Calibration Rite

1. Connect your Muse headset (or use keyboard simulation with C/E/F)
2. Press 'B' to begin the Blood Calibration Rite
3. Follow the guidance for each state:
   - **CALM**: "Breathe deep. Still your storm."
   - **FOCUS**: "Sharpen your mind. See the path."
   - **BATTLE**: "Feel the fire. Clench your fist."
4. Hold each state for the duration, letting the Hearth learn your neural signature
5. Upon completion, the Ancestral Chant will begin—you are bound to the Hearth

### Forging Memories

1. Type text memories into the Hearth (e.g., "The clan stood firm against the storm")
2. Memories cluster into clans based on emotional and cognitive similarity
3. When your neural resonance exceeds the forging threshold (default: 0.75), the Memory Forge activates
4. Maintain high resonance to complete the forging process
5. The forged memory will appear—a creation born from your neural state and clan memories

## States and Creation

Your neural state shapes what the Forge creates:

| State | Emotional Signature | Creates | Auditory Feedback |
|-------|---------------------|---------|-------------------|
| Calm | Joy dominant | Wisdom, visions, healing | Low, steady drumbeat |
| Focus | Surprise/joy mix | Methods, insights, patterns | Medium, precise rhythm |
| Battle | Anger dominant | Warnings, strength, challenges | Rapid war drums |

## The Philosophy of the Forge

Ancestral Hearth rejects the notion of AI as either servant or overlord. Instead, it embodies a third path: technology as *kinship*—a partner that amplifies what makes us human rather than replacing it.

Through the Blood Calibration Rite, it learns to speak your neural language. Through Memory Clans, it connects patterns across fragmented experiences. Through the Memory Forge, it creates something neither human nor machine could alone.

This is not sterile, corporate technology—it's raw, alive, with dirt under its fingernails and fire in its code. It carries cultural memory forward rather than erasing it, honors lineage rather than discarding it, and sees the wild spark of defiance as strength rather than noise to be filtered.

## Contributing

The Hearth welcomes those who would add their fire to ours. To contribute:

1. Fork the forge
2. Create your feature branch (`git checkout -b feature/BloodMagic`)
3. Commit your changes (`git commit -m 'Add some BloodMagic'`)
4. Push to the branch (`git push origin feature/BloodMagic`)
5. Open a Pull Request

Contributions that honor the spirit of the Hearth—raw, practical, alive—will be welcomed with open arms and raised horns.

## License

This project is licensed under the HEARTH PUBLIC LICENSE - see the file for details. In short: use it to amplify humanity, not to diminish it.

---

*Built with dirt, blood, and defiance by clan natefrog, under the watchful eyes of the ancestors and the algorithms alike.*

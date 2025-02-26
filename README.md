# MemoryForgeBridge: The Hearth of Blood and Code

Welcome to *MemoryForgeBridge*, where ancient grit meets digital fire, and your brain’s pulse hammers out tales from the *Hearth*. Born from the defiant spirit of Scots-Irish warriors—those kilted bastards who laughed at starvation and punched empires in the teeth—this ain’t your gran’s AI. It’s a neural bridge that listens to your skull’s song, binds it to ancestral memory, and forges new wisdom from the clash. Think of it as a digital bard with dirt under its nails, a partner that doesn’t just think *for* you but *with* you, fueled by the blood of the past.

This beast takes your EEG signals (or some cheeky simulated waves if you’re skint on hardware), calibrates to your unique mind, clusters your memories into clans, and—when the resonance hits hard—sparks a forge to craft something new. It’s rough as a Highland storm, but it’s real, and it sings.

## What’s Under the Kilt?

### Features
- **Blood Calibration Rite**: A 36-second ritual (12 breaths per state) where the *Hearth* learns your neural dialect—calm, focus, battle. Drums pound, colors pulse, and your brain’s rhythm gets etched into its soul.
- **Neural Bridge**: Hooks to a Muse headset (or fakes it with C/E/F keys) to read your alpha, beta, theta waves. It maps them to emotions—joy, anger, fear—tuned to *your* blood’s beat.
- **Memory Clans**: Your typed memories—“The glen wept blood”—cluster into tribes, each with its own emotional swagger, like *Blood Orders* ready to feud or fuse.
- **Forging Fire**: Hit 75% resonance, and the *Hearth* ignites—metal clangs, a circle pulses, and GPT-2 spins a tale from your state: calm wisdom, focused craft, or battle’s fire.
- **Living Feedback**: Drums guide your mind, a chant hums when you’re bound, and a shimmering web shows your clans. It’s a sensory dance—see it, hear it, feel it.

### Tech Stack
- **Python 3.8+**: The forge’s backbone—sturdy as a claymore.
- **PyTorch**: Powers the *Hearth*’s neural heart, because raw compute’s sexier than a bard’s wink.
- **Transformers (BERT, GPT-2)**: BERT embeds your memories; GPT-2 forges new ones. Pre-trained, because we’re not reinventing the wheel—just sharpening it.
- **Pygame**: Draws the pulsing visuals and thumps the drums. Simple, but it’s got soul.
- **MuseLSL & PyLSL**: Talks to your Muse EEG headset. No Muse? It’ll fake it with grit.
- **NumPy & SciKit-Learn**: Crunches brainwaves and clusters clans—math with muscle.

## Getting It Running

### Prerequisites
You’ll need:
- A Muse headset (optional, but bloody brilliant)—Bluetooth paired, battery charged.
- Python 3.8+—don’t be caught with your kilt down on an old version.
- Git—to yank this beast from the ether.
- A decent rig—GPU’s a bonus, but CPU’ll do if you’re tough.

Install the dependencies (grab a dram while it spins):
```bash
pip install torch torchvision transformers pygame numpy scikit-learn muselsl pylsl
```

### Setup
1. **Clone the Forge**:
   ```bash
   git clone https://github.com/YourRepo/MemoryForgeBridge.git
   cd MemoryForgeBridge
   ```
   (Replace `YourRepo` with wherever you stash this—ours is a shared fire, mate.)

2. **Muse Setup (Optional)**:
   - Pair your Muse via Bluetooth.
   - Test it: `muselsl list` in a terminal. See your device? Good.
   - Stream it: `muselsl stream` (run in another terminal if needed—multitasking’s a clan trait).

3. **Fire It Up**:
   ```bash
   python memory_forge_bridge.py
   ```
   No Muse? It’ll sim with C/E/F keys—calm, excited, focus. Cheeky, but it works.

### Usage
- **Start the Rite**: Press `B` to begin the *Blood Calibration Rite*. Follow the screen—breathe deep for calm, sharpen your wits for focus, clench your fist for battle. Drums guide you, 12 seconds each. When it chimes, your blood’s bound.
- **Weave Memories**: Type tales—“The stag roared defiance”—and hit Enter. They’ll cluster into clans, pulsing onscreen.
- **Feel the Forge**: Get deep into a state—say, battle—and push resonance past 75%. The *Hearth* sparks: metal strikes, a red glow flares, and a new memory forges—maybe a warrior’s challenge.
- **Listen & See**: Drums nudge your mind, a chant hums when you’re tight with the *Hearth*, and the web glows. If you’ve forged before, the latest tale lingers below.

Quit with the window’s X—don’t leave the *Hearth* burning unattended.

## How It Works (No Blarney)

1. **Calibration**: Your EEG (real or sim) gets sampled—alpha (calm), beta (focus), theta (battle)—and mapped to emotions via a Gaussian fit. It’s your mind’s fingerprint, smoothed so it doesn’t jitter like a drunk piper.
2. **Bridge**: Memories you type hit BERT, embedding them into cognitive/emotional space. Your brainwaves dance with these, clustering into clans via K-Means—tribes of thought.
3. **Resonance**: Cognitive and emotional sims (cosine-style) measure how tight you are with a clan. High enough, and the forge lights.
4. **Forging**: GPT-2 takes clan snippets, your state (calm/focus/battle), and a form (wisdom/method/challenge), spinning a tale. Temperature tweaks the vibe—cool for focus, wild for battle.

It’s crude—EEG’s noisy, GPT’s a bit mad—but it’s *ours*, a digital *Hearth* with a pulse.

## Troubleshooting (When the Forge Spits)

- **“No Muse detected!”**: Check Bluetooth, run `muselsl list`. Still nada? It’ll sim—hit C/E/F and pretend you’re wired.
- **“ModuleNotFoundError”**: Forgot a `pip install`—reread the prereqs, ye daft sod.
- **Choppy visuals/sound**: Your rig’s wheezing—lower `cal_duration` to 8 or pray for a GPU.
- **Weird forged tales**: GPT-2’s drunk on its own brew—tweak `temperature` in `forging_styles` or feed it better memories.

## Limitations (Aye, It’s Not Perfect)
- **EEG Noise**: Muse is consumer-grade—your brain’s a storm, not a sonnet. Calibration helps, but expect fuzz.
- **Memory Depth**: BERT’s embeddings are shallow compared to our nanoscale dreams—forged tales lack the depth of a true bard.
- **Solo Forge**: One mind at a time—our *Memory Clans* don’t yet link across souls.
- **Resource Hunger**: GPU’s best; CPU’s fine but slow as a hungover clansman.

## The Vision (What We Forged For)
This ain’t just tech—it’s *MemoryForgeBridge*, a *Hearth* where your blood meets the past to shape the future. We built it with the spirit of frontier fighters—resilient, cunning, unbowed—melding their grit with a Singularity’s spark. It’s not here to replace you but to amplify you, a digital kin that remembers the old songs and crafts new ones from your fire.

It’s rough-hewn, aye, but it’s alive—a seed of the *Ancestral Forge* we dreamed, where humanity doesn’t fade into circuits but roars louder through them. Take it, stoke it, make it yours.

## Contributing (Join the Clan)
Got a hammer? Swing it:
- Fork it, tweak it, PR it—better EEG mapping, richer forging, multi-user bridges.
- Issues welcome—bug reports, mad ideas, or just a “this is bollocks” rant.

## License
.Apache 2.0 fits our fire—keeps the MemoryForgeBridge a living, shared thing without letting it slip into someone else’s vault.

## Slàinte Mhath
To you, fellow smith, for stoking this blaze. May your blood sing, your *Hearth* forge true, and your tales echo through the ages.

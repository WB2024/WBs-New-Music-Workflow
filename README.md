# [![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support-yellow?logo=buy-me-a-coffee)](https://buymeacoffee.com/succinctrecords)

# 🎵 WB's Music Workflow

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Essentia](https://img.shields.io/badge/Essentia-2.1b6-green.svg)](https://essentia.upf.edu/)

**An end-to-end music library pipeline — from download to perfectly tagged, sorted library.**

Combines ML-powered audio analysis, persistent artist genre definitions, and automatic file sorting into a single interactive workflow. Tag your music in MusicBrainz Picard, run this script, and everything lands in the right place with the right tags.

---

## ✨ What It Does

```
New music download
       │
       ▼
 MusicBrainz Picard ──► Tagged Directory
 (quality metadata)      (Picard naming applied)
                                │
                        workflow.py runs
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                  ▼
      Essentia ML          Sort artists        Genre enforcement
   genre/mood tags     new vs. existing       (artist definitions)
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                    ▼
     New Artist Staging                    Clean Music Library
   (define genre, then                  (merged into correct
      move to library)                   artist folder)
```

### The three tools, working together

| Script | Role | Technology |
|--------|------|------------|
| `tag_music.py` | ML-based genre + mood tagging | Essentia / TensorFlow |
| `music_genre_enforcer.py` | Enforces your personal genre per artist | Mutagen |
| `workflow.py` | Orchestrates everything | Python stdlib |

---

## 🗂️ Library Structure

The workflow is built around the same folder structure that MusicBrainz Picard outputs:

```
/your/music/library/
├── A/
│   └── Aphex Twin/
│       └── [1992] Selected Ambient Works 85-92/
│           ├── 01 - Aphex Twin. Xtal.flac
│           └── ...
├── B/
│   └── Bowie, David/
│       └── [1972] The Rise and Fall of Ziggy Stardust/
│           └── ...
├── #/
│   └── 2Pac/
│       └── [1996] All Eyez on Me/
│           └── ...
└── Various Artists/
    └── [2020] Compilation Title/
        └── ...
```

- **Top level** — grouping folders by first letter (`A`, `B`, `#`, `Various Artists`, etc.)
- **Second level** — artist/sort name folders
- **Third level+** — `[Year] Album Title` folders containing tracks

This same structure applies to the **Tagged directory** (Picard output) and the **Clean library** (final destination).

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Linux** (Debian/Ubuntu recommended)
- **MusicBrainz Picard** — configured to save files to your Tagged directory
- **~100MB disk space** for Essentia models
- **8GB+ RAM** recommended for Essentia analysis

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/WB2024/WBs-New-Music-Workflow.git
cd WBs-New-Music-Workflow
```

---

### Step 2 — Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Debian/Ubuntu alternative** (no venv needed):
> ```bash
> sudo apt install -y python3-mutagen
> pip install essentia-tensorflow numpy
> ```

---

### Step 4 — Download Essentia ML models (~87 MB)

```bash
bash download_models.sh
```

Models are saved to `~/essentia_models/`. This only needs to be done once.

---

### Step 5 — Configure MusicBrainz Picard

In Picard → **Options → File Naming**:

1. Enable **"Rename files when saving"**
2. Set the destination folder to your **Tagged directory** (e.g. `/srv/music/Tagged`)
3. Use the file naming script from the Picard documentation to match the `INITIAL/ArtistSort/[Year] Album/` structure

---

### Step 6 — Run the workflow

```bash
source venv/bin/activate   # if using venv
python workflow.py
```

On first run, a setup wizard will ask for:

- **Tagged directory** — where Picard saves files
- **Clean library** — your main organised music library
- **New artist staging directory** — holding area for new artists before genre definition
- **Essentia settings** — genre/mood thresholds, format, workers, etc.

Settings are saved to `~/.config/music-workflow/settings.json` and reused on every future run.

---

## 📋 Menu Reference

```
══════════════════════════════════════════════════════════════════════════════
 Music Workflow
══════════════════════════════════════════════════════════════════════════════
  Tagged directory  : /srv/music/Tagged  (12 files waiting)
  Clean library     : /srv/music/Clean
  New artist dir    : /srv/music/NewArtists  (2 artist(s) pending)
  Dry run           : No
  Definitions       : 347 artist(s) defined
══════════════════════════════════════════════════════════════════════════════

  1) Run workflow  (Essentia → sort → enforce → move)
  2) Manage new artists  (define genres + move to library)
  3) Settings
  4) Edit artist definitions
  5) Toggle dry run
  q) Quit
```

### 1) Run Workflow

The full pipeline in one step:

| Stage | Description |
|-------|-------------|
| **Essentia Analysis** | Runs ML models on all files in Tagged dir, writing `GENRE` and `MOOD` tags |
| **Sort Artists** | Compares artist folders in Tagged against Clean library |
| **Stage New Artists** | Moves new artist folders to the staging directory |
| **Genre Enforcement** | Applies your defined genre to all files for existing artists |
| **Move to Library** | Merges existing artist files into the correct Clean library folder |

> **Tip:** Run with dry run enabled first to preview exactly what will happen.

---

### 2) Manage New Artists

Shows all artists currently in the staging directory. For each one:

- Displays file count and current genre status
- Prompts for genre definitions (with `l` / `p` / `s` / `?` commands — same as Genre Enforcer)
- Once defined, applies the genre tag and moves to Clean library

```
Found 2 artist(s) in staging directory:

  1. Portishead (23 files) [no genre defined]
  2. Massive Attack (47 files) [no genre defined]

Define genres for 2 undefined artist(s)? [Y/n]: y

Genre for [Massive Attack] (or l/p/?/s): p
Existing genres:
  [1] Electronic - Trip Hop
  [2] Hip-Hop
  ...
Pick number (blank cancels): 1
  -> Set [Massive Attack] = Electronic - Trip Hop
```

---

### 3) Settings

Numbered editor for every setting — no need to re-run the wizard:

```
  Directories:
    1) Tagged directory      : /srv/music/Tagged
    2) Clean library         : /srv/music/Clean
    3) New artist directory  : /srv/music/NewArtists

  Essentia:
    4) Analysis mode         : Genres & Moods
    5) Number of genres      : 3
    6) Genre threshold       : 15%
    7) Genre format          : parent_child
    8) Mood threshold        : 0.5%
    9) Confidence tags       : Yes
   10) Overwrite existing    : No
   11) Workers               : auto
   12) Max audio duration    : 300s
   13) Model directory       : ~/essentia_models

  General:
   14) Dry run               : No
```

Press `s` to save, `b` to discard changes.

---

### 4) Edit Artist Definitions

Opens the numbered picker from the Genre Enforcer to edit, add, or delete genre definitions:

```
[   0] 2Pac                    ->  Hip-Hop
[   1] Aphex Twin              ->  Electronic - Ambient
[   2] Bowie, David            ->  Rock - Art
...

Commands: [number]=edit | d [number]=delete | q=back
```

---

### 5) Toggle Dry Run

Instantly flip dry run on/off. In dry run mode no files are modified, moved, or tagged — everything is previewed only.

---

## ⚙️ Essentia Settings Reference

### Analysis Mode

| Option | Description |
|--------|-------------|
| `Genres & Moods` | Runs both ML models — writes `GENRE` and `MOOD` tags |
| `Genres only` | Only the genre model is loaded and run |
| `Moods only` | Only the mood model is loaded and run |

### Genre Format

Raw model output uses Discogs taxonomy format: `Rock---Alternative Rock`

| Style | Output | Description |
|-------|--------|-------------|
| `parent_child` *(default)* | `Rock - Alternative Rock` | Full context, clean separator |
| `child_parent` | `Alternative Rock - Rock` | Subgenre first |
| `child_only` | `Alternative Rock` | Subgenre only |
| `raw` | `Rock---Alternative Rock` | No formatting |

### Thresholds

**Genres** — model predicts across 400 Discogs classes. A 15% threshold is a good starting point:

| Threshold | Behaviour |
|-----------|-----------|
| 5% | Very inclusive — more genres, including speculative ones |
| 15% | Balanced *(recommended)* |
| 25% | Strict — only high-confidence tags |

**Moods** — naturally much lower confidence than genres (typically 0.1–5%):

| Threshold | Behaviour |
|-----------|-----------|
| 0.5% | Inclusive *(good starting point)* |
| 1–2% | Moderate |
| 3%+ | Strict — few or no moods |

### Tag Formats by Audio Container

| Format | Genre tag | Mood tag | Confidence tag |
|--------|-----------|----------|----------------|
| FLAC, OGG, Opus | `GENRE` | `MOOD` | `ESSENTIA_GENRE` / `ESSENTIA_MOOD` |
| MP3, AIFF, WAV, DSF | `TCON` | `COMM:Essentia Mood` | `COMM:Essentia Genre` |
| M4A, MP4, AAC | `©gen` | `----:com.apple.iTunes:MOOD` | `©cmt` |
| WMA | `WM/Genre` | `WM/Mood` | `ESSENTIA_GENRE` |
| WavPack, APE, Musepack | `Genre` | `Mood` | `Essentia Genre` |

---

## 🗄️ Configuration & Data Files

All persistent data lives in `~/.config/music-workflow/`:

```
~/.config/music-workflow/
├── settings.json          # All workflow settings (directories, Essentia params)
└── artist_genres.json     # Your artist → genre mappings
```

The Essentia tagger also writes a timestamped log file per run to `~/.config/music-workflow/logs/`:

```
~/.config/music-workflow/logs/
└── essentia_tagger_20260405_143022.log
```

**Portability:** Copy `artist_genres.json` to another machine and all your definitions come with you.

---

## 🔄 Updating

### Update the scripts

```bash
cd WBs-New-Music-Workflow
git pull
```

Your settings and artist definitions in `~/.config/music-workflow/` are never touched by an update — only the scripts themselves change.

### Update Essentia models

Re-run the model downloader to get the latest versions:

```bash
bash download_models.sh
```

---

## 🛠️ Troubleshooting

### "No module named 'numpy'" or "No module named 'essentia'"

The Essentia dependencies aren't installed, or the virtual environment isn't activated:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "Could not load model" / models not found

```bash
bash download_models.sh
```

Check that `~/essentia_models/` contains the `.pb` and `.json` files.

### "Tagged directory does not exist"

Go to **Settings → 1** and update the Tagged directory path.

### "Library base does not exist" in genre enforcement

Go to **Settings → 2** and verify the Clean library path.

### No moods above threshold

Moods have very low confidence (0.1–5% is normal). Lower the mood threshold in Settings:

```
Settings → 8) Mood threshold  [0.5]: 0.1
```

### TensorFlow warnings about libcudart

```
Could not load dynamic library 'libcudart.so.11.0'
```

Safe to ignore — this just means no GPU was found and CPU will be used instead.

### Files "skipped" during genre enforcement

Normal behaviour. Skipped = the file already has the correct genre tag and does not need rewriting.

---

## ⚡ Performance

Essentia analysis speed depends entirely on hardware (CPU-only):

| Hardware | Per track | 500 tracks | 2000 tracks |
|----------|-----------|------------|-------------|
| Intel Core i3 (older) | ~10–15s | 2–4 hrs | 8–16 hrs |
| Modern laptop (i5/i7) | ~5–8s | ~1 hr | ~3–4 hrs |

**Tips:**
- Increase **Workers** in Settings to use more CPU cores in parallel
- Reduce **Max audio duration** (e.g. 120s) — genre/mood models don't need the full track
- Run overnight for large libraries

Memory usage: ~2–3 GB RAM while Essentia is running (models + audio buffers).

---

## 🧩 Architecture

The workflow is built from three Python modules in the same directory:

```
workflow.py                  ← Entry point / orchestrator
├── imports tag_music        ← Essentia analysis (lazy — only loaded when needed)
└── imports music_genre_enforcer  ← Genre definitions and enforcement
```

**`workflow.py`** — Orchestrates the pipeline. Handles all user interaction, settings, path management, file moving, and calls into the two specialist modules.

**`tag_music.py`** — ML audio analysis. Uses Essentia to extract embeddings from the raw audio waveform and predict genres and moods. Exposes `run_tagging()` for programmatic use.

**`music_genre_enforcer.py`** — Artist-defined genre enforcement. Reads your `artist_genres.json` and writes the configured genre to every file under each artist folder. Reusable as a standalone script or imported module.

---

## 📜 License

MIT — see [LICENSE](LICENSE) for details.

> Essentia itself is AGPL-3.0. The pre-trained MTG models are CC BY-NC-ND 4.0 (non-commercial use).

---

## 🙏 Credits

- [Essentia](https://essentia.upf.edu/) — Audio analysis library by Music Technology Group, Universitat Pompeu Fabra
- [Mutagen](https://mutagen.readthedocs.io/) — Python audio metadata library
- [MusicBrainz Picard](https://picard.musicbrainz.org/) — The best open-source music tagger
- Pre-trained models from [MTG](https://www.upf.edu/web/mtg), trained on Discogs and MTG-Jamendo datasets

---

*Made with ❤️ for people who take their music library seriously.*  
*If this saves you hours of manual work, consider [buying me a coffee](https://buymeacoffee.com/succinctrecords) ☕*

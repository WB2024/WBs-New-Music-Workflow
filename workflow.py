#!/usr/bin/env python3
"""
Music Workflow - Orchestrates the music tagging pipeline.

Pipeline:
  1. New music → tagged in MusicBrainz Picard → saved to Tagged directory
  2. Run this script:
     a. Essentia analyses and writes genre/mood tags to all files
     b. Artist folders compared against Clean library
     c. New artists moved to staging directory
     d. Existing artists get genre enforcement, then moved to Clean library
  3. Manage new artists: define genres, apply, move to library

Config: ~/.config/music-workflow/
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure sibling modules can be imported regardless of cwd
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import music_genre_enforcer as enforcer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_DIR_NAME = "music-workflow"
SETTINGS_FILE_NAME = "settings.json"
DEFINITIONS_FILE_NAME = "artist_genres.json"
VARIOUS_ARTISTS = "Various Artists"
BANNER = "=" * 78
LOCKFILE_NAME = ".workflow.lock"

DEFAULT_SETTINGS: dict = {
    "tagged_dir": "",
    "clean_library_dir": "",
    "new_artist_dir": "",
    "essentia": {
        "enable_genres": True,
        "enable_moods": True,
        "top_n_genres": 3,
        "genre_threshold": 15.0,      # percentage (converted to fraction for Essentia)
        "mood_threshold": 0.5,        # percentage
        "genre_format": "parent_child",
        "write_confidence_tags": True,
        "overwrite_existing": False,
        "workers": 0,                 # 0 = auto (half CPU cores)
        "max_audio_duration": 300,
        "model_dir": "~/essentia_models",
    },
    "dry_run": False,
}


# ---------------------------------------------------------------------------
# Config directory
# ---------------------------------------------------------------------------

def config_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / APP_DIR_NAME
    return Path.home() / ".config" / APP_DIR_NAME


# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------

def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError):
        return default


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_settings(cfg_dir: Path) -> dict:
    raw = _read_json(cfg_dir / SETTINGS_FILE_NAME, None)
    if not isinstance(raw, dict):
        return _deep_copy_defaults()
    merged = _deep_copy_defaults()
    merged.update({k: v for k, v in raw.items() if k != "essentia"})
    if isinstance(raw.get("essentia"), dict):
        ess = dict(DEFAULT_SETTINGS["essentia"])
        ess.update(raw["essentia"])
        merged["essentia"] = ess
    return merged


def _deep_copy_defaults() -> dict:
    d = dict(DEFAULT_SETTINGS)
    d["essentia"] = dict(DEFAULT_SETTINGS["essentia"])
    return d


def save_settings(cfg_dir: Path, settings: dict) -> None:
    _write_json(cfg_dir / SETTINGS_FILE_NAME, settings)


@contextmanager
def workflow_lock(cfg_dir: Path, blocking: bool = True):
    """Acquire an exclusive lock to prevent concurrent workflow runs.

    Args:
        cfg_dir: Config directory where the lock file lives.
        blocking: If True, block until the lock is available.
                  If False, raise RuntimeError immediately if locked.
    """
    lock_path = cfg_dir / LOCKFILE_NAME
    cfg_dir.mkdir(parents=True, exist_ok=True)
    lock_fd = open(lock_path, "w")
    locked = False
    try:
        flags = fcntl.LOCK_EX if blocking else (fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            fcntl.flock(lock_fd, flags)
            locked = True
        except OSError:
            raise RuntimeError("Another workflow instance is already running.")
        lock_fd.write(str(os.getpid()))
        lock_fd.flush()
        yield
    finally:
        if locked:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


def load_definitions(cfg_dir: Path) -> Dict[str, str]:
    return enforcer.load_definitions(cfg_dir)


def save_definitions(cfg_dir: Path, defs: Dict[str, str]) -> None:
    enforcer.save_definitions(cfg_dir, defs)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def clear() -> None:
    os.system("clear" if os.name != "nt" else "cls")


def pause(msg: str = "Press Enter to continue...") -> None:
    try:
        input(msg)
    except (EOFError, KeyboardInterrupt):
        print()


def prompt(msg: str, default: str = "") -> str:
    if default:
        s = input(f"{msg} [{default}]: ").strip()
        return s if s else default
    return input(f"{msg}: ").strip()


def prompt_yes_no(msg: str, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    s = input(f"{msg} [{suffix}]: ").strip().lower()
    if not s:
        return default
    return s in ("y", "yes", "1", "true")


def prompt_path(msg: str, default: str = "") -> str:
    """Prompt for a filesystem path with ~ expansion."""
    while True:
        p = prompt(msg, default).strip().strip("'\"")
        p = os.path.expanduser(p)
        if not p:
            print("  Path cannot be empty.")
            continue
        return p


def prompt_int(msg: str, default: int, min_val: int = None, max_val: int = None) -> int:
    while True:
        raw = input(f"{msg} [{default}]: ").strip()
        if not raw:
            return default
        try:
            v = int(raw)
            if min_val is not None and v < min_val:
                print(f"  Must be at least {min_val}")
                continue
            if max_val is not None and v > max_val:
                print(f"  Must be at most {max_val}")
                continue
            return v
        except ValueError:
            print("  Enter a valid number.")


def prompt_float(msg: str, default: float, min_val: float = None, max_val: float = None) -> float:
    while True:
        raw = input(f"{msg} [{default}]: ").strip()
        if not raw:
            return default
        try:
            v = float(raw)
            if min_val is not None and v < min_val:
                print(f"  Must be at least {min_val}")
                continue
            if max_val is not None and v > max_val:
                print(f"  Must be at most {max_val}")
                continue
            return v
        except ValueError:
            print("  Enter a valid number.")


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------

def discover_artists(base_dir: str) -> List[Tuple[str, str, Path]]:
    """Discover artist directories from a Picard-structured directory.

    Expected layout (from Picard naming script):
      base/INITIAL/ArtistSort/[Year] Album/tracks
      base/Various Artists/[Year] Album/tracks

    Returns a list of (relative_dir, artist_name, full_path) tuples where:
      - relative_dir: path relative to base (e.g. "A/Aphex Twin")
      - artist_name:  folder name used for genre lookups
      - full_path:    absolute Path to the artist directory
    """
    base = Path(base_dir)
    artists: List[Tuple[str, str, Path]] = []

    if not base.exists():
        return artists

    try:
        top_entries = sorted(os.scandir(base), key=lambda e: e.name.casefold())
    except PermissionError:
        return artists

    for group_entry in top_entries:
        if not group_entry.is_dir():
            continue

        if group_entry.name == VARIOUS_ARTISTS:
            # VA is treated as a single "artist" whose dir contains albums
            artists.append((VARIOUS_ARTISTS, VARIOUS_ARTISTS, Path(group_entry.path)))
            continue

        # Regular group folder (letter / # / etc.) — children are artist dirs
        try:
            artist_entries = sorted(
                os.scandir(group_entry.path), key=lambda e: e.name.casefold()
            )
        except PermissionError:
            continue

        for artist_entry in artist_entries:
            if artist_entry.is_dir():
                rel = f"{group_entry.name}/{artist_entry.name}"
                artists.append((rel, artist_entry.name, Path(artist_entry.path)))

    return artists


def artist_exists_in_library(clean_dir: str, relative_dir: str) -> bool:
    """Check whether an artist directory exists in the clean library."""
    return (Path(clean_dir) / relative_dir).is_dir()


def count_audio_files(path: Path) -> int:
    count = 0
    for _, _, files in os.walk(path):
        for f in files:
            if os.path.splitext(f)[1].lower() in enforcer.SUPPORTED_EXTS_DEFAULT:
                count += 1
    return count


# ---------------------------------------------------------------------------
# File / directory operations
# ---------------------------------------------------------------------------

def merge_tree(src: str, dst: str) -> int:
    """Move *src* tree into *dst*, merging directories.  Returns files moved."""
    moved = 0
    dst_path = Path(dst)

    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            if os.path.isdir(d):
                moved += merge_tree(s, d)
            else:
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.move(s, d)
                for _, _, files in os.walk(d):
                    moved += len(files)
        else:
            os.makedirs(str(dst_path), exist_ok=True)
            shutil.move(s, d)
            moved += 1

    # Remove now-empty source directory
    try:
        os.rmdir(src)
    except OSError:
        pass

    return moved


def cleanup_empty_dirs(root: str) -> None:
    """Remove empty directories bottom-up, leaving *root* itself intact."""
    for dirpath, _dirnames, _filenames in os.walk(root, topdown=False):
        if dirpath == root:
            continue
        if not os.listdir(dirpath):
            try:
                os.rmdir(dirpath)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Essentia integration (lazy import to avoid requiring essentia at startup)
# ---------------------------------------------------------------------------

def run_essentia(tagged_dir: str, settings: dict) -> Optional[str]:
    """Run Essentia tagger on *tagged_dir*.  Returns log file path or None."""
    try:
        import tag_music
    except ImportError as e:
        print(f"  Error importing tag_music: {e}")
        print("  Make sure essentia-tensorflow, mutagen, and numpy are installed.")
        return None

    ess = settings.get("essentia", {})

    overrides = {
        "dry_run": settings.get("dry_run", False),
        "enable_genres": ess.get("enable_genres", True),
        "enable_moods": ess.get("enable_moods", True),
        "top_n_genres": ess.get("top_n_genres", 3),
        "genre_threshold": ess.get("genre_threshold", 15.0) / 100.0,
        "mood_threshold": ess.get("mood_threshold", 0.5) / 100.0,
        "genre_format": ess.get("genre_format", "parent_child"),
        "write_confidence_tags": ess.get("write_confidence_tags", True),
        "overwrite_existing": ess.get("overwrite_existing", False),
        "verbose": True,
        "workers": ess.get("workers", 0) or max(1, (os.cpu_count() or 2) // 2),
        "max_audio_duration": ess.get("max_audio_duration", 300),
        "model_dir": ess.get("model_dir", "~/essentia_models"),
    }

    log_dir = str(config_dir() / "logs")
    return tag_music.run_tagging(tagged_dir, overrides, log_dir)


# ---------------------------------------------------------------------------
# Genre enforcement helper
# ---------------------------------------------------------------------------

def _make_enforcer_cfg(settings: dict) -> enforcer.AppConfig:
    """Build an AppConfig suitable for apply_genre_to_artist_dir."""
    return enforcer.AppConfig(
        library_base=Path("/unused"),
        supported_exts=enforcer.SUPPORTED_EXTS_DEFAULT,
        dry_run=settings.get("dry_run", False),
    )


# ---------------------------------------------------------------------------
# Workflow execution
# ---------------------------------------------------------------------------

def run_workflow(settings: dict, definitions: Dict[str, str], cfg_dir: Path, *, auto: bool = False) -> None:
    """Execute the full workflow pipeline."""
    tagged_dir = settings["tagged_dir"]
    clean_dir = settings["clean_library_dir"]
    new_artist_dir = settings["new_artist_dir"]
    dry_run = settings["dry_run"]

    # --- Validate paths ---
    if not os.path.isdir(tagged_dir):
        print(f"Tagged directory does not exist: {tagged_dir}")
        return
    if not os.path.isdir(clean_dir):
        print(f"Clean library does not exist: {clean_dir}")
        return
    os.makedirs(new_artist_dir, exist_ok=True)

    # --- Step 0: Discover contents of tagged directory ---
    print("\nScanning tagged directory...")
    artists = discover_artists(tagged_dir)

    if not artists:
        print("No music found in tagged directory. Nothing to process.")
        return

    total_files = sum(count_audio_files(p) for _, _, p in artists)
    print(f"Found {len(artists)} artist(s), ~{total_files} audio files\n")

    # --- Step 1: Essentia analysis ---
    print(f"{'─' * 70}")
    print("Step 1: Essentia Analysis")
    print(f"{'─' * 70}")

    if auto or prompt_yes_no("Run Essentia analysis on tagged files?", default=True):
        log_file = run_essentia(tagged_dir, settings)
        if log_file:
            print(f"\nEssentia log: {log_file}")
    else:
        print("Skipping Essentia analysis.")

    # --- Step 2: Sort artists into new vs existing ---
    print(f"\n{'─' * 70}")
    print("Step 2: Sorting Artists")
    print(f"{'─' * 70}")

    # Re-scan (Essentia may have been skipped, but structure is unchanged)
    artists = discover_artists(tagged_dir)
    existing: List[Tuple[str, str, Path]] = []
    new: List[Tuple[str, str, Path]] = []

    for rel_dir, name, path in artists:
        if artist_exists_in_library(clean_dir, rel_dir):
            existing.append((rel_dir, name, path))
        else:
            new.append((rel_dir, name, path))

    print(f"  Existing in library : {len(existing)}")
    print(f"  New artists         : {len(new)}")

    # --- Step 3: Move new artists to staging ---
    if new:
        print(f"\nMoving {len(new)} new artist(s) to staging directory...")
        for rel_dir, name, path in new:
            dest = Path(new_artist_dir) / rel_dir
            if dry_run:
                print(f"  [DRY RUN] {name} → {dest}")
            else:
                moved = merge_tree(str(path), str(dest))
                print(f"  {name} → {dest} ({moved} files)")

    # --- Step 4: Genre enforcement for existing artists ---
    if existing:
        print(f"\n{'─' * 70}")
        print("Step 3: Genre Enforcement")
        print(f"{'─' * 70}")

        enforcer_cfg = _make_enforcer_cfg(settings)
        enforced = 0
        skipped = 0

        for rel_dir, name, path in existing:
            if not path.exists():
                continue
            genre = definitions.get(name, "").strip()
            if genre:
                counts = enforcer.apply_genre_to_artist_dir(path, genre, enforcer_cfg)
                print(f"  {name}: genre='{genre}' "
                      f"(files={counts['total']} changed={counts['changed']})")
                enforced += 1
            else:
                print(f"  {name}: no genre defined — skipping enforcement")
                skipped += 1

        print(f"\n  Enforced: {enforced}  Skipped: {skipped}")

    # --- Step 5: Move existing artists to clean library ---
    if existing:
        print(f"\n{'─' * 70}")
        print("Step 4: Moving to Library")
        print(f"{'─' * 70}")

        for rel_dir, name, path in existing:
            if not path.exists():
                continue
            dest = Path(clean_dir) / rel_dir
            if dry_run:
                print(f"  [DRY RUN] {name} → {dest}")
            else:
                moved = merge_tree(str(path), str(dest))
                print(f"  {name} → {dest} ({moved} files)")

    # --- Step 6: Cleanup empty directories in tagged ---
    cleanup_empty_dirs(tagged_dir)

    # --- Done ---
    print(f"\n{'=' * 70}")
    if dry_run:
        print("Workflow complete (DRY RUN — no files were moved)")
    else:
        print("Workflow complete!")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# New artist management
# ---------------------------------------------------------------------------

def manage_new_artists(
    settings: dict, definitions: Dict[str, str], cfg_dir: Path
) -> Dict[str, str]:
    """Interactive management of new artists in the staging directory."""
    new_artist_dir = settings["new_artist_dir"]
    clean_dir = settings["clean_library_dir"]
    dry_run = settings["dry_run"]

    if not os.path.isdir(new_artist_dir):
        print("New artist directory does not exist.")
        pause()
        return definitions

    artists = discover_artists(new_artist_dir)

    if not artists:
        print("No new artists found in staging directory.")
        pause()
        return definitions

    # Display what we have
    print(f"\nFound {len(artists)} artist(s) in staging directory:\n")
    for i, (rel_dir, name, path) in enumerate(artists, 1):
        fc = count_audio_files(path)
        genre = definitions.get(name, "").strip()
        status = f"genre: {genre}" if genre else "no genre defined"
        print(f"  {i}. {name} ({fc} files) [{status}]")

    artist_names = [name for _, name, _ in artists]
    undefined = [n for n in artist_names if not definitions.get(n, "").strip()]

    # Prompt for undefined artists
    if undefined:
        print()
        if prompt_yes_no(
            f"Define genres for {len(undefined)} undefined artist(s)?", default=True
        ):
            definitions = enforcer.prompt_for_new_artists(definitions, artist_names)
            save_definitions(cfg_dir, definitions)
            print("\nDefinitions saved.")
    else:
        print("\nAll artists already have genre definitions.")

    # Determine which artists are ready to move (have a genre defined)
    ready = [(r, n, p) for r, n, p in artists if definitions.get(n, "").strip()]

    if not ready:
        print("\nNo artists with genre definitions to move.")
        pause()
        return definitions

    if not prompt_yes_no(
        f"\nApply genres and move {len(ready)} artist(s) to library?", default=True
    ):
        pause()
        return definitions

    enforcer_cfg = _make_enforcer_cfg(settings)

    print()
    for i, (rel_dir, name, path) in enumerate(ready, 1):
        genre = definitions[name].strip()

        # Apply genre tag
        counts = enforcer.apply_genre_to_artist_dir(path, genre, enforcer_cfg)

        # Move to clean library
        dest = Path(clean_dir) / rel_dir
        if dry_run:
            print(
                f"  [{i}/{len(ready)}] {name}: genre='{genre}', "
                f"{counts['total']} files [DRY RUN]"
            )
        else:
            moved = merge_tree(str(path), str(dest))
            print(
                f"  [{i}/{len(ready)}] {name}: genre='{genre}', "
                f"{moved} files → {dest}"
            )

    cleanup_empty_dirs(new_artist_dir)

    print("\nDone!")
    pause()
    return definitions


# ---------------------------------------------------------------------------
# Settings UI — first-run wizard
# ---------------------------------------------------------------------------

def _prompt_essentia_settings(ess: dict) -> dict:
    """Interactively configure Essentia settings."""
    print("\n  Analysis mode:")
    print("    1 = Genres & Moods (both)")
    print("    2 = Genres only")
    print("    3 = Moods only")
    mode = prompt_int("  Mode", 1, 1, 3)
    ess["enable_genres"] = mode in (1, 2)
    ess["enable_moods"] = mode in (1, 3)

    if ess["enable_genres"]:
        print()
        ess["top_n_genres"] = prompt_int(
            "  Number of genres per track (1-10)", ess["top_n_genres"], 1, 10
        )
        ess["genre_threshold"] = prompt_float(
            "  Genre confidence threshold %", ess["genre_threshold"], 1, 50
        )
        print("  Genre format:  1=parent_child  2=child_parent  3=child_only  4=raw")
        fmt_choice = prompt_int("  Format", 1, 1, 4)
        fmt_map = {1: "parent_child", 2: "child_parent", 3: "child_only", 4: "raw"}
        ess["genre_format"] = fmt_map[fmt_choice]

    if ess["enable_moods"]:
        print()
        ess["mood_threshold"] = prompt_float(
            "  Mood confidence threshold %", ess["mood_threshold"], 0.01, 20
        )

    print()
    ess["write_confidence_tags"] = prompt_yes_no(
        "  Write confidence score tags?", default=ess["write_confidence_tags"]
    )
    ess["overwrite_existing"] = prompt_yes_no(
        "  Overwrite existing tags?", default=ess["overwrite_existing"]
    )

    cpu_count = os.cpu_count() or 2
    ess["workers"] = prompt_int(
        f"  Parallel workers (0=auto, 1-{cpu_count})", ess["workers"], 0, cpu_count
    )
    ess["max_audio_duration"] = prompt_int(
        "  Max audio duration (seconds, 0=unlimited)",
        ess["max_audio_duration"], 0, 3600,
    )
    ess["model_dir"] = prompt_path(
        "  Essentia model directory", ess["model_dir"]
    )

    return ess


def first_run_wizard(cfg_dir: Path) -> dict:
    """Initial configuration wizard for first-time users."""
    clear()
    print(BANNER)
    print(" Music Workflow — First Time Setup")
    print(BANNER)
    print()

    settings = _deep_copy_defaults()

    # --- Directory paths ---
    print("Directory Configuration")
    print("─" * 70)
    settings["tagged_dir"] = prompt_path(
        "  Tagged directory (Picard output)"
    )
    settings["clean_library_dir"] = prompt_path(
        "  Clean music library"
    )
    settings["new_artist_dir"] = prompt_path(
        "  New artist staging directory"
    )

    # --- Essentia ---
    print(f"\nEssentia Settings")
    print("─" * 70)
    if prompt_yes_no("  Use default Essentia settings?", default=True):
        print(
            "  Defaults: genres+moods, 3 genres, 15% threshold, "
            "parent_child, 0.5% mood threshold"
        )
    else:
        settings["essentia"] = _prompt_essentia_settings(settings["essentia"])

    # --- General ---
    print(f"\nGeneral")
    print("─" * 70)
    settings["dry_run"] = prompt_yes_no(
        "  Enable dry run (test without writing)?", default=True
    )

    # --- Create directories if needed ---
    for key in ("tagged_dir", "clean_library_dir", "new_artist_dir"):
        p = settings[key]
        if not os.path.exists(p):
            if prompt_yes_no(f"  Create '{p}'?", default=True):
                os.makedirs(p, exist_ok=True)

    save_settings(cfg_dir, settings)
    print(f"\nSettings saved to {cfg_dir / SETTINGS_FILE_NAME}")
    pause()
    return settings


# ---------------------------------------------------------------------------
# Settings UI — edit existing settings
# ---------------------------------------------------------------------------

def edit_settings(cfg_dir: Path, settings: dict) -> dict:
    """Numbered setting editor."""
    while True:
        clear()
        ess = settings.get("essentia", {})

        # Analysis mode label
        if ess.get("enable_genres") and ess.get("enable_moods"):
            mode_label = "Genres & Moods"
        elif ess.get("enable_genres"):
            mode_label = "Genres only"
        else:
            mode_label = "Moods only"

        workers = ess.get("workers", 0)
        workers_label = "auto" if workers == 0 else str(workers)

        print(BANNER)
        print(" Settings")
        print(BANNER)

        print("\n  Directories:")
        print(f"    1) Tagged directory      : {settings.get('tagged_dir', '')}")
        print(f"    2) Clean library         : {settings.get('clean_library_dir', '')}")
        print(f"    3) New artist directory  : {settings.get('new_artist_dir', '')}")

        print(f"\n  Essentia:")
        print(f"    4) Analysis mode         : {mode_label}")
        print(f"    5) Number of genres      : {ess.get('top_n_genres', 3)}")
        print(f"    6) Genre threshold       : {ess.get('genre_threshold', 15)}%")
        print(f"    7) Genre format          : {ess.get('genre_format', 'parent_child')}")
        print(f"    8) Mood threshold        : {ess.get('mood_threshold', 0.5)}%")
        conf = "Yes" if ess.get("write_confidence_tags", True) else "No"
        print(f"    9) Confidence tags       : {conf}")
        ow = "Yes" if ess.get("overwrite_existing", False) else "No"
        print(f"   10) Overwrite existing    : {ow}")
        print(f"   11) Workers               : {workers_label}")
        dur = ess.get("max_audio_duration", 300)
        print(f"   12) Max audio duration    : {dur}s")
        print(f"   13) Model directory       : {ess.get('model_dir', '~/essentia_models')}")

        dr = "Yes" if settings.get("dry_run", False) else "No"
        print(f"\n  General:")
        print(f"   14) Dry run               : {dr}")

        print(f"\n    s) Save and go back")
        print(f"    b) Back (discard changes)")
        print()

        choice = input("  Edit [number]: ").strip().lower()

        if choice == "s":
            save_settings(cfg_dir, settings)
            print("  Settings saved.")
            pause()
            return settings

        if choice == "b":
            return load_settings(cfg_dir)

        if choice == "1":
            settings["tagged_dir"] = prompt_path(
                "  Tagged directory", settings.get("tagged_dir", "")
            )
        elif choice == "2":
            settings["clean_library_dir"] = prompt_path(
                "  Clean library", settings.get("clean_library_dir", "")
            )
        elif choice == "3":
            settings["new_artist_dir"] = prompt_path(
                "  New artist directory", settings.get("new_artist_dir", "")
            )
        elif choice == "4":
            print("    1 = Genres & Moods  |  2 = Genres only  |  3 = Moods only")
            m = prompt_int("    Mode", 1, 1, 3)
            ess["enable_genres"] = m in (1, 2)
            ess["enable_moods"] = m in (1, 3)
        elif choice == "5":
            ess["top_n_genres"] = prompt_int(
                "  Genres per track (1-10)", ess.get("top_n_genres", 3), 1, 10
            )
        elif choice == "6":
            ess["genre_threshold"] = prompt_float(
                "  Genre threshold %", ess.get("genre_threshold", 15), 1, 50
            )
        elif choice == "7":
            print("    1=parent_child  2=child_parent  3=child_only  4=raw")
            f = prompt_int("    Format", 1, 1, 4)
            fmt_map = {1: "parent_child", 2: "child_parent", 3: "child_only", 4: "raw"}
            ess["genre_format"] = fmt_map[f]
        elif choice == "8":
            ess["mood_threshold"] = prompt_float(
                "  Mood threshold %", ess.get("mood_threshold", 0.5), 0.01, 20
            )
        elif choice == "9":
            ess["write_confidence_tags"] = not ess.get("write_confidence_tags", True)
        elif choice == "10":
            ess["overwrite_existing"] = not ess.get("overwrite_existing", False)
        elif choice == "11":
            cpu = os.cpu_count() or 2
            ess["workers"] = prompt_int(
                f"  Workers (0=auto, 1-{cpu})", ess.get("workers", 0), 0, cpu
            )
        elif choice == "12":
            ess["max_audio_duration"] = prompt_int(
                "  Max duration (seconds, 0=unlimited)",
                ess.get("max_audio_duration", 300), 0, 3600,
            )
        elif choice == "13":
            ess["model_dir"] = prompt_path(
                "  Model directory", ess.get("model_dir", "~/essentia_models")
            )
        elif choice == "14":
            settings["dry_run"] = not settings.get("dry_run", False)
        else:
            print("  Invalid option.")
            pause()


# ---------------------------------------------------------------------------
# Service setup
# ---------------------------------------------------------------------------

SERVICE_NAME = "music-workflow-watcher"


def setup_service(cfg_dir: Path, settings: dict) -> None:
    """Interactive setup for a systemd user service that watches the tagged directory."""
    tagged_dir = settings.get("tagged_dir", "")
    if not tagged_dir:
        print("  Tagged directory not configured. Please configure settings first.")
        pause()
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Detect virtual environment
    venv_dir = os.environ.get("VIRTUAL_ENV", "")
    if not venv_dir:
        candidate = os.path.join(script_dir, "venv")
        if os.path.isdir(candidate):
            venv_dir = candidate

    clear()
    print(BANNER)
    print(" Service Setup — Automatic File Watcher")
    print(BANNER)
    print()
    print("  This creates a systemd user service that watches your tagged")
    print("  directory and automatically runs the workflow when new files appear.")
    print()
    print("  How it works:")
    print("    • Uses inotifywait (event-driven, near-zero CPU while idle)")
    print("    • Detects new/moved files in the tagged directory")
    print("    • Waits for file activity to settle (debounce)")
    print("    • Runs 'workflow.py --auto' (Essentia → sort → enforce → move)")
    print("    • Starts automatically on boot (with configurable delay)")
    print()

    # --- Check dependency ---
    if shutil.which("inotifywait") is None:
        print("  ⚠️  inotifywait not found!")
        print("  Install with: sudo apt install inotify-tools")
        print()
        if prompt_yes_no("  Install inotify-tools now? (requires sudo)", default=True):
            try:
                subprocess.run(
                    ["sudo", "apt", "install", "-y", "inotify-tools"], check=True
                )
                print()
                print("  ✅ inotify-tools installed.")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("  ❌ Installation failed. Please install manually.")
                pause()
                return
        else:
            print("  Cannot continue without inotify-tools.")
            pause()
            return
    else:
        print("  ✅ inotifywait found")

    print()
    print("  Configuration")
    print("  " + "─" * 68)

    startup_delay = prompt_int(
        "  Startup delay in seconds (wait for remote mounts, 0=none)", 60, 0, 600
    )
    debounce = prompt_int(
        "  Debounce interval in seconds (settle time after last file event)",
        30, 5, 300,
    )
    venv_dir = prompt_path("  Python virtual environment directory", venv_dir)

    if not os.path.isfile(os.path.join(venv_dir, "bin", "activate")):
        print(f"  ⚠️  No activate script found at {venv_dir}/bin/activate")
        if not prompt_yes_no("  Continue anyway?", default=False):
            pause()
            return

    # --- Generate watcher script ---
    watcher_path = os.path.join(script_dir, "watcher.sh")
    watcher_content = f"""#!/usr/bin/env bash
# Music Workflow — File Watcher
# Auto-generated by workflow.py service setup
#
# Tagged directory : {tagged_dir}
# Debounce         : {debounce}s

TAGGED_DIR="{tagged_dir}"
DEBOUNCE={debounce}
SCRIPT_DIR="{script_dir}"
VENV_ACTIVATE="{venv_dir}/bin/activate"

cd "$SCRIPT_DIR" || exit 1
source "$VENV_ACTIVATE" || exit 1

log() {{ echo "$(date '+%Y-%m-%d %H:%M:%S') [music-workflow] $*"; }}

log "Watcher started"
log "Monitoring: $TAGGED_DIR"
log "Debounce: ${{DEBOUNCE}}s"

# Wait for tagged directory to become available (e.g. remote mount)
while [[ ! -d "$TAGGED_DIR" ]]; do
    log "Tagged directory not available yet, retrying in 30s..."
    sleep 30
done

log "Tagged directory available, watching for changes..."

while true; do
    # Block until a file event occurs (zero CPU while waiting)
    inotifywait -r -q -e close_write,moved_to "$TAGGED_DIR" > /dev/null 2>&1

    log "File activity detected, debouncing for ${{DEBOUNCE}}s..."

    # Debounce: keep waiting while new events arrive within the window.
    # inotifywait -t returns exit code 2 on timeout (no new events) which
    # is falsy, ending the loop. Exit code 0 (event detected) continues.
    while inotifywait -r -q -t "$DEBOUNCE" -e close_write,moved_to "$TAGGED_DIR" > /dev/null 2>&1; do
        log "More activity detected, resetting debounce timer..."
    done

    log "Activity settled, running workflow..."
    python workflow.py --auto 2>&1
    log "Workflow complete, resuming watch..."
done
"""

    # --- Generate systemd unit file ---
    service_lines = [
        "[Unit]",
        "Description=Music Workflow — File Watcher",
        "After=remote-fs.target network-online.target",
        "",
        "[Service]",
        "Type=simple",
    ]
    if startup_delay > 0:
        service_lines.append(f"ExecStartPre=/bin/sleep {startup_delay}")
    service_lines.extend([
        f"ExecStart={watcher_path}",
        "Restart=on-failure",
        "RestartSec=30",
        "StandardOutput=journal",
        "StandardError=journal",
        "",
        "[Install]",
        "WantedBy=default.target",
    ])
    service_content = "\n".join(service_lines) + "\n"

    systemd_user_dir = Path.home() / ".config" / "systemd" / "user"
    service_path = systemd_user_dir / f"{SERVICE_NAME}.service"

    # --- Summary ---
    print()
    print("  Summary")
    print("  " + "─" * 68)
    print(f"    Watcher script  : {watcher_path}")
    print(f"    Service file    : {service_path}")
    print(f"    Tagged dir      : {tagged_dir}")
    print(f"    Startup delay   : {startup_delay}s")
    print(f"    Debounce        : {debounce}s")
    print(f"    Virtual env     : {venv_dir}")
    print()

    if not prompt_yes_no("  Proceed with installation?", default=True):
        pause()
        return

    print()

    # Stop existing service if already running
    if service_path.exists():
        print("  ℹ️  Existing service found — updating configuration.")
        subprocess.run(
            ["systemctl", "--user", "stop", f"{SERVICE_NAME}.service"],
            capture_output=True, text=True,
        )

    # Write watcher script
    with open(watcher_path, "w") as f:
        f.write(watcher_content)
    os.chmod(watcher_path, 0o755)
    print(f"  ✅ Watcher script: {watcher_path}")

    # Write service file
    systemd_user_dir.mkdir(parents=True, exist_ok=True)
    with open(service_path, "w") as f:
        f.write(service_content)
    print(f"  ✅ Service file: {service_path}")

    # Reload systemd
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    print("  ✅ systemd daemon reloaded")

    # Enable service
    result = subprocess.run(
        ["systemctl", "--user", "enable", f"{SERVICE_NAME}.service"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("  ✅ Service enabled (will start on boot)")
    else:
        print(f"  ⚠️  Enable failed: {result.stderr.strip()}")

    # Lingering — needed for user services to start at boot before login
    print()
    user = os.getenv("USER", "")
    linger_check = subprocess.run(
        ["loginctl", "show-user", user, "--property=Linger"],
        capture_output=True, text=True,
    )
    linger_enabled = "Linger=yes" in linger_check.stdout

    if linger_enabled:
        print("  ✅ Lingering already enabled (service will start at boot)")
    else:
        print("  Lingering is required for the service to start at boot")
        print("  (before you log in). This requires elevated privileges.")
        print()
        if prompt_yes_no("  Enable lingering now? (requires sudo)", default=True):
            result = subprocess.run(
                ["sudo", "loginctl", "enable-linger", user], check=False
            )
            if result.returncode == 0:
                print("  ✅ Lingering enabled")
            else:
                print(f"  ⚠️  Failed. Run manually: sudo loginctl enable-linger {user}")

    # Start service
    print()
    if prompt_yes_no("  Start the service now?", default=True):
        result = subprocess.run(
            ["systemctl", "--user", "start", f"{SERVICE_NAME}.service"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print("  ✅ Service started!")
        else:
            print(f"  ⚠️  Start failed: {result.stderr.strip()}")

    # Show helpful commands
    print()
    print("  " + "─" * 68)
    print("  Useful commands:")
    print(f"    Status  : systemctl --user status {SERVICE_NAME}")
    print(f"    Logs    : journalctl --user -u {SERVICE_NAME} -f")
    print(f"    Stop    : systemctl --user stop {SERVICE_NAME}")
    print(f"    Restart : systemctl --user restart {SERVICE_NAME}")
    print(f"    Disable : systemctl --user disable {SERVICE_NAME}")
    print()
    pause()


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def main_menu(cfg_dir: Path) -> int:
    settings = load_settings(cfg_dir)

    # First-run check: if no paths are configured, run the wizard
    if not settings.get("tagged_dir"):
        settings = first_run_wizard(cfg_dir)

    definitions = load_definitions(cfg_dir)

    while True:
        clear()
        print(BANNER)
        print(" Music Workflow")
        print(BANNER)

        tagged = settings.get("tagged_dir", "(not set)")
        clean = settings.get("clean_library_dir", "(not set)")
        new_art = settings.get("new_artist_dir", "(not set)")
        dry = "Yes" if settings.get("dry_run") else "No"
        def_count = len([k for k, v in definitions.items() if v and v.strip()])

        # Count pending new artists
        pending_count = 0
        if os.path.isdir(settings.get("new_artist_dir", "")):
            pending_count = len(discover_artists(settings["new_artist_dir"]))

        # Count files waiting in tagged
        tagged_files = 0
        if os.path.isdir(settings.get("tagged_dir", "")):
            tagged_artists = discover_artists(settings["tagged_dir"])
            tagged_files = sum(count_audio_files(p) for _, _, p in tagged_artists)

        print(f"  Tagged directory  : {tagged}", end="")
        if tagged_files:
            print(f"  ({tagged_files} files waiting)")
        else:
            print()
        print(f"  Clean library     : {clean}")
        print(f"  New artist dir    : {new_art}", end="")
        if pending_count:
            print(f"  ({pending_count} artist(s) pending)")
        else:
            print()
        print(f"  Dry run           : {dry}")
        print(f"  Definitions       : {def_count} artist(s) defined")
        print(BANNER)
        print()
        print("  1) Run workflow  (Essentia → sort → enforce → move)")
        print("  2) Manage new artists  (define genres + move to library)")
        print("  3) Settings")
        print("  4) Edit artist definitions")
        print("  5) Toggle dry run")
        print("  6) Setup service  (auto-run on new files)")
        print("  q) Quit")
        print()

        choice = input("  Choice: ").strip().lower()

        if choice == "q":
            return 0

        if choice == "1":
            try:
                with workflow_lock(cfg_dir, blocking=False):
                    run_workflow(settings, definitions, cfg_dir)
            except RuntimeError as e:
                print(f"  {e}")
            pause()

        elif choice == "2":
            definitions = manage_new_artists(settings, definitions, cfg_dir)

        elif choice == "3":
            settings = edit_settings(cfg_dir, settings)

        elif choice == "4":
            definitions = enforcer.edit_defined_artist_picker(definitions, cfg_dir)
            # definitions are saved inside the picker

        elif choice == "5":
            settings["dry_run"] = not settings["dry_run"]
            save_settings(cfg_dir, settings)
            print(f"  Dry run is now: {'ON' if settings['dry_run'] else 'OFF'}")
            pause()

        elif choice == "6":
            setup_service(cfg_dir, settings)

        else:
            print("  Invalid choice.")
            pause()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Music Workflow — Essentia analysis, genre enforcement, library management"
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Run the full workflow non-interactively (for use with the file watcher service)",
    )
    args = parser.parse_args()

    cfg = config_dir()

    if args.auto:
        settings = load_settings(cfg)
        if not settings.get("tagged_dir"):
            print("Error: No settings configured. Run workflow.py interactively first.")
            return 1
        try:
            with workflow_lock(cfg, blocking=False):
                definitions = load_definitions(cfg)
                run_workflow(settings, definitions, cfg, auto=True)
        except RuntimeError as e:
            print(f"Skipping: {e}")
        return 0

    return main_menu(cfg)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(1)

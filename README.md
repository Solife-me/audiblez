# Audiblez: Generate  audiobooks from e-books

[![Installing via pip and running](https://github.com/santinic/audiblez/actions/workflows/pip-install.yaml/badge.svg)](https://github.com/santinic/audiblez/actions/workflows/pip-install.yaml)
[![Git clone and run](https://github.com/santinic/audiblez/actions/workflows/git-clone-and-run.yml/badge.svg)](https://github.com/santinic/audiblez/actions/workflows/git-clone-and-run.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/audiblez)
![PyPI - Version](https://img.shields.io/pypi/v/audiblez)

### v4 Now with Graphical interface, CUDA support, and many languages!

![Audiblez GUI on MacOSX](./imgs/mac.png)

Audiblez generates `.m4b` audiobooks from regular `.epub` e-books and text-based `.pdf` files,
using Kokoro's high-quality speech synthesis, or any OpenAI-compatible TTS server such as Ollama.
Stories can be narrated with a different voice for each character.

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) is a recently published text-to-speech model with just 82M params and very natural sounding output.
It's released under Apache licence and it was trained on < 100 hours of audio.
It currently supports these languages: 🇺🇸 🇬🇧 🇪🇸 🇫🇷 🇮🇳 🇮🇹 🇯🇵 🇧🇷 🇨🇳

On a Google Colab's T4 GPU via Cuda, **it takes about 5 minutes to convert "Animal's Farm" by Orwell** (which is about 160,000 characters) to audiobook, at a rate of about 600 characters per second.

On my M2 MacBook Pro, on CPU, it takes about 1 hour, at a rate of about 60 characters per second.


## How to install the Command Line tool

If you have Python 3.11 or 3.12 on your computer, you can install it with pip (Kokoro doesn't support Python 3.13+ yet).
You also need `espeak-ng` and `ffmpeg` installed on your machine:

```bash
sudo apt install ffmpeg espeak-ng                   # on Ubuntu/Debian 🐧
pip install audiblez
```

```bash
brew install ffmpeg espeak-ng                       # on Mac 🍏
pip install audiblez
```

Then you can convert an .epub directly with:

```
audiblez book.epub -v af_sky
```

It will first create a bunch of `book_chapter_1.wav`, `book_chapter_2.wav`, etc. files in the same directory,
and at the end it will produce a `book.m4b` file with the whole book you can listen with VLC or any
audiobook player.
It will only produce the `.m4b` file if you have `ffmpeg` installed on your machine.

## PDFs

PDFs work like e-books: `audiblez sermons.pdf`. Audiblez rebuilds paragraphs from the page layout,
drops running headers, footers and page numbers, rejoins paragraphs and hyphenated words across line and page breaks,
and splits the book into chapters using the PDF's bookmarks (or larger-font headings when there are none).
Long sections are split into parts of at most ~40 minutes, always between two paragraphs, so a cut never lands
mid-sentence. Audio is written to disk as it's generated, so memory use stays flat however long the book is.
Only PDFs with a text layer are supported (no OCR for scanned pages).

## Other TTS providers (Ollama and OpenAI-compatible servers)

Besides the built-in Kokoro, audiblez can use any server that implements the OpenAI `/v1/audio/speech` API,
like Ollama's speech models (e.g. `phonon`), OpenAI, Kokoro-FastAPI or LocalAI:

```bash
audiblez book.epub --base-url http://localhost:11434/v1 --model phonon -v Hugh
```

For regular use, create a config file with `audiblez --init-config` and edit
`~/.config/audiblez/config.toml`:

```toml
[tts]
provider = "ollama"          # a provider below, or "kokoro"
model = "phonon"
voice = "Hugh"
speed = 1.0

[llm]                        # used for character voices and --auto-narrator
provider = "ollama"
model = "gemma4:26b-mlx"     # or a cloud model such as "kimi-k3:cloud"
reasoning_effort = "none"    # faster; "low"/"medium" lets thinking models think

[providers.ollama]
type = "openai"
base_url = "http://MacBook-Pro-2.local:11434/v1"
# api_key_env = "OPENAI_API_KEY"

[providers.ollama.voices]    # gender/accent tags used for casting
Hugh = { gender = "male", accent = "british" }
"The Snow Queen" = { gender = "female", style = "character" }  # never cast as an ordinary person
```

`audiblez --list-voices` lists the voices of the selected provider. `audiblez --audition-voices` renders a sample of
every voice into one `.m4b` (one chapter per voice) and prints a `[providers.<name>.voices]` snippet with
pitch-based gender guesses for you to correct and paste into the config. `--provider`, `--model` and `--base-url`
override the config for one run. If a server only supports speed 1.0, the speed is applied when the `.m4b` is made.

## Character voices

```bash
audiblez novel.epub --characters            # 1. detect characters, write novel.cast.toml, stop
audiblez novel.epub --characters            # 2. synthesize with the (edited) cast
```

With `--characters`, an LLM (the `[llm]` section, or `--llm-model`) reads the book a few pages at a time and labels
who speaks every quotation, keeping a running list of characters with their gender. Narration stays in the
narrator's voice, and each character gets a voice of their gender, preferring the narrator's accent. Major characters
get their own voices; minor ones share. The first run writes `<book>.cast.toml` and stops so you can review and
change voices. The second run makes the audiobook. Attributions are cached in `<book>.speakers.json`, so changing
voices doesn't call the LLM again. Add `--no-review` to do both steps at once.

To make the cast sound like a conversation rather than clips stitched together, every clip is trimmed of the silence
TTS models pad it with and followed by a pause that fits (short after "…,” said she", a turn-taking gap between
speakers), and every clip is levelled to the same loudness. A quote split by a dialogue tag ("“My dear,” said she,
“have you heard?”") is spoken as one utterance and cut at its pause, so it keeps its intonation.
Replies are also primed: each reply is synthesized right after the line it answers, in the replying voice, and that
line is then cut off, so the reply is spoken as a response. Dialogue tags are primed the same way with the end of the
quote before them, since TTS models mangle a fragment like "said her mother" spoken on its own
(`--no-prime-replies`, or `prime_replies = false` under `[characters]`, turns this off). Characters with the most
lines get the most expressive voices of their gender: `--audition-voices` measures each voice's pitch variation
and prints it as `expressiveness` for the config.

## Speech clean-up and verification

Before speech, abbreviations that TTS models misread are spelled out (Mr., Mrs., Dr., St., &c.), old spellings
are modernized (to-morrow, to-day) and Roman numerals after Chapter/Book/Part become numbers. Every clip is trimmed
of the silence models pad it with, levelled to the same loudness, and regenerated if it has a long silence in the
middle or a length that doesn't fit its text.

With `pip install "audiblez[verify]"`, audiblez also listens to every clip with a small local speech recognizer
(faster-whisper) and regenerates clips where the TTS model dropped or invented words (it happens: phonon once
replaced half a sentence of a Spurgeon sermon with five seconds of silence). It's used automatically when installed
and roughly doubles synthesis time; `--no-verify` or `verify = false` under `[tts]` turns it off.

`--auto-narrator` asks the LLM about the author (or first-person narrator) and picks a narrator voice of the matching
gender and accent, e.g. a British male voice for a Charles Spurgeon sermon. It only switches voice when the
LLM is confident and a matching voice is tagged. `-v` always wins.

### Voice profiles and casting

Casting needs to know what each voice sounds like. Kokoro's voices are tagged from their names (`bm_george` is a
British man). For other providers, `audiblez --profile-voices` measures every voice and prints a config table;
add `--update-config` to write it straight into your config:

```bash
pip install "audiblez[profile]"
audiblez --profile-voices --update-config
audiblez --list-voices       # gender, age, accent, pitch and expressiveness of every voice
```

It renders a few sentences per voice and measures gender and apparent age with audEERING's
[age/gender model](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender) (licensed for
non-commercial use), accent with [CommonAccent](https://huggingface.co/Jzuluaga/accent-id-commonaccent_ecapa), and
pitch and expressiveness (pitch variation on dialogue lines) directly. Mark creature/novelty voices with
`style = "character"` so they're only cast as creatures.

With an LLM configured, the LLM casts the characters: it gets every voice's profile plus each character's
gender, age, line count and a short description of their personality from speaker detection, and follows explicit
rules (match gender and age, distinct voices for main characters, lively voices for excitable characters, the
setting's accent, no novelty voices for ordinary people). Its choices are checked (gender, a main character never
shares a voice) and the reason for each is written on the cast sheet. `llm_casting = false` under `[characters]`
uses the built-in rules instead. Untagged voices are never cast.

## Hermes agent skill

[`skills/audiblez`](./skills/audiblez/SKILL.md) is a skill for the [Hermes agent](https://hermes-agent.nousresearch.com),
so it can install audiblez and make audiobooks for you:

```bash
hermes skills install Solife-me/audiblez/skills/audiblez
```

## How to run the GUI

The GUI is a simple graphical interface to use audiblez.
You need some extra dependencies to run the GUI:

```
sudo apt install ffmpeg espeak-ng 
sudo apt install libgtk-3-dev        # just for Ubuntu/Debian 🐧, Windows/Mac don't need this
  
pip install audiblez pillow wxpython
```

Then you can run the GUI with:
```
audiblez-ui
```

## How to run on Windows

After many trials, on Windows we recommend to install audiblez in a Python venv:

1. Open a Windows terminal
2. Create anew folder: `mkdir audiblez`
3. Enter the folder: `cd audiblez`
4. Create a venv: `python -m venv venv`
5. Activate the venv: `.\venv\Scripts\Activate.ps1`
6. Install the dependencies: `pip install audiblez pillow wxpython`
7. Now you can run `audiblez` or `audiblez-ui`
8. For Cuda support, you need to install Pytorch accordingly: https://pytorch.org/get-started/locally/


## Speed

By default the audio is generated using a normal speed, but you can make it up to twice slower or faster by specifying a speed argument between 0.5 to 2.0:

```
audiblez book.epub -v af_sky -s 1.5
```

## Supported Voices

Use `-v` option to specify the voice to use. Available voices are listed here.
The first letter is the language code and the second is the gender of the speaker e.g. `im_nicola` is an italian male voice.

[For hearing samples of Kokoro-82M voices, go here](https://claudio.uk/posts/audiblez-v4.html)

| Language                  | Voices                                                                                                                                                                                                                                     |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 🇺🇸 American English     | `af_alloy`, `af_aoede`, `af_bella`, `af_heart`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`, `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa` |
| 🇬🇧 British English      | `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`, `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`                                                                                                                                          |
| 🇪🇸 Spanish              | `ef_dora`, `em_alex`, `em_santa`                                                                                                                                                                                                           |
| 🇫🇷 French               | `ff_siwis`                                                                                                                                                                                                                                 |
| 🇮🇳 Hindi                | `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`                                                                                                                                                                                                |
| 🇮🇹 Italian              | `if_sara`, `im_nicola`                                                                                                                                                                                                                     |
| 🇯🇵 Japanese             | `jf_alpha`, `jf_gongitsune`, `jf_nezumi`, `jf_tebukuro`, `jm_kumo`                                                                                                                                                                         |
| 🇧🇷 Brazilian Portuguese | `pf_dora`, `pm_alex`, `pm_santa`                                                                                                                                                                                                           |
| 🇨🇳 Mandarin Chinese     | `zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi`, `zm_yunjian`, `zm_yunxi`, `zm_yunxia`, `zm_yunyang`                                                                                                                                 |

For more detaila about voice quality, check this document: [Kokoro-82M voices](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)

## How to run on GPU

By default, audiblez runs on CPU. If you pass the option `--cuda` it will try to use the Cuda device via Torch.

Check out this example: [Audiblez running on a Google Colab Notebook with Cuda ](https://colab.research.google.com/drive/164PQLowogprWQpRjKk33e-8IORAvqXKI?usp=sharing]).

We don't currently support Apple Silicon, as there is not yet a Kokoro implementation in MLX. As soon as it will be available, we will support it.

## Manually pick chapters to convert

Sometimes you want to manually select which chapters/sections in the e-book to read out loud.
To do so, you can use `--pick` to interactively choose the chapters to convert (without running the GUI).


## Help page

For all the options available, you can check the help page `audiblez --help`:

```
usage: audiblez [-h] [-v VOICE] [-p] [-s SPEED] [-c] [-o FOLDER] [--no-verify] [--max-chapters N]
                [--provider PROVIDER] [--model MODEL] [--base-url URL] [--list-voices]
                [--config FILE] [--init-config] [--profile-voices] [--update-config]
                [--audition-voices] [--characters] [--no-review]
                [--prime-replies | --no-prime-replies] [--auto-narrator]
                [--llm-provider LLM_PROVIDER] [--llm-model LLM_MODEL]
                [file_path]

positional arguments:
  file_path             Path to the epub or pdf file

options:
  -h, --help            show this help message and exit
  -v VOICE, --voice VOICE
                        Choose narrating voice (default: from config, or af_sky). Kokoro voices
                        are listed below
  -p, --pick            Interactively select which chapters to read in the audiobook
  -s SPEED, --speed SPEED
                        Set speed from 0.5 to 2.0
  -c, --cuda            Use GPU via Cuda in Torch if available
  -o FOLDER, --output FOLDER
                        Output folder for the audiobook and temporary files
  --no-verify           Skip checking each clip with speech recognition (used when faster-whisper
                        is installed: pip install "audiblez[verify]")
  --max-chapters N      Only read the first N selected chapters (handy for a quick test)

TTS provider:
  --provider PROVIDER   TTS provider name from the config, or "kokoro" (default: from config, or
                        kokoro)
  --model MODEL         TTS model for OpenAI-compatible providers, e.g. phonon
  --base-url URL        Base URL of an OpenAI-compatible TTS server, e.g.
                        http://localhost:11434/v1
  --list-voices         List the voices of the selected provider and exit
  --config FILE         Config file (default: ~/.config/audiblez/config.toml)
  --init-config         Write a starter config file and exit
  --profile-voices      Measure every voice's gender, age, accent, pitch and expressiveness (needs
                        "audiblez[profile]") and print a config snippet; with --update-config,
                        write it into the config
  --update-config       With --profile-voices: replace the provider's voices table in the config
                        with the measured profiles
  --audition-voices     Render a sample of every voice into one m4b, print a config snippet, and
                        exit

character voices (needs an LLM, see [llm] in the config):
  --characters          Read each character's dialogue in their own voice. The first run detects
                        the characters and writes <book>.cast.toml for review; run again to
                        synthesize
  --no-review           With --characters, synthesize right away without stopping to review the
                        cast sheet
  --prime-replies, --no-prime-replies
                        Synthesize each reply right after the line it answers (then cut that off),
                        so it flows as a response (default: on; [characters] prime_replies in the
                        config)
  --auto-narrator       Pick the narrator voice from the author's or narrator's gender and origin
                        (e.g. a British male voice for Spurgeon)
  --llm-provider LLM_PROVIDER
                        Provider for the LLM (default: [llm] provider in the config)
  --llm-model LLM_MODEL
                        LLM model, e.g. gemma4:26b-mlx or kimi-k3:cloud (default: [llm] model in
                        the config)

examples:
  audiblez book.epub -v af_sky
  audiblez sermons.pdf --provider ollama --model phonon -v Hugh
  audiblez --list-voices --provider ollama
  audiblez novel.epub --characters --llm-model gemma4:26b-mlx

to configure providers, run:
  audiblez --init-config

to run GUI just run:
  audiblez-ui
```

## Author

by [Claudio Santini](https://claudio.uk) in 2025, distributed under MIT licence.

Related Article: [Audiblez v4: Generate Audiobooks from E-books](https://claudio.uk/posts/audiblez-v4.html)

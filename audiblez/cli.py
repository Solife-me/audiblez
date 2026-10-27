# -*- coding: utf-8 -*-
import argparse
import sys

from audiblez.voices import available_voices_str


def cli_main():
    epilog = ('examples:\n' +
              '  audiblez book.epub -v af_sky\n' +
              '  audiblez sermons.pdf --provider ollama --model phonon -v Hugh\n' +
              '  audiblez --list-voices --provider ollama\n' +
              '  audiblez novel.epub --characters --llm-model gemma4:26b-mlx\n\n' +
              'to configure providers, run:\n'
              '  audiblez --init-config\n\n' +
              'to run GUI just run:\n'
              '  audiblez-ui\n\n' +
              'available Kokoro voices:\n' +
              available_voices_str)
    parser = argparse.ArgumentParser(epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('file_path', nargs='?', help='Path to the epub or pdf file')
    parser.add_argument('-v', '--voice', default=None, help='Choose narrating voice (default: from config, or af_sky). Kokoro voices are listed below')
    parser.add_argument('-p', '--pick', default=False, help=f'Interactively select which chapters to read in the audiobook', action='store_true')
    parser.add_argument('-s', '--speed', default=None, help=f'Set speed from 0.5 to 2.0', type=float)
    parser.add_argument('-c', '--cuda', default=False, help=f'Use GPU via Cuda in Torch if available', action='store_true')
    parser.add_argument('-o', '--output', default='.', help='Output folder for the audiobook and temporary files', metavar='FOLDER')
    parser.add_argument('--no-verify', action='store_true', help='Skip checking each clip with speech recognition (used when faster-whisper is installed: pip install "audiblez[verify]")')
    parser.add_argument('--skip-footnotes', action=argparse.BooleanOptionalAction, default=None, help='Don\'t read footnotes, endnotes and their reference markers (default: on; [text] skip_footnotes in the config)')
    parser.add_argument('--roman-numerals', action=argparse.BooleanOptionalAction, default=None, help='Read Roman numerals as numbers where context makes them certain: CHAPTER XII, "II. Secondly", Charles II, chap. lxiii (default: on; [text] roman_numerals in the config)')
    parser.add_argument('--foreign-scripts', choices=['transliterate', 'skip', 'read'], default=None, help='Greek and Hebrew quotations, which no TTS voice can read: transliterate them into English letters ("tremontas kai phoboumenous"), skip them, or read them as written (default: transliterate; [text] foreign_scripts in the config)')
    parser.add_argument('--article', action='store_true', help='Read only an article\'s title, author, date and body, skipping ads, menus, cookie notices, links, captions and references (for PDFs of web pages; uses the [llm] if configured)')
    parser.add_argument('--max-chapters', type=int, default=None, metavar='N', help='Only read the first N selected chapters (handy for a quick test)')

    tts = parser.add_argument_group('TTS provider')
    tts.add_argument('--provider', default=None, help='TTS provider name from the config, or "kokoro" (default: from config, or kokoro)')
    tts.add_argument('--model', default=None, help='TTS model for OpenAI-compatible providers, e.g. phonon')
    tts.add_argument('--base-url', default=None, metavar='URL', help='Base URL of an OpenAI-compatible TTS server, e.g. http://localhost:11434/v1')
    tts.add_argument('--list-voices', action='store_true', help='List the voices of the selected provider and exit')
    tts.add_argument('--config', default=None, metavar='FILE', help='Config file (default: ~/.config/audiblez/config.toml)')
    tts.add_argument('--init-config', action='store_true', help='Write a starter config file and exit')
    tts.add_argument('--profile-voices', action='store_true', help='Measure every voice\'s gender, age, accent, pitch and expressiveness (needs "audiblez[profile]") and print a config snippet; with --update-config, write it into the config')
    tts.add_argument('--update-config', action='store_true', help='With --profile-voices: replace the provider\'s voices table in the config with the measured profiles')
    tts.add_argument('--audition-voices', action='store_true', help='Render a sample of every voice into one m4b, print a config snippet, and exit')

    cast = parser.add_argument_group('character voices (needs an LLM, see [llm] in the config)')
    cast.add_argument('--characters', action='store_true', help='Read each character\'s dialogue in their own voice. The first run detects the characters and writes <book>.cast.toml for review; run again to synthesize')
    cast.add_argument('--no-review', action='store_true', help='With --characters, synthesize right away without stopping to review the cast sheet')
    cast.add_argument('--prime-replies', action=argparse.BooleanOptionalAction, default=None, help='Synthesize each reply right after the line it answers (then cut that off), so it flows as a response (default: on; [characters] prime_replies in the config)')
    cast.add_argument('--auto-narrator', action='store_true', help='Pick the narrator voice from the author\'s or narrator\'s gender and origin (e.g. a British male voice for Spurgeon)')
    cast.add_argument('--llm-provider', default=None, help='Provider for the LLM (default: [llm] provider in the config)')
    cast.add_argument('--llm-model', default=None, help='LLM model, e.g. gemma4:26b-mlx or kimi-k3:cloud (default: [llm] model in the config)')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    from audiblez.config import load_config, write_config_template, provider_settings
    if args.init_config:
        print('Wrote', write_config_template(args.config))
        return
    cfg = load_config(args.config)
    if '_path' in cfg:
        print('Using config', cfg['_path'])

    from audiblez.tts import create_engine
    provider = args.provider or ('adhoc' if args.base_url else cfg['tts']['provider'])
    engine = create_engine(cfg, provider=provider, model=args.model, base_url=args.base_url)

    if args.list_voices:
        print_voices(engine)
        return
    if args.profile_voices:
        from audiblez.profile import profile_voices, voices_section, update_config
        from audiblez.config import config_path
        profiles = profile_voices(engine, cache_dir=config_path(args.config).parent)
        if args.update_config:
            update_config(config_path(args.config), engine.name, profiles)
            print(f'Updated [providers.{engine.name}.voices] in {config_path(args.config)}')
        else:
            print('\n' + voices_section(engine.name, profiles))
        return
    if args.audition_voices:
        from audiblez.audition import audition
        audition(engine, args.output)
        return
    if not args.file_path:
        parser.error('the path of an epub or pdf file is required')

    voice = args.voice
    if voice is None:
        if provider == cfg['tts']['provider']:
            voice = cfg['tts']['voice']
        else:
            voice = provider_settings(cfg, provider).get('default_voice') or ('af_sky' if engine.name == 'kokoro' else 'default')
    speed = args.speed if args.speed is not None else float(cfg['tts'].get('speed', 1.0))

    if args.cuda:
        import torch.cuda
        if torch.cuda.is_available():
            print('CUDA GPU available')
            torch.set_default_device('cuda')
        else:
            print('CUDA GPU not available. Defaulting to CPU')

    llm = None
    if args.characters or args.auto_narrator or args.article:
        from audiblez.llm import create_llm
        try:
            llm = create_llm(cfg, provider=args.llm_provider, model=args.llm_model)
            print(f'Using LLM {llm.model} at {llm.base_url}')
        except ValueError:
            if not args.article:
                raise

    from audiblez.verify import create_verifier
    verifier = None if args.no_verify else create_verifier(cfg)

    from audiblez.core import main
    main(args.file_path, voice, args.pick, speed, args.output, max_chapters=args.max_chapters, engine=engine,
         characters=args.characters, llm=llm, review=not args.no_review, auto_narrator=args.auto_narrator and args.voice is None,
         prime_replies=args.prime_replies if args.prime_replies is not None else bool(cfg['characters'].get('prime_replies')),
         verifier=verifier, llm_casting=bool(cfg['characters'].get('llm_casting', True)),
         skip_footnotes=args.skip_footnotes if args.skip_footnotes is not None else bool(cfg['text'].get('skip_footnotes', True)),
         roman_numerals=args.roman_numerals if args.roman_numerals is not None else bool(cfg['text'].get('roman_numerals', True)),
         foreign_scripts=args.foreign_scripts or cfg['text'].get('foreign_scripts', 'transliterate'), article=args.article)


def print_voices(engine):
    from tabulate import tabulate
    rows = [[v.name, v.gender, f'{v.age} (~{v.age_years:.0f})' if v.age_years else v.age,
             v.accent + ('?' if v.accent_confidence < 0.75 else ''), v.pitch or '', v.expressiveness or '', v.style]
            for v in engine.list_voices().values()]
    if not rows:
        print(f'{engine.name} did not report its voices. Add them to the config under [providers.{engine.name}.voices].')
        return
    print(tabulate(rows, headers=['Voice', 'Gender', 'Age', 'Accent', 'Pitch (Hz)', 'Expressiveness', 'Style']))


if __name__ == '__main__':
    cli_main()

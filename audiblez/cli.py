# -*- coding: utf-8 -*-
import argparse
import sys

from audiblez.voices import voices, available_voices_str


def cli_main():
    voices_str = ', '.join(voices)
    epilog = ('examples:\n' +
              '  audiblez book.epub -v af_sky\n' +
              '  audiblez sermons.pdf --provider ollama --model phonon -v Hugh\n' +
              '  audiblez --list-voices --provider ollama\n\n' +
              'to configure providers, run:\n'
              '  audiblez --init-config\n\n' +
              'to run GUI just run:\n'
              '  audiblez-ui\n\n' +
              'available Kokoro voices:\n' +
              available_voices_str)
    parser = argparse.ArgumentParser(epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('file_path', nargs='?', help='Path to the epub or pdf file')
    parser.add_argument('-v', '--voice', default=None, help=f'Choose narrating voice (default: from config, or af_sky). Kokoro voices: {voices_str}')
    parser.add_argument('-p', '--pick', default=False, help=f'Interactively select which chapters to read in the audiobook', action='store_true')
    parser.add_argument('-s', '--speed', default=None, help=f'Set speed from 0.5 to 2.0', type=float)
    parser.add_argument('-c', '--cuda', default=False, help=f'Use GPU via Cuda in Torch if available', action='store_true')
    parser.add_argument('-o', '--output', default='.', help='Output folder for the audiobook and temporary files', metavar='FOLDER')
    parser.add_argument('--max-chapters', type=int, default=None, metavar='N', help='Only read the first N selected chapters (handy for a quick test)')

    tts = parser.add_argument_group('TTS provider')
    tts.add_argument('--provider', default=None, help='TTS provider name from the config, or "kokoro" (default: from config, or kokoro)')
    tts.add_argument('--model', default=None, help='TTS model for OpenAI-compatible providers, e.g. phonon')
    tts.add_argument('--base-url', default=None, metavar='URL', help='Base URL of an OpenAI-compatible TTS server, e.g. http://localhost:11434/v1')
    tts.add_argument('--list-voices', action='store_true', help='List the voices of the selected provider and exit')
    tts.add_argument('--config', default=None, metavar='FILE', help='Config file (default: ~/.config/audiblez/config.toml)')
    tts.add_argument('--init-config', action='store_true', help='Write a starter config file and exit')

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

    from audiblez.core import main
    main(args.file_path, voice, args.pick, speed, args.output, max_chapters=args.max_chapters, engine=engine)


def print_voices(engine):
    from tabulate import tabulate
    rows = [[v.name, v.gender, v.accent] for v in engine.list_voices().values()]
    if not rows:
        print(f'{engine.name} did not report its voices. Add them to the config under [providers.{engine.name}.voices].')
        return
    print(tabulate(rows, headers=['Voice', 'Gender', 'Accent']))


if __name__ == '__main__':
    cli_main()

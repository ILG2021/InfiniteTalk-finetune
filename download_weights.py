"""Download only the files used by this trainer, from pinned Hugging Face revisions.

Default: print an offline plan. --download transfers weights; --check checks local
file presence only. Set HF_ENDPOINT/HF_TOKEN externally if needed.
"""
import argparse
import json
from pathlib import Path


WAN_FILES = (
    'config.json', 'diffusion_pytorch_model.safetensors.index.json',
    *(f'diffusion_pytorch_model-{i:05d}-of-00007.safetensors' for i in range(1, 8)),
    'Wan2.1_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth',
    'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth',
    *(f'google/umt5-xxl/{name}' for name in (
        'special_tokens_map.json', 'spiece.model', 'tokenizer.json', 'tokenizer_config.json')),
    *(f'xlm-roberta-large/{name}' for name in (
        'sentencepiece.bpe.model', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json')),
)
MODELS = {
    'wan': dict(repo='Wan-AI/Wan2.1-I2V-14B-480P', directory='Wan2.1-I2V-14B-480P',
                revision='6b73f84e66371cdfe870c72acd6826e1d61cf279', files=WAN_FILES),
    'infinitetalk': dict(repo='MeiGen-AI/InfiniteTalk', directory='InfiniteTalk',
                         revision='d59847ebdacf19245bfca3fb23311c0cada8378a',
                         files=('single/infinitetalk.safetensors',)),
    # Same conversion branch linked by upstream InfiniteTalk's README, pinned
    # to a commit rather than a movable refs/pr/1. Avoid duplicate .bin/fairseq.
    'audio': dict(repo='TencentGameMate/chinese-wav2vec2-base', directory='chinese-wav2vec2-base',
                  revision='5f7a6cfdfc5440ec78748f79a9dd66f077c2b463',
                  files=('config.json', 'preprocessor_config.json', 'model.safetensors')),
    'caption': dict(repo='Qwen/Qwen3-VL-2B-Instruct', directory='Qwen3-VL-2B-Instruct',
                    revision='89644892e4d85e24eaac8bacfd4f463576704203',
                    files=('config.json', 'generation_config.json', 'model.safetensors',
                           'chat_template.json', 'preprocessor_config.json', 'video_preprocessor_config.json',
                           'tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json')),
}
MODELS['wan_aux'] = dict(MODELS['wan'], files=tuple(f for f in WAN_FILES if not f.startswith('diffusion_pytorch_model')))
MODELS['fp8'] = dict(MODELS['infinitetalk'], files=(
    'quant_models/infinitetalk_single_fp8.safetensors', 'quant_models/infinitetalk_single_fp8.json'))
PROFILES = {'training': ('wan_aux', 'fp8', 'audio'),
            'training-original': ('wan', 'infinitetalk', 'audio'),
            'fp8': ('fp8',), 'audio': ('audio',), 'caption': ('caption',),
            'all': ('wan_aux', 'fp8', 'audio', 'caption')}


def selected_models(profile):
    return [MODELS[name] for name in PROFILES[profile]]


def missing_files(root, models):
    missing = []
    for model in models:
        for filename in model['files']:
            path = root/model['directory']/filename
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(path)
    return missing


def download(root, models, downloader):
    for model in models:
        destination = root/model['directory']
        for filename in model['files']:
            print(f"Downloading {model['repo']}: {filename}", flush=True)
            downloader(repo_id=model['repo'], filename=filename,
                       revision=model['revision'], local_dir=str(destination))
        missing = missing_files(root, [model])
        if missing:
            raise RuntimeError(f'Download incomplete: {missing[0]}')
        receipt = destination/'.download-revision.json'
        temp = receipt.with_suffix('.json.tmp')
        temp.write_text(json.dumps(model, indent=2), encoding='utf-8')
        temp.replace(receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights_dir', type=Path, default=Path('weights'))
    parser.add_argument('--profile', choices=PROFILES, default='training')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--download', action='store_true')
    action.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    root = args.weights_dir.resolve()
    models = selected_models(args.profile)
    if args.download:
        from huggingface_hub import hf_hub_download
        download(root, models, hf_hub_download)
    elif not args.check:
        for model in models:
            print(f"{model['repo']} @ {model['revision']} -> {root/model['directory']}")
            for filename in model['files']:
                print(f'  {filename}')
        print('Plan only. Add --download to download, or --check for offline file-presence checks.')
        return 0
    missing = missing_files(root, models)
    for path in missing:
        print(f'Missing or empty: {path}')
    if missing:
        return 1
    print('All selected files are present and non-empty (not a checksum or model-load validation).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

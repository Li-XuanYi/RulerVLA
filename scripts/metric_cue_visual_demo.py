import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

VIEWS = (
    ('agentview', 'observation/image', 'observation/agent_cue'),
    ('wrist', 'observation/wrist_image', 'observation/wrist_cue'),
)


def preview_overlay(rgb, cue):
    rgb = np.asarray(rgb)
    cue = np.asarray(cue)
    if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
        raise ValueError('Expected real uint8 policy RGB at 224x224')
    if cue.shape != rgb.shape or cue.dtype != np.float32 or not np.isfinite(cue).all():
        raise ValueError('Expected finite float32 spatial cue at 224x224')
    if cue.min() < 0 or cue.max() > 1:
        raise ValueError('Spatial cue must be in [0, 1]')
    strength = cue.max(axis=-1, keepdims=True)
    color = np.divide(cue, strength, out=np.zeros_like(cue), where=strength > 0)
    alpha = .65 * strength
    return Image.fromarray(np.rint(rgb * (1-alpha) + color * 255 * alpha).astype(np.uint8))


def render_panel(title, prepared):
    size = 224
    gap = 16
    width = size * 2 + gap * 3
    height = 556
    canvas = Image.new('RGB', (width, height), '#ffffff')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((gap, 9), title, fill='#111111', font=font)
    for column, (view, rgb_key, cue_key) in enumerate(VIEWS):
        left = gap + column * (size + gap)
        rgb = prepared[rgb_key]
        cue = prepared[cue_key]
        draw.text((left, 39), f'{view}: RGB + cue preview', fill='#111111', font=font)
        canvas.paste(preview_overlay(rgb, cue), (left, 57))
        draw.text((left, 298), f'{view}: cue channels', fill='#111111', font=font)
        cue_rgb = np.rint(cue * 255).astype(np.uint8)
        canvas.paste(Image.fromarray(cue_rgb), (left, 316))
    return canvas.resize((width * 2, height * 2), Image.Resampling.NEAREST)


def main():
    from rulervla.adapters.metric_cue import MetricViewAdapter
    from rulervla.config import load_metric_method_config
    from rulervla.cues.motion_echo import MotionHistory

    parser = argparse.ArgumentParser(description='真实 LIBERO 双视角标尺与 Echo 视觉示例')
    parser.add_argument('--sample', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--previous', type=Path)
    args = parser.parse_args()
    sample_path = args.sample.resolve()
    with np.load(sample_path, allow_pickle=False) as source:
        sample = {key: source[key] for key in source.files}
    sample['task'] = str(sample['task'])
    sample.pop('actions', None)

    geometry, ruler, echo = load_metric_method_config(ROOT / 'configs/metric_ruler.yaml')
    configurations = {
        'blank': (geometry, replace(ruler, enabled=False, mode='blank'), echo),
        'ruler': (geometry, ruler, echo),
        'combined': (geometry, ruler, replace(echo, enabled=True)),
    }
    geometry = configurations['ruler'][0]
    if any(value[0] != geometry for value in configurations.values()):
        raise ValueError('Demo modes have different camera geometry contracts')
    echo_config = configurations['combined'][2]
    if not echo_config.enabled or configurations['ruler'][2].enabled:
        raise ValueError('Demo configs do not isolate the ruler and Echo')
    current_step = int(sample['frame_index'])
    history = MotionHistory(
        True, int(sample['episode_index']), current_step,
        current_step - echo_config.history_steps,
        sample['ee_world_m'], sample['past_ee_world_m'],
    )
    if current_step < echo_config.history_steps:
        raise ValueError('Sample has insufficient real EE history')

    variants = (
        ('ruler', 'Metric ruler', configurations['ruler'][1], configurations['ruler'][2], None),
        ('echo', 'Motion Echo', configurations['blank'][1], echo_config, history),
        ('ruler_echo', 'Metric ruler + Motion Echo', configurations['combined'][1], echo_config, history),
    )
    prepared_by_mode = {}
    metadata = {}
    for name, title, ruler, echo, current_history in variants:
        adapter = MetricViewAdapter(geometry, ruler, echo)
        prepared = adapter.prepare(sample, history=current_history)
        prepared_by_mode[name] = prepared
        metadata[name] = {'prompt': prepared['prompt'], 'views': {}}
        for view, _, cue_key in VIEWS:
            cue = prepared[cue_key]
            if not np.any(cue):
                raise ValueError(f'{name} has an empty cue in {view}')
            echo_info = adapter.last_diagnostics[view]['echo']
            if echo.enabled and not echo_info['rendered']:
                raise ValueError(f'{name} Echo is not visible in {view}: {echo_info}')
            metadata[name]['views'][view] = {
                'cue_sha256': hashlib.sha256(np.ascontiguousarray(cue).tobytes()).hexdigest(),
                'nonzero_pixels': int(np.count_nonzero(cue.max(axis=-1))),
                'echo': echo_info,
            }
    for _, rgb_key, _ in VIEWS:
        reference = prepared_by_mode['ruler'][rgb_key]
        if any(not np.array_equal(reference, prepared[rgb_key])
               for prepared in prepared_by_mode.values()):
            raise ValueError('Demo modes changed the original RGB image')

    args.out.mkdir(parents=True, exist_ok=False)
    panels = []
    for name, title, _, _, _ in variants:
        panel = render_panel(title, prepared_by_mode[name])
        panel.save(args.out / f'{name}.png')
        for view, rgb_key, cue_key in VIEWS:
            cue = prepared_by_mode[name][cue_key]
            Image.fromarray(np.rint(cue*255).astype(np.uint8)).save(args.out / f'{name}_{view}_cue.png')
            preview_overlay(prepared_by_mode[name][rgb_key], cue).save(args.out / f'{name}_{view}_overlay.png')
        panels.append(panel)
    gap = 24
    contact = Image.new('RGB', (panels[0].width * 3 + gap * 4,
                                panels[0].height + gap * 2), '#e9edf2')
    for index, panel in enumerate(panels):
        contact.paste(panel, (gap + index * (panel.width + gap), gap))
    contact.save(args.out / 'comparison.png')
    if args.previous is not None:
        comparison = Image.new('RGB', (panels[0].width*2, (panels[0].height+32)*2), 'white')
        draw = ImageDraw.Draw(comparison)
        for row, (name, new_panel) in enumerate((('echo', panels[1]), ('ruler_echo', panels[2]))):
            for column, (label, panel) in enumerate((('Previous', Image.open(args.previous / f'{name}.png')),
                                                    ('Current', new_panel))):
                left, top = column*panels[0].width, row*(panels[0].height+32)
                draw.text((left+16, top+8), label, fill='black')
                comparison.paste(panel, (left, top+32))
        comparison.save(args.out / 'render_comparison.png')
    manifest = {
        'schema': 'rulervla_metric_visual_demo_v1',
        'sample': str(sample_path),
        'sample_sha256': hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        'episode_index': int(sample['episode_index']),
        'frame_index': current_step,
        'past_frame_index': history.past_step,
        'measured_displacement_m': history.displacement_m.tolist(),
        'visualization': 'overlay alpha is 0.65 times cue intensity; cue-only row is the exact prepared cue',
        'previous_render': str(args.previous.resolve()) if args.previous is not None else None,
        'modes': metadata,
    }
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'out': str(args.out), 'sample_sha256': manifest['sample_sha256'],
                      'measured_displacement_m': manifest['measured_displacement_m'],
                      'modes': list(metadata)}))


if __name__ == '__main__':
    main()

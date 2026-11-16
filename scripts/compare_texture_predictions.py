"""Compare two texture-compare-only reports on CPU; never run inference."""
import argparse
import json
from pathlib import Path


def compare(upstream, optimized):
    rows = []
    left, right = upstream.get('texture_trace', {}), optimized.get('texture_trace', {})
    def add(stage, keys, fingerprint):
        a = {key: left.get(key) for key in keys}
        b = {key: right.get(key) for key in keys}
        try:
            fa, fb = fingerprint(a), fingerprint(b)
            complete = fa is not None and fb is not None
        except (KeyError, TypeError):
            complete, fa, fb = False, None, None
        rows.append(dict(stage=stage, status=('match' if fa == fb else 'diverged') if complete else 'missing',
                         upstream=a, optimized=b))
    add('preprocessed_image', ['preprocessed_image'], lambda x: x['preprocessed_image']['sha256'])
    for resolution in (512, 1024):
        key = f'conditioning_{resolution}'
        add(key, [key], lambda x, key=key: [x[key][k]['sha256'] for k in ('cond', 'neg_cond')])
    add('shape_slat', ['shape_slat'], lambda x: [x['shape_slat'][k] for k in ('coords_sha256', 'feats_sha256')])
    # History retention / verbosity affect storage or logging, not sampling math.
    mathematical = ('steps', 'guidance_strength', 'guidance_rescale', 'guidance_interval', 'rescale_t', 'sigma_min')
    add('texture_sampler_settings', ['sampler_params'], lambda x: [x['sampler_params'][k] for k in mathematical])
    for stage, keys, hashes in (
        ('texture_initial_noise', ['noise', 'noise_sha256', 'noise_coords_sha256'], ['noise_sha256', 'noise_coords_sha256']),
        ('normalized_shape_conditioning', ['concat_cond', 'concat_cond_sha256'], ['concat_cond_sha256']),
        ('predicted_normalized_tex_slat', ['predicted_normalized_tex_slat', 'predicted_normalized_tex_slat_sha256'], ['predicted_normalized_tex_slat_sha256']),
        ('decoder_input', ['decoder_input_denormalized_tex_slat', 'decoder_input_sha256', 'decoder_input_coords_sha256'], ['decoder_input_sha256', 'decoder_input_coords_sha256']),
        ('raw_decoder_output', ['raw_decoder_output', 'raw_decoder_output_sha256'], ['raw_decoder_output_sha256']),
    ):
        add(stage, keys, lambda x, hashes=hashes: [x[k] for k in hashes] if all(x[k] is not None for k in hashes) else None)
    missing = [row['stage'] for row in rows if row['status'] == 'missing']
    first = next((row['stage'] for row in rows if row['status'] == 'diverged'), None)
    return dict(schema_version=1, complete=not missing,
                first_observed_divergence=first,
                first_divergence=first if not missing else None,
                missing_stages=missing, stages=rows,
                run_status={'upstream': upstream.get('status'), 'optimized': optimized.get('status')},
                stage_wall_times={label: [{k: stage.get(k) for k in ('stage', 'status', 'time_s')}
                                          for stage in report.get('stage_metrics', [])]
                                  for label, report in [('upstream', upstream), ('optimized', optimized)]},
                interpretation='Exact byte differences locate observed divergence, not its cause or material correctness.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--optimized', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    reports = [json.loads(Path(path).read_text(encoding='utf-8')) for path in (args.upstream, args.optimized)]
    result = compare(*reports)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('complete', 'first_divergence', 'first_observed_divergence', 'missing_stages')}, indent=2))


if __name__ == '__main__':
    main()

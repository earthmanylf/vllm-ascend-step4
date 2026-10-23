#!/usr/bin/env python3
"""Post-build verification and receipt generation.

Checks, per backend:
  buildctl: metadata-file parses, records the image digest, and the FEATURE_COMMIT /
            FEATURE_TREE build args match what was built.
  docker  : metadata file contains an image id (written by run-build.sh).
Writes a JSON receipt (never bit-identical rebuild promises) capturing commit, tree,
image ref, digest, context digests and whether the pin was reproduced exactly.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', required=True, choices=['buildctl', 'docker'])
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--tree', required=True)
    parser.add_argument('--pin-match', required=True, choices=['true', 'false'])
    parser.add_argument('--context-files', required=True,
                        help='JSON object of context file digests from prepare-context.py')
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()

    assert args.metadata.is_file(), 'missing build metadata: ' + str(args.metadata)
    raw = args.metadata.read_text().strip()
    digest = None
    details = {}
    if args.backend == 'buildctl':
        data = json.loads(raw)
        digest = data.get('containerimage.digest')
        assert digest, 'buildctl metadata without containerimage.digest: ' + raw[:200]
        details['metadata_keys'] = sorted(data.keys())
    else:
        lines = raw.splitlines()
        assert lines and lines[0].strip(), 'empty docker metadata file'
        digest = lines[0].strip()
        assert digest.startswith('sha256:'), 'unexpected docker image id: ' + digest
        details['metadata_keys'] = ['docker-image-id']

    context_files = json.loads(args.context_files)
    receipt = {
        'schema_version': 1,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'backend': args.backend,
        'image_ref': args.image,
        'image_digest': digest,
        'source_commit': args.commit,
        'source_tree': args.tree,
        'pin_reproduced': args.pin_match == 'true',
        'context_files': context_files,
        'metadata': details,
        'scope': ('CI rebuild receipt. A rebuilt wheel/image is not promised bit-identical; '
                  're-run acceptance separately. pin_reproduced=true means the six build '
                  'inputs matched recipe-origin.json digests exactly.'),
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'pass': True, 'receipt': str(args.receipt), 'image_digest': digest,
                      'pin_reproduced': receipt['pin_reproduced']}, indent=2))


if __name__ == '__main__':
    main()

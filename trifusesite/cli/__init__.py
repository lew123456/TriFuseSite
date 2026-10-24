import argparse
import sys
from importlib import import_module


COMMANDS = {
    'prepare-datasets': ('prepare_datasets', 'Prepare benchmark inputs'),
    'prepare': ('prepare', 'Prepare a custom input manifest'),
    'fetch-structures': ('fetch_structures', 'Download protein structures'),
    'extract-esm': ('extract_esm', 'Extract ESM-2 embeddings'),
    'validate-datasets': ('validate_datasets', 'Validate benchmark files'),
    'train': ('train', 'Train a task model'),
    'predict': ('predict', 'Predict residue binding sites'),
    'evaluate': ('evaluate', 'Evaluate a trained checkpoint'),
    'cross-validate': ('cross_validate', 'Run PPI cross-validation'),
    'repeat': ('repeat', 'Run repeated training and evaluation'),
    'paired-analysis': ('paired_analysis', 'Analyze paired structures'),
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='trifusesite')
    commands = parser.add_subparsers(dest='command', metavar='command')
    for command, (_, description) in COMMANDS.items():
        commands.add_parser(command, help=description, add_help=False)
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        parser.print_help()
        return
    selected = parser.parse_args(arguments[:1])
    module = import_module('trifusesite.cli.' + COMMANDS[selected.command][0])
    return module.main(arguments[1:])

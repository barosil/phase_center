# -*- coding: utf-8 -*-
"""Load LIT datasets for horn and create nice datasets for importing in PANDAS dataframes.

LIT files have angles, intensity and phases for each measumentents. These data goes into a single csv for each measurement.

Example:
    make_dataset.py --file "../../data/external/UIRAPURU.xlsx" --output "../../data/raw/"
"""
__version__ = 1.0

import argparse
import sys
import pandas as pd


def make_dataset(args):
    """Load LIT datasets for horn and create nice datasets for importing in PANDAS dataframes with tidy format.

    Args:
        args (dict): argparse object or dictionary. Parameters are `--file` and `--output`. `args`.

    Returns:
        type: None.

    """
    name = args.file.split('/')[-1].split('.')[0].replace(' ','_')
    output_path = args.output
    try:
        beams_raw = pd.read_excel(args.file, sheet_name=None)
        medidas = list(beams_raw.keys())
        for dataset in medidas:
            filename =  output_path + name + '_' + dataset.replace(' ','_')  + '.csv'
            freqs = list(map(lambda val: float(val.replace(',', '.').replace('GHz', '')), beams_raw[dataset].columns[1::2]))
            phases = beams_raw[dataset].iloc[1:,2::2]
            amplitudes = beams_raw[dataset].iloc[1:,1::2]
            phases.columns = freqs
            amplitudes.columns = freqs
            phases['ANGLE'] = beams_raw[dataset].iloc[1:,0]
            amplitudes['ANGLE'] = beams_raw[dataset].iloc[1:,0]
            df_phase = pd.melt(phases, id_vars=['ANGLE'], value_vars=freqs, var_name='FREQ', value_name='PHASE')
            df_amplitude = pd.melt(amplitudes, id_vars=['ANGLE'], value_vars=freqs, var_name='FREQ', value_name='AMPLITUDE')
            df = df_phase.merge(df_amplitude, on=['ANGLE', 'FREQ'])
            try:
                df.to_csv(filename, index=False, encoding='utf8', mode = 'x')
            except FileExistsError as e:
                print('File already exists. Terminating')
                return None
    except FileNotFoundError as e:
        print('Arquivo não encontrado. Verifique o caminho.' + str(e))
        sys.exit(1)
    return None


def main():
    """Executa função para criação de datasets. Apenas na CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", required=True, help="Caminho para dados do LIT")
    parser.add_argument("--output", "-o", required=False, help="Caminho    para gravar dados", default='../data/raw/')
    args = parser.parse_args()
    make_dataset(args)
    return None


if __name__ == "__main__":
    main()

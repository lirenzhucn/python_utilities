from math import log10
import os
from imageproc.polymask import polymask
import readline
import glob
import click
import numpy as np


def readline_input(prompt, prefill=''):
    """work just like input() but with optional prefill"""
    readline.set_startup_hook(lambda: readline.insert_text(prefill))
    try:
        return input(prompt)
    finally:
        readline.set_startup_hook()


def prompt_choose(choices, default=None, return_val=False):
    n = len(choices)
    numDigits = int(log10(n)) + 1
    template = '%%%dd. %%s' % numDigits
    for ind, item in enumerate(choices):
        click.echo(template % (ind + 1, item))
    val = click.prompt('Please choose one', type=click.IntRange(1, n),
                       default=default)
    if return_val:
        return choices[val-1]
    else:
        return val


def read_datacube(input_file):
    """Read data cube file based on its file type"""
    _, ext = os.path.splitext(input_file)
    if ext == '.mat':
        import hdf5storage as h5
        matDict = h5.loadmat(input_file)
        keys = [k for k in list(matDict.keys()) if not k.startswith('_')]
        if len(keys) > 1:
            choice = prompt_choose(['{:}: shape={:}, dtype={:}'.format(
                k, matDict[k].shape, matDict[k].dtype)
                                for k in keys])
        else:
            choice = 1
        return matDict[keys[choice-1]]
    elif ext == '.npy':
        return np.load(input_file)
    elif ext == '.tif' or ext == '.tiff':
        import tifffile
        x = np.dstack(tifffile.imread(input_file))
        return x
    else:
        raise ValueError('unrecognized file type %s' % ext[1:])


def make_cache_folder(input_file):
    folder, filename = os.path.split(os.path.abspath(input_file))
    cacheFolder = os.path.join(folder, '.' + filename)
    if not os.path.exists(cacheFolder):
        os.mkdir(cacheFolder)
    return cacheFolder


def load_mask(cacheFolder, filename, name, cube, outFile=None, axis=0):
    try:
        # read mask file, assumed to be a numpy file.
        filepath = os.path.join(cacheFolder, filename)
        mask = np.load(filepath)
        redraw = click.confirm('Mask loaded, do you want to redraw',
                               prompt_suffix='? ')
        if not redraw:
            # if the mask was successfully loaded from the file, we don't need
            # to save it again.
            return mask
    except IOError:
        pass
    click.echo('== Acquire %s' % name)
    mask = polymask(np.mean(cube, axis=axis), color_map='jet')
    saveMask = click.confirm('Save %s' % name, prompt_suffix='?', default=True)
    if saveMask:
        if outFile is None:
            outFile = filename
        np.save(os.path.join(cacheFolder, outFile), mask)
    return mask


def complete_file_path(text, state):
    return (glob.glob(text + '*') + [None])[state]


def prepare_output_filepath(path):
    outpath = os.path.splitext(path)[0] + '.tif'
    outdir, filename = os.path.split(outpath)
    if filename.startswith('cube'):
        filename = 'im' + filename[4:]
    outpath = os.path.join(outdir, filename)
    return outpath


def prompt_file_path(msg, prefill=''):
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind('tab: complete')
    readline.set_completer(complete_file_path)
    out = readline_input(msg, prefill=prefill)
    readline.set_completer()
    return out

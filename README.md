# Utility modules used by multiple scripts or spikes
This repository is supposed to be very fluid. Many modules would start from here
and might be moved to a dedicated package or project.

## What should be included
Modules that are frequently used during interactive sessions, including IPython
Console and Notebook, are the main targets of this repository. Modules shared by
multiple local scripts are also suitable here, until maybe when these scripts
become part of a standalone project, at which point, these modules should be
incorporated into that project.

## Current tool set

- imageproc
- sigproc
- exputil
- MAPGG.py
- TwIST.py
- debugy.py

## `imageproc`
This sub-package contains utilities for image visualization, editing,
input/output, and processing.

## `sigproc`
This sub-package hosts utility modules used for mainly 1D signal processing.

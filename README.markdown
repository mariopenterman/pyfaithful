# pyfaithful
***A byte- and line-faithful Python decompiler***

pyfaithful translates compiled Python byte-code back into valid, human-readable
Python source, and supports byte-code from any version of Python. Its focus is
**faithful reconstruction**: for Python 3.11 in particular it aims for decompiled
source that recompiles to the *same* byte-code (`co_code`) and, as far as the
`.pyc` allows, the *same* source positions (`co_positions`) as the original.

pyfaithful includes both a byte-code disassembler (`pycdas`) and a
decompiler (`pycdc`), and is written in C++.

## Building pyfaithful
* Generate a project or makefile with [CMake](http://www.cmake.org) (See CMake's documentation for details)
  * The following options can be passed to CMake to control debug features:

    | Option | Description |
    | --- | --- |
    | `-DCMAKE_BUILD_TYPE=Debug` | Produce debugging symbols |
    | `-DENABLE_BLOCK_DEBUG=ON` | Enable block debugging output |
    | `-DENABLE_STACK_DEBUG=ON` | Enable stack debugging output |

* Build the generated project or makefile
  * For projects (e.g. MSVC), open the generated project file and build it
  * For makefiles, just run `make`
  * To run tests (on \*nix or MSYS), run `make check JOBS=4` (optional
    `FILTER=xxxx` to run only certain tests)

## Usage
**To run pycdas**, the PYC Disassembler:
`./pycdas [PATH TO PYC FILE]`
The byte-code disassembly is printed to stdout.

**To run pycdc**, the PYC Decompiler:
`./pycdc [PATH TO PYC FILE]`
The decompiled Python source is printed to stdout.
Any errors are printed to stderr.

**Marshalled code objects**:
Both tools support Python marshalled code objects, as output from `marshal.dumps(compile(...))`.

To use this feature, specify `-c -v <version>` on the command line - the version must be specified as the objects themselves do not contain version metadata.

## Authors, License, Credits
pyfaithful is an independent fork of **Decompyle++**, which is the work of
**Michael Hansen and Darryl Pogue**, with additional upstream contributions from
charlietang98, Kunal Parmar, Olivier Iffrig, and Zlodiy. The original project
lives at https://github.com/zrax/pycdc.

Fork modifications (2026) are Copyright (C) Mario Penterman.

pyfaithful is released under the terms of the **GNU General Public License,
version 3** — the same license as the original — and it remains GPLv3. See the
`LICENSE` file for the full text. This fork is not endorsed by or affiliated with the original
authors.

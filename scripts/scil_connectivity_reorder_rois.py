#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Re-order one or many connectivity matrices' rows and columns. The connectivity
matrices can come, for instance, from scil_connectivity_computes_matrices.py

To state the new order, use a text file with the following formatting:
The first row are the new x order and the second row the new y order, using
integer values separated by a space. The resulting matrix does not have to be
square (supports unequal number of x and y).

The values refer to the coordinates (starting at 0) in the matrix, but if the
--labels_list parameter is used, the values will refer to the label which will
be converted to the appropriate coordinates. This file must be the same as the
one provided to the scil_tractogram_segment_connections_from_labels.py.

To subsequently use scil_viz_connectivity.py with a lookup table, you
must use a label-based reording json and use --labels_list.

You can also use the Optimal Leaf Ordering (OLO) algorithm to transform a
sparse matrix into an ordering that reduces the matrix bandwidth. The output
file can then be re-used with --in_ordering. Only one input can be used with
this option, we recommand an average streamline count or volume matrix.

Formerly: scil_reorder_connectivity.py
-----------------------------------------------------------------------------
Reference:
[1] Rubinov, Mikail, and Olaf Sporns. "Complex network measures of brain
    connectivity: uses and interpretations." Neuroimage 52.3 (2010):
    1059-1069.
-----------------------------------------------------------------------------
"""

import argparse
import logging
import os

import numpy as np

from scilpy.connectivity.matrix_tools import (compute_olo,
                                              apply_reordering)
from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             load_matrix_in_any_format,
                             save_matrix_in_any_format,
                             assert_outputs_exist,
                             add_verbose_arg,
                             assert_output_dirs_exist_and_empty)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_matrices', nargs='+',
                   help='Connectivity matrices in .npy or .txt format.')
    ord = p.add_mutually_exclusive_group(required=True)
    ord.add_argument('--in_ordering',
                     help='Txt file with the first row as x and second as y.')
    ord.add_argument('--optimal_leaf_ordering', metavar='OUT_FILE',
                     help='Output a text file with an ordering that aligns'
                          'structures along the diagonal.')

    p.add_argument('--out_suffix',
                   help="Suffix for the output matrices filenames. It will "
                        "be appended to each input matrix's name.")
    p.add_argument('--out_dir',
                   help='Output directory for the re-ordered matrices.\n'
                        'If not set, each output matrix will be saved in '
                        'the same \ndirectory as the input matrix.')
    p.add_argument('--labels_list',
                   help='List saved by the decomposition script,\n'
                        '--in_ordering must contain labels rather than '
                        'coordinates (.txt).')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def parse_ordering(in_ordering_file, labels_list=None):
    """
    Read the ordering file, which should contain two lines, with integer
    values for each.
    """
    # Cannot load with numpy in case of non-squared matrix (unequal x/y)
    with open(in_ordering_file, 'r') as my_file:
        lines = my_file.readlines()
        if len(lines) != 2:
            raise ValueError("The ordering file should contain exactly two "
                             "lines of text.")
        ordering = [[int(val) for val in lines[0].split()],
                    [int(val) for val in lines[1].split()]]
    if labels_list:
        labels_list = np.loadtxt(labels_list,
                                 dtype=np.int16).tolist()
        # If the reordering file refers to labels and not indices
        real_ordering = [[], []]
        real_ordering[0] = [labels_list.index(i) for i in ordering[0]]
        real_ordering[1] = [labels_list.index(i) for i in ordering[1]]
        return real_ordering

    return ordering


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_matrices,
                        [args.labels_list, args.in_ordering])
    assert_output_dirs_exist_and_empty(parser, args, [], args.out_dir)
    # Verification of output matrices names will be done below.

    if args.optimal_leaf_ordering is not None:
        if args.out_suffix or args.out_dir:
            logging.warning("Options --out_suffix and --out_dir are ignored "
                            "with option --optimal_leaf_ordering.")
        if len(args.in_matrices) > 1:
            parser.error('Only one input is supported with RCM.')
        assert_outputs_exist(parser, args, args.optimal_leaf_ordering)

        matrix = load_matrix_in_any_format(args.in_matrices[0])
        perm = compute_olo(matrix).astype(np.uint16)
        np.savetxt(args.optimal_leaf_ordering, [perm.tolist(), perm.tolist()],
                   fmt='%i')
    else:
        # Verify all the possible outputs to avoid overwriting files
        if args.out_suffix is None:
            args.out_suffix = ""

        out_filenames = []
        for filename in args.in_matrices:
            out_dir = os.path.dirname(filename) if args.out_dir is None \
                else args.out_dir
            basename, ext = os.path.splitext(filename)
            basename = os.path.basename(basename)

            curr_filename = os.path.join(
                out_dir, '{}{}.{}'.format(basename, args.out_suffix, ext[1:]))
            out_filenames.append(curr_filename)
        assert_outputs_exist(parser, args, out_filenames)

        ordering = parse_ordering(args.in_ordering, args.labels_list)

        for in_name, out_name in zip(args.in_matrices, out_filenames):
            matrix = load_matrix_in_any_format(in_name)
            reordered_matrix = apply_reordering(matrix, ordering)
            save_matrix_in_any_format(out_name, reordered_matrix)


if __name__ == "__main__":
    main()

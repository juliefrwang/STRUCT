import argparse

def argument_parser_target():
    parser = argparse.ArgumentParser(description="SPLASH-structure: a statistical approach to identify "
                                                 "RNA secondary structures from raw sequencing data, "
                                                 "bypassing multiple sequence alignment.")

    # Required arguments
    parser.add_argument("output_prefix", help="Prefix for naming the output result folder.")
    parser.add_argument("splash_output_file", help="Path to the SPLASH output file.")


    # Options
    parser.add_argument("-a", "--element_annotation", action="store_true",
                        help="Enable element annotation on targets.", )
    parser.add_argument("--noncanon", type=str, default=None,
                        help="Non-canonical pair set for the extended "
                             "(SPC+BPC) test. Comma-separated two-base "
                             "tokens, e.g. 'GU' or 'GU,GA'; U is read as "
                             "T; both orientations are always included. "
                             "'none' (or omitting this flag) runs the "
                             "legacy WCF-only path, byte-identical to "
                             "prior versions. This single flag both gates "
                             "and parameterizes the extension.")
    parser.add_argument("--wobble", action="store_true",
                        help="DEPRECATED alias for '--noncanon GU'. Kept "
                             "for one release; prefer --noncanon. Ignored "
                             "if --noncanon is given explicitly.")
    parser.add_argument("--titv", type=float, default=0.5,
                        help="Aggregate Ti/Tv event ratio (= #Ti / #Tv) "
                             "assumed by the SVP/BPC null. Default 0.5 "
                             "reproduces the uniform identity null of the "
                             "original published model. Biological data "
                             "typically sits near 2.0; pass --titv 2 to "
                             "match such samples. Only consumed when the "
                             "extension is active (--noncanon / --wobble).")

    arguments = parser.parse_args()

    return vars(arguments)

def argument_parser_compactor():
    parser = argparse.ArgumentParser(description="SPLASH-structure: a statistical approach to identify "
                                                 "RNA secondary structures from raw sequencing data, "
                                                 "bypassing multiple sequence alignment.")

    # Required arguments
    parser.add_argument("output_prefix", help="Prefix for naming the output result folder.")
    parser.add_argument("compactor_file", help="Path to the compactor file.")


    # Options
    parser.add_argument("-a", "--element_annotation", action="store_true",
                        help="Enable element annotation on compactors.", )
    parser.add_argument("--noncanon", type=str, default=None,
                        help="Non-canonical pair set for the extended "
                             "(SPC+BPC) test. Comma-separated two-base "
                             "tokens, e.g. 'GU' or 'GU,GA'; U is read as "
                             "T; both orientations are always included. "
                             "'none' (or omitting this flag) runs the "
                             "legacy WCF-only path, byte-identical to "
                             "prior versions. This single flag both gates "
                             "and parameterizes the extension.")
    parser.add_argument("--wobble", action="store_true",
                        help="DEPRECATED alias for '--noncanon GU'. Kept "
                             "for one release; prefer --noncanon. Ignored "
                             "if --noncanon is given explicitly.")
    parser.add_argument("--titv", type=float, default=0.5,
                        help="Aggregate Ti/Tv event ratio (= #Ti / #Tv) "
                             "assumed by the SVP/BPC null. Default 0.5 "
                             "reproduces the uniform identity null of the "
                             "original published model. Biological data "
                             "typically sits near 2.0; pass --titv 2 to "
                             "match such samples. Only consumed when the "
                             "extension is active (--noncanon / --wobble).")

    arguments = parser.parse_args()

    return vars(arguments)
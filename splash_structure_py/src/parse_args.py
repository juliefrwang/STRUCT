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
    parser.add_argument("--wobble", action="store_true",
                        help="Enable the non-WCF (G·U wobble) extension. "
                             "Uses wobble-aware stem detection and the SVP "
                             "p-value combining single-position-compatible "
                             "(SPC) and base-pair-covariation (BPC) events. "
                             "Default off; the original WCF-only path is "
                             "byte-identical to prior versions when this "
                             "flag is omitted.")

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
 
    arguments = parser.parse_args()

    return vars(arguments)
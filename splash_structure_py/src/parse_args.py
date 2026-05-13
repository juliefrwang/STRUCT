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
    parser.add_argument("--titv", type=float, default=0.5,
                        help="Aggregate Ti/Tv event ratio (= #Ti / #Tv) "
                             "assumed by the SVP/BPC null. Default 0.5 "
                             "reproduces the uniform identity null of the "
                             "original published model. Biological data "
                             "typically sits near 2.0; pass --titv 2 to "
                             "match such samples. Only consumed when "
                             "--wobble is set.")

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
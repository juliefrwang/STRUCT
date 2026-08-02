# The file takes in two argements: 
# 1. Compactor output from SPLASH 
# 2. Directory to save output

using Combinatorics
using DataFrames
using CSV

# trim string 
function trim_compactor(input::AbstractString, anchor_len::Int)
    return input[anchor_len+1:end]
end

# precompute parsing indices
function precompute_idx(input::AbstractString, seg_num::Int)
    len = length(input)
    base_len, extra = divrem(len, seg_num)
    seg_lengths = fill(base_len, seg_num)
    seg_lengths[1:extra] .+= 1  # Distribute the extra characters to the first segments

    # Compute start and end indices
    indices = cumsum([1; seg_lengths])
    # segments = [input[indices[i]:indices[i+1]-1] for i in 1:seg_num]
    return indices
end

# parse string
function parse_string(input::AbstractString, seg_num::Int, indices)
    # len = length(input)
    # base_len, extra = divrem(len, seg_num)
    # seg_lengths = fill(base_len, seg_num)
    # seg_lengths[1:extra] .+= 1  # Distribute the extra characters to the first segments

    # # Compute start and end indices
    # indices = cumsum([1; seg_lengths])
    segments = [input[indices[i]:indices[i+1]-1] for i in 1:seg_num]
    return segments
end

# combine segments based on index_comb
function string_combinations(input::AbstractString, anchor_len::Int, index_comb::Vector{Vector{Int}}, seg_num::Int)
    trimmed_input = trim_compactor(input, anchor_len)
    indices = precompute_idx(trimmed_input, seg_num)
    strings = parse_string(trimmed_input, seg_num, indices)
    scrambled_list = []
    for comb in index_comb
        push!(scrambled_list, (strings[comb[1]] * strings[comb[2]], strings[comb[3]] * strings[comb[4]]))
    end
    return scrambled_list
end

# get the rank of each segment
function sortperm_rev(vector)
    return sortperm(vector, rev=true)
end

# get base target
function get_base_target(seg_vec, rank_vec)
    return seg_vec[findmin(rank_vec)[2]]
end

# get the weight of target
function get_wgt(support_vec)
    return support_vec ./ sum(support_vec)
end

function main()
    # read in the compactor file
    df =  CSV.read(ARGS[1], DataFrame, delim='\t', header=true)

    # get anchor length
    anchor_len = length(df[1,1])

    # Add compactor rank (sorted by support descendingly) to a new column. 
    transform!(groupby(df, [:anchor,]), [:support => sortperm_rev => :support_rank])

    # filter out anchors whose 2nd most abundant compactor has support < 2
    anchors_to_remove = unique(df[[(row.support_rank == 2 && row.support < 2) for row in eachrow(df)], :anchor])
    df = filter(row -> !(row.anchor in anchors_to_remove), df)

    # generate scrambled compactor segments, indexed as <Destruction><No.><Combination><No.> and stack them into a new column 'segment
    index_comb = [[1, 2, 3, 4], 
                  [1, 3, 2, 4], 
                  [1, 4, 2, 3]] 
    seg_num = 4

    transform!(df, :, :compactor => ByRow(x -> string_combinations(x, anchor_len, index_comb, seg_num)) => [:D1, :D2, :D3])
    pivoted_df = stack(df, r"D")
    rename!(pivoted_df, Dict(:variable => :segment_index, :value => :segment))

    # get target weight 
    transform!(groupby(pivoted_df, [:anchor, :segment_index]), [:support => get_wgt => :compactor_weight])
    
    # filter compactor abundance >.05
    filter!(pivoted_df -> pivoted_df.compactor_weight .>= 0.05, pivoted_df)

    # add 'base_target' column and filter rows to remove those where 'base_target' is equal to 'segment'.
    transform!(groupby(pivoted_df, [:anchor, :segment_index]), [[:segment, :support_rank] => get_base_target => :base_target])
    filter!(pivoted_df -> pivoted_df.base_target != pivoted_df.segment, pivoted_df)
    
    # Recalculate 'compactor_weight' after filtering, grouping by 'anchor' and 'segment_index'.
    transform!(groupby(pivoted_df, [:anchor, :segment_index]), [:support => get_wgt => :compactor_weight])
        
    # place base_target tuple and segment tuple to seperate columns
    transform!(pivoted_df, :segment => ByRow(x -> (x[1], x[2])) => [:S1, :S2])
    transform!(pivoted_df, :base_target => ByRow(x -> (x[1], x[2])) => [:base_S1, :base_S2])
    select!(pivoted_df, Not([:segment, :base_target]))
    
    # write tsv 
    out_file = string(ARGS[2], "/", "processed_compactors.tsv")
    CSV.write(out_file, pivoted_df, delim='\t')
end


main()



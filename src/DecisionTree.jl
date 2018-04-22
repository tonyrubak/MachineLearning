module DecisionTree
using DataFramesMeta

export decision_tree_create
abstract type TreeNode{T} end

struct LeafNode{T} <: TreeNode{T}
    value::T
end

struct BranchNode{T} <: TreeNode{T}
    feature::Symbol
    left_branch::TreeNode{T}
    right_branch::TreeNode{T}
end

function node_mistakes(labels)
    retval = 0
    if length(labels) == 0
        retval = 0
    else
        num_ones = size(labels[labels .== 1,:])[1]
        num_nones = size(labels[labels .== -1,:])[1]
        retval = (num_ones ≥ num_nones) ? num_nones : num_ones
    end
    retval
end

function choose_splitting_feature(data, features, target)
    best_feature = :none
    best_error = 10.

    n = size(data)[1]

    for feature in features
        left_split = data[data[feature] .== 0,:]
        right_split = data[data[feature] .== 1,:]

        left_mistakes = node_mistakes(left_split[:,target])
        right_mistakes = node_mistakes(right_split[:,target])

        error = (left_mistakes + right_mistakes) / n

        if error < best_error
            best_feature = feature
            best_error = error
        end
    end
    best_feature
end

function create_leaf(values)
    pred = 0
    num_ones = length(values[values .== 1,:])
    num_nones = length(values[values .== -1,:])

    if num_ones > num_nones
        pred = 1
    else
        pred = -1
    end
    LeafNode(pred)
end

function decision_tree_create(data, features, target, current_depth = 0, max_depth = 10)
    remaining_features = copy(features)
    target_values = data[:, target]

    println("--------------------------------------------------------------------")
    println("Subtree, depth = $current_depth ($(length(target_values)) data points).")

    if node_mistakes(data[:, target]) == 0
        println("Stopping condition 1 reached.")
        return create_leaf(target_values)
    elseif remaining_features == []
        println("Stopping condition 2 reached.")
        return create_leaf(target_values)
    elseif current_depth ≥ max_depth
        println("Reached maximum depth. Stopping for now.")
        return create_leaf(target_values)
    end
    
    splitting_feature = choose_splitting_feature(data, features, target)
    left_split = data[data[splitting_feature] .== 0,:]
    right_split = data[data[splitting_feature] .== 1,:]
    delete!(remaining_features, splitting_feature)
    println("Splitting on $splitting_feature")
    if size(left_split)[1] == size(data)[1]
        println("Creating leaf node")
        return create_leaf(left_split[:, target])
    elseif size(right_split)[1] == size(data)[1]
        println("Creating leaf node")
        return create_leaf(right_split[:, target])
    end
    
    left_tree = decision_tree_create(left_split, remaining_features,
                                     target,
                                     current_depth + 1,
                                     max_depth)
    right_tree = decision_tree_create(right_split, remaining_features,
                                      target,
                                      current_depth + 1,
                                      max_depth)
    return BranchNode(splitting_feature, left_tree, right_tree)
end

# @testset "Node Mistakes Tests" begin
    # @test node_mistakes([-1, -1, 1, 1, 1]) == 2
    # @test node_mistakes([-1, -1, 1, 1, 1, 1]) == 2
    # @test node_mistakes([-1, -1, -1, -1, -1, 1, 1]) == 2
# end
end

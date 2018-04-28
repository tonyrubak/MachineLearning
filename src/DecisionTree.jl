module DecisionTree
using DataFrames
using DataFramesMeta

export decision_tree_create, classify, complexity, evaluate

abstract type TreeNode{T} end

struct LeafNode{T} <: TreeNode{T}
    value::T
end

struct BranchNode{T} <: TreeNode{T}
    feature::Symbol
    left::TreeNode{T}
    right::TreeNode{T}
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

function information(labels)
    retval = 0.
    n = size(labels)[1]
    if n ≠ 0
        num_ones = size(labels[labels .== 1,:])[1]
        num_nones = size(labels[labels .== -1,:])[1]

        p_ones = num_ones / n
        p_nones = num_nones / n
        
        retval = -(p_ones * log(2, p_ones) + p_nones * log(2, p_nones))
    end
    retval
end

function choose_splitting_feature(data, features, target, ::Type{Val{:Error}})
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

function choose_splitting_feature(data, features, target, ::Type{Val{:IG}})
    best_feature = :none
    best_ig = 0

    n = size(data)[1]

    i₀ = information(data[:,target])

    for feature in features
        left_split = data[data[feature] .== 0,:]
        right_split = data[data[feature] .== 1,:]

        n_l = size(left_split)[1]
        n_r = size(right_split)[1]

        left_information = information(left_split[:,target])
        right_information = information(right_split[:,target])

        ig = i₀ - n_l/n * left_information + n_r/n * right_information

        if ig > best_ig
            best_feature = feature
            best_ig = ig
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

function minimum_node_size(data, min_node_size)
    if size(data)[1] < min_node_size
        true
    else
        false
    end
end

function error_reduction(error_before_split, error_after_split)
    error_before_split - error_after_split
end

function decision_tree_create(data, features, target, current_depth = 0,
                              max_depth = 10, min_node_size = 1,
                              min_error_reduction = 0.0,  method = :IG)
    remaining_features = copy(features)
    target_values = data[:, target]
    n = size(data)[1]

    println("--------------------------------------------------------------------")
    println("Subtree, depth = $current_depth ($(length(target_values)) data points).")

    # Check for the two base stopping conditions
    # 1. All data are one class
    # 2. There are no more features to split on
    # and the first two early stopping conditions
    # 1. The tree has reached the maximum depth
    # 2. The node doesn't contain sufficient data points
    if node_mistakes(data[:, target]) == 0
        println("Stopping condition 1 reached.")
        return create_leaf(target_values)
    elseif remaining_features == []
        println("Stopping condition 2 reached.")
        return create_leaf(target_values)
    elseif current_depth ≥ max_depth
        println("Reached maximum depth. Stopping for now.")
        return create_leaf(target_values)
    elseif minimum_node_size(data, min_node_size) == true
        println("Reached minimum node size.")
        return create_leaf(target_values)
    end
    
    splitting_feature = choose_splitting_feature(data, features, target, Val{method})

    # Check the third early stopping condition
    # 3. No split improves the metric
    if method == :IG && splitting_feature == :none
        println("No information gain. Creating leaf node.")
        return create_leaf(target_values)
    end
    
    left_split = data[data[splitting_feature] .== 0,:]
    right_split = data[data[splitting_feature] .== 1,:]

    # Check the third early stopping condition
    # 3. Minimum error reduction
    error_before_split = node_mistakes(target_values) / n
    left_mistakes = node_mistakes(left_split[:,target])
    right_mistakes = node_mistakes(right_split[:,target])
    error_after_split = (left_mistakes + right_mistakes) / n
    if error_reduction(error_before_split, error_after_split) < min_error_reduction
        println("Minimum error reduction reached.")
        return create_leaf(target_values)
    end
    
    delete!(remaining_features, splitting_feature)
    println("Splitting on $splitting_feature")
    if size(left_split)[1] == n
        println("Creating leaf node")
        return create_leaf(left_split[:, target])
    elseif size(right_split)[1] == n
        println("Creating leaf node")
        return create_leaf(right_split[:, target])
    end
    
    left_tree = decision_tree_create(left_split, remaining_features,
                                     target,
                                     current_depth + 1,
                                     max_depth, method)
    right_tree = decision_tree_create(right_split, remaining_features,
                                      target,
                                      current_depth + 1,
                                      max_depth, method)
    return BranchNode(splitting_feature, left_tree, right_tree)
end

function classify(tree::LeafNode, x, annotate = false)
    if annotate
        println("At leaf, predicting $(tree.value)")
    end
    return tree.value
end

function classify(tree::BranchNode, x, annotate = false)
    split_value::Int64 = x[tree.feature][1]
    if annotate
        println("Split on $(tree.feature) = $split_value")
    end
    if split_value == 0
        return classify(tree.left, x, annotate)
    else
        return classify(tree.right, x, annotate)
    end
end

function evaluate{T}(tree::TreeNode{T}, data, target)
    n = size(data)[1]
    preds = map(x -> classify(tree, x), eachrow(data))
    error = preds .* data[:,target]
    size(error[error .< 0])[1] / n
end

# @testset "Node Mistakes Tests" begin
    # @test node_mistakes([-1, -1, 1, 1, 1]) == 2
    # @test node_mistakes([-1, -1, 1, 1, 1, 1]) == 2
    # @test node_mistakes([-1, -1, -1, -1, -1, 1, 1]) == 2
# end
end

module DecisionTree
using DataFrames
using DataFramesMeta

export weighted_decision_tree_create, decision_tree_create, classify, complexity, evaluate, adaboost

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
                                     max_depth, min_node_size,
                                     min_error_reduction, method)
    right_tree = decision_tree_create(right_split, remaining_features,
                                      target,
                                      current_depth + 1,
                                      max_depth, min_node_size,
                                      min_error_reduction, method)
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

function evaluate(tree::TreeNode, data, target)
    n = size(data)[1]
    preds = map(x -> classify(tree, x), eachrow(data))
    error = preds .* data[:,target]
    size(error[error .< 0])[1] / n
end

function complexity(tree::BranchNode)
    complexity(tree.left) + complexity(tree.right)
end

function complexity(tree::LeafNode)
    1
end

# @testset "Node Mistakes Tests" begin
    # @test node_mistakes([-1, -1, 1, 1, 1]) == 2
    # @test node_mistakes([-1, -1, 1, 1, 1, 1]) == 2
    # @test node_mistakes([-1, -1, -1, -1, -1, 1, 1]) == 2
# end

### Weighted Decision Tree

function node_mistakes_weighted(labels, weights)
    wt_positive = any(labels .== 1) ? sum(weights[labels .== 1]) : 0
    wt_mistakes_neg = wt_positive
    wt_negative = any(labels .== -1) ? sum(weights[labels .== -1]) : 0
    wt_mistakes_pos = wt_negative


    if wt_mistakes_pos <= wt_mistakes_neg
        (wt_mistakes_pos, 1)
    else
        (wt_mistakes_neg, -1)
    end
end

function weighted_splitting_feature(data, features, target, weights)
    best_feature = :none
    best_error = Inf
    n = float(size(data)[1])

    for feature in features
        left_split = data[data[feature] .== 0,:]
        right_split = data[data[feature] .== 1,:]
        left_wts = weights[data[feature] .== 0,:]
        right_wts = weights[data[feature] .== 1,:]
        error = (node_mistakes_weighted(left_split, left_wts)[1] +
            node_mistakes_weighted(right_split, right_wts)[1]) /
            sum(weights)
        if error < best_error
            best_feature = feature
            best_error = error
        end
    end
    best_feature 
end

function create_leaf(values, weights)
    weigted_error, best_class = node_mistakes_weighted(values, weights)
    LeafNode(best_class)
end

function weighted_decision_tree_create(data, features, target, weights,
                              current_depth = 0, max_depth = 10)
    remaining_features = copy(features)
    target_values = data[:, target]
    n = size(data)[1]

    println("--------------------------------------------------------------------")
    println("Subtree, depth = $current_depth ($(length(target_values)) data points).")

    # Check for the two base stopping conditions
    # 1. Error is 0
    # 2. There are no more features to split on
    # and the early stopping conditions
    # 1. The tree has reached the maximum depth
    if node_mistakes_weighted(data[:, target], weights)[1] <= 1e-15
        println("Stopping condition 1 reached.")
        return create_leaf(target_values, weights)
    elseif remaining_features == []
        println("Stopping condition 2 reached.")
        return create_leaf(target_values, weights)
    elseif current_depth ≥ max_depth
        println("Reached maximum depth. Stopping for now.")
        return create_leaf(target_values, weights)
    end

    splitting_feature = weighted_splitting_feature(data, features,
                                                   target, weights)
    delete!(remaining_features, splitting_feature)
    
    left_split = data[data[splitting_feature] .== 0,:]
    left_weights = weights[data[splitting_feature] .== 0,:]
    right_split = data[data[splitting_feature] .== 1,:]
    right_weights = weights[data[splitting_feature] .== 1,:]

    println("Splitting on $splitting_feature")
    if size(left_split)[1] == n
        println("Creating leaf node")
        return create_leaf(left_split[:, target], weights)
    elseif size(right_split)[1] == n
        println("Creating leaf node")
        return create_leaf(right_split[:, target], weights)
    end
    
    left_tree = weighted_decision_tree_create(left_split,
                                              remaining_features,
                                              target, left_weights,
                                              current_depth + 1,
                                              max_depth)
    right_tree = weighted_decision_tree_create(right_split,
                                               remaining_features,
                                               target, right_weights,
                                               current_depth + 1,
                                               max_depth)
    return BranchNode(splitting_feature, left_tree, right_tree)
end

## Boosting
function adaboost(data, features, target, num_stumps)
    alpha = repeat([1.], inner = size(data)[1])
    weights = []
    tree_stumps = []
    values = data[:,target]

    for t in 1:num_stumps
        println("=====================================================")
        println("Adaboost Iteration $t")
        println("=====================================================")

        stump = weighted_decision_tree_create(data, features, target,
                                              alpha, 0, 1)
        push!(tree_stumps, stump)

        preds = map(x -> classify(stump, x), eachrow(data))

        correct = preds .== values
        incorrect = preds .≠ values

        weighted_error = sum(alpha[incorrect])/sum(alpha)
        weight = 0.5 * log((1-weighted_error)/weighted_error)
        push!(weights, weight)

        adjustment = map(x -> x ? exp(-weight) : exp(weight), correct)
        alpha = alpha .* adjustment
        s_alpha = sum(alpha)
        alpha = alpha ./ s_alpha
    end
    weights, tree_stumps
end

function predict(weights, stumps, data)
    scores = zeros(Float64, size(data)[1])
    
    for (i, stump) in enumerate(stumps)
        preds = map(x -> classify(stump, x), eachrow(data))
        scores = scores + weights[i] .* preds
    end

    map(x -> x > 0 ? 1 : -1, scores)
end
end

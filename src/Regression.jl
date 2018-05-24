module Regression

export Model, fit_logistic_ga, fit_logistic_cg, fit_logistic_sgd, predict_logistic

struct Model{T<:Real}
    w::Vector{T}
end

function logit(x)
    1. / (1. + exp(x))
end

function predict_logistic(features, w)
    wth = features * w
    logit.(-wth)
end

function feature_deriv_logistic(errors, feature)
    dot(errors, feature)
end

function compute_log_likelihood_logistic(model, features, output)
    ind = map(x -> (x == 1) ? 1 : 0, output)
    scores = features * model.w
    logexp = log.(1. + exp.(-scores))

    mask = isinf.(logexp)
    logexp[mask] = -scores[mask]

    sum((ind .- 1) .* scores .- logexp)
end

function compute_avg_log_likelihood_logistic(model, features, output)
    ind = map(x -> (x == 1) ? 1 : 0, output)
    scores = features * model.w
    logexp = log.(1. + exp.(-scores))

    mask = isinf.(logexp)
    logexp[mask] = -scores[mask]

    sum((ind .- 1) .* scores .- logexp) / size(features)[1]
end

function fit_logistic_ga(feature_matrix, output, w, step_size, max_iter)
    indicators = map(x -> (x == 1) ? 1 : 0, output)
    for iter in 1:max_iter
        preds = predict_logistic(feature_matrix, w)
        errors = indicators .- preds
        for j in eachindex(w)
            deriv = feature_deriv_logistic(errors, feature_matrix[:, j])
            w[j] = w[j] + step_size * deriv 
        end
    end
    Model(w)
end

function fit_logistic_cg(feature_matrix, output, w, λ, max_iter)
    indicators = map(x -> (x == 1) ? 1 : 0, output)
    preds = predict_logistic(feature_matrix, w)
    errors = indicators .- preds
    u = g = feature_matrix' * errors

    for i in 1:max_iter
        Xw = feature_matrix * w
        a = logit.(Xw) .*
            (1 .- logit.(Xw)) .*
            (feature_matrix * u) .^ 2

        uthu = -(λ*u'*u + sum(a))
        w = w .- g'*u / uthu * u
        
        preds = predict_logistic(feature_matrix, w)
        errors = indicators .- preds
        g_n = feature_matrix' * errors .- λ*w
        β = g_n' * (g_n .- g) / (u' * (g_n .- g))
        g = g_n
        u = g .- u*β
    end
    Model(w)
end

function fit_logistic_sgd(feature_matrix, output, w, step_size, batch_size,
                          max_iter, window)
    rows = size(feature_matrix)[1]
    coefficients = Matrix{Float64}(size(w)[1], window)
    log_ll = Vector{Float64}(max_iter)
    # Shuffle the matrix in case there is some
    # internal structure
    perm = randperm(rows)
    feature_matrix = feature_matrix[perm,:]
    output = output[perm]
    inds = map(x -> (x == 1) ? 1 : 0, output)

    base = 1 # Base index for current batch
    for iter in 1:max_iter
        batch_features = feature_matrix[base:base+batch_size,:]
        preds = predict_logistic(batch_features, w)
        errors = inds[base:base+batch_size] .- preds
        deriv = batch_features' * errors
        w = w .+ (1. / batch_size) * step_size .* deriv
        @views lp = compute_avg_log_likelihood_logistic(Model(w),
                                                        batch_features,
                                                        output[base:base+batch_size])
        push!(log_ll, lp)

        # If a complete pass was made reshuffle data
        base += batch_size
        
        if base + batch_size > rows
            perm = randperm(rows)
            feature_matrix = feature_matrix[perm,:]
            output = output[perm]
            inds = inds[perm]
            base = 1
        end
        coefficients[:,iter % window + 1] = w
    end
    mean(coefficients,2), log_ll # Return the average of the last `window` coefficients
end
end

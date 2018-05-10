module Regression

export Model, fit_logistic, predict_logistic

struct Model{T<:Real}
    w::Vector{T}
end

function predict_logistic(features, w)
    preds = similar(w)
    wth = features * w
    1 ./ (1+exp.(-wth))
end

function feature_deriv_logistic(errors, feature)
    dot(errors, feature)
end

function compute_log_likelihood_logistic(model, features, output)
    ind = map(x -> (x == 1) ? 1 : 0, output)
    scores = features * w
    logexp = log.(1. + exp.(-scores))

    mask = isinf.(logexp)
    logexp[mask] = -scores[mask]

    sum((ind - 1) .* scores - logexp)
end

function fit_logistic(feature_matrix, output, w, step_size, max_iter)
    indicators = map(x -> (x == 1) ? 1 : 0, output)
    for iter in 1:max_iter
        preds = predict_logistic(feature_matrix, w)
        errors = indicators - preds
        for j in eachindex(w)
            deriv = feature_deriv_logistic(errors, feature_matrix[:, j])
            w[j] = w[j] + step_size * deriv 
        end
    end
    Model(w)
end
end

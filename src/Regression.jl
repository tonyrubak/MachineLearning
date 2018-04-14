module Regression

export Model, fit_logistic, predict_logistic

struct Model{T<:Real}
    β::Vector{T}
end

function predict_logistic(features, β)
    wth = features * β
    1 ./ (1+exp.(-wth))
end

function feature_deriv_logistic(errors, feature)
    dot(errors, feature)
end

function compute_log_likelihood_logistic(model, features, output)
    ind = map(x -> (x == 1) ? 1 : 0, output)
    scores = features * model.β
    logexp = log.(1. + exp.(-scores))

    mask = isinf.(logexp)
    logexp[mask] = -scores[mask]

    sum((ind - 1) .* scores - logexp)
end

function fit_logistic(feature_matrix, output, β, step_size, max_iter)
    indicators = map(x -> (x == 1) ? 1 : 0, output)
    for iter in 1:max_iter
        preds = predict_logistic(feature_matrix, β)
        errors = indicators - preds
        for j in eachindex(β)
            δ = feature_deriv_logistic(errors, feature_matrix[:, j])
            β[j] = β[j] + step_size * δ
        end
    end
    Model(β)
end
end

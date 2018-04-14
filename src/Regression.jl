module Regression

export Model, fit

struct Model{T<:Real}
    β::Vector{T}
end

function predict_logistic(model, features)
    wth = features * model.β
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

function fit_logistic(feature_matrix, output, initial_β, step_size, max_iter)
    for iter in 1:max_iter
        
    end
end
end

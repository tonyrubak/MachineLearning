module Regression

export Model, fit_logistic_ga, fit_logistic_cg, predict_logistic

struct Model{T<:Real}
    w::Vector{T}
end

function logit(x)
    1. / (1. + exp(x))
end

function predict_logistic(features, w)
    wth = features * w
    1 ./ (1 .+ exp.(-wth))
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

    sum((ind .- 1) .* scores .- logexp)
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
end

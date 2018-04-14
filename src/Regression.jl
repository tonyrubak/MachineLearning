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

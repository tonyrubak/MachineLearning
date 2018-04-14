module Regression

export Model, fit

struct Model{T<:Real}
    β::Vector{T}
end

function predict_logistic(model, features)
    wth = features * model.β
    1 ./ (1+exp.(-wth))
end

end

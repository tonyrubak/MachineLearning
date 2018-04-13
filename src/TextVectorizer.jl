module TextVectorizer
export Vectorizer, fit_vectorizer, transform, transform_test

struct Vectorizer{T<:Dict}
    dfs::Vector{Int64}
    idxs::T
end

function fit_vectorizer(documents::Array{Array{SubString{String},1},1})
    vocab = Set{String}()
    idxs = Dict{String, Int64}()
    dfs = Vector{Int64}()
    for doc in documents
        for word in doc
            if !(haskey(idxs, word))
                push!(dfs, 1)
                idxs[word] = length(dfs)
            else
                idx = idxs[word]
                dfs[idx] += 1
            end
        end
    end
    Vectorizer(dfs, idxs)
end

row_transform = function(vectorizer, document)
    row = spzeros(Int64, length(vectorizer.dfs))
    for word in document
        if word in vectorizer.vocabulary
            row[vectorizer.idxs[word]] += 1
        end
    end
    row
end

function transform(vectorizer, documents)
    hcat(sparse(row_transform.(vectorizer, documents))...)'
end

function transform_test(vectorizer, documents)
    @time row_transform(vectorizer, documents[1])
end
end

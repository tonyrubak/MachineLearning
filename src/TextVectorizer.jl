module TextVectorizer
export Vectorizer, fit_vectorizer, transform, transform_test

struct Vectorizer{T<:Dict}
    dfs::Vector{Int64}
    idxs::T
end

fit_vectorizer = function(documents::Array{Array{SubString{String},1},1})
    vocab = Set{String}()
    dfs = Dict{String, Int64}()
    idxs = Dict{String, Int64}()
    for doc in documents
        for word in doc
            if !(word in vocab)
                push!(vocab, word)
                dfs[word] = 1
                idxs[word] = length(idxs) + 1
            else
                dfs[word] += 1
            end
        end
    end
    Vectorizer(vocab, dfs, idxs)
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

transform = function(vectorizer, documents)
    hcat(sparse(row_transform.(vectorizer, documents))...)'
end
end

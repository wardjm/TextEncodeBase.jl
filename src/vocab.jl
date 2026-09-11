using StaticArrays

abstract type AbstractVocabulary{T} end

Base.eltype(::AbstractVocabulary{T}) where T = T

struct Vocab{T, A<:AbstractVector{T}} <: AbstractVocabulary{T}
    list::A
    unk::T
    unki::Int
end

"""
    Vocab(data::Vector{<:AbstractString}, unk::AbstractString="[UNK]")

Constructor for `Vocab`. `data` is the list of vocabulary word, can be nonunique.
 The actual list will be the unique version of `data` (i.e. `vocab.list = unique(data)`).
 `unk` is the indicator word for all unknown words. `unk` can be either in or not in `data`,
 depends on the use case.
"""
Vocab(data::AbstractVector, unk::AbstractString="[UNK]") = Vocab{String}(data, unk)

"""
    Vocab{T}(data::AbstractVector, unk) where T

construct Vocab with element type `T`. `unk` must be specified.
"""
function Vocab{T}(data::AbstractVector, unk) where T
    udata = Vector{T}(undef, length(data))
    unk = T(unk)
    unique!(map!(T, udata, data))
    list = SizedVector{length(udata)}(OverwritableLookupVector(udata))
    i = findfirst(==(unk), list)
    unki = isnothing(i) ? 0 : i
    return Vocab(list, unk, unki)
end

Base.length(v::Vocab) = length(v.list)

function Base.show(io::IO, v::Vocab)
    print(io, "Vocab{", eltype(v), ", ", nameof(typeof(v.list)), '}')
    print(io, "(size = ", length(v))
    print(io, ", unk = ", v.unk)
    print(io, ", unki = ", v.unki, ')')
end


lookup_index(v::Vocab, word) = lookup_index(v.list, v.unki, word)
lookup_index(list::SizedVector, unki, word) = lookup_index(list.data, unki, word)
lookup_index(list, unki, word) = (i = findfirst(==(word), list); isnothing(i) ? unki : i)

lookup_word(v::Vocab, index) = lookup_word(v.list, v.unk, index)
lookup_word(list::SizedVector, unk, index) = lookup_word(list.data, unk, index)
lookup_word(list, unk, index) = 0 < index <= length(list) ? @inbounds(list[index]) : unk

lookup(v::Vocab) = Base.Fix1(lookup,  v)
lookup(::Type{T}, v::Vocab) where T = lookup $ T $ v

lookup(::Type{I}, v::Vocab{T}, word::T) where {T, I<:Integer} = I(lookup_index(v, word))
lookup(::Type{I}, v::Vocab{<:Integer}, word::Integer) where I<:Integer = I(lookup_index(v, word))
lookup(::Type{I}, v::Vocab{<:AbstractString}, word::AbstractString) where I<:Integer = I(lookup_index(v, word))
lookup(::Type{T}, v::Vocab{T}, index::Integer) where T = lookup_word(v, index)
lookup(::Type{<:Integer}, v::Vocab{T}, i::Integer) where T = throw(DomainError(i, "Cannot lookup the value $i in the vocabulary: value should have the same type as Vocab's element type ($(eltype(v)))"))

lookup(v::Vocab{T}, word::T) where T = lookup(Int, v, word)
lookup(v::Vocab{<:AbstractString}, word::AbstractString) = lookup(Int, v, word)
lookup(v::Vocab{<:Integer}, index::Integer) = lookup_word(v, index)
lookup(v::Vocab, index::Integer) = lookup(eltype(v), v, index)

lookup(v::Vocab, i, j, k...) = (lookup(v, i), lookup(v, j), map(lookup(v), k)...)
lookup(v::Vocab, is::Union{AbstractArray, Tuple, NamedTuple}) = map(lookup(v), is)
lookup(::Type{T}, v::Vocab, i, j, k...) where T = lookup(T, v, (i, j, k...))
lookup(::Type{T}, v::Vocab, is::Union{AbstractArray, Tuple, NamedTuple}) where T = map(lookup(T, v), is)

lookup(::Type{OneHot}, v::Vocab, i) = lookup_onehot(v, i)
lookup(T::Type{OneHot}, v::Vocab, is::AbstractArray) = OneHotArray{length(v)}(lookup_onehot(v, is))
lookup(::Type{OneHot}, v::Vocab, is::Union{Tuple, NamedTuple}) = map(lookup(OneHot, v), is)

lookup_onehot(v::Vocab, i) = OneHot(length(v))(lookup(UInt32, v, i))
lookup_onehot(v::Vocab, is::AbstractArray) = map(lookup_onehot $ v, is)

function lookup(T::Type{OneHot}, v::Vocab, i, j, k...)
    c = 0
    si = lookup(T, v, i)
    c += si isa OneHot ? 1 : length(parent(si))
    sj = lookup(T, v, j)
    c += sj isa OneHot ? 1 : length(parent(sj))
    sk = map(lookup(T, v), k)
    c += sum(k->k isa OneHot ? 1 : length(parent(k)), sk; init=0)
    arr = Vector{OneHot(length(v))}(undef, c); empty!(arr)
    si isa OneHot ? push!(arr, si) : append!(arr, parent(si))
    sj isa OneHot ? push!(arr, sj) : append!(arr, parent(sj))
    for k in sk
        k isa OneHot ? push!(arr, k) : append!(arr, parent(k))
    end
    return OneHotArray(arr)
end

lookup(v::Vocab, i::OneHot) = lookup(v, Int(i))
lookup(v::Vocab, i::OneHotArray) = lookup(v, reinterpret(UInt32, i))
lookup(::Type{T}, v::Vocab{T}, i::OneHot) where T = lookup(v, i)
lookup(::Type{T}, v::Vocab{T}, i::OneHotArray) where T = lookup(v, i)


"""
    lookup(v::Vocab, x)

Lookup `x` in `v`. `lookup` words depends on the type of `x`. If `x` is an integer,
 return the `x`-th word on the vocabulary list (i.e. `v.list[x]`) and return the unknown word
 if `x` is out-of-bound (`v.unk`). If `x` is a string, return the indice of `x` in the vocabulary
 list (i.e `findfirst(==(x), v.list`) and return the unknown indice if `x` not found in the list.
 If the unknown word `v.unk` is in the list, the unknown indice is its indice, otherwise 0.

This function is bidirectional except for `Vocab{<:Integer}`. For integer vocabulary, this function
 only get the `x`-th word (`v.list[x]`). Use `lookup(Int, v, x)` for explicit indice lookup.

# Example
```julia
julia> vocab = Vocab(["a", "b", "c", "a", "b", "c"])
Vocab{String, StaticArrays.SizedVector{3, String, Vector{String}}}(size = 3, unk = [UNK], unki = 0)

julia> vocab_unk = Vocab(["a", "b", "xxx"], "xxx")
Vocab{String, StaticArrays.SizedVector{3, String, Vector{String}}}(size = 3, unk = xxx, unki = 3)

julia> lookup(vocab, "b")
2

julia> lookup(vocab, "d")
0

julia> lookup(vocab_unk, "d")
3

julia> lookup(vocab, 1)
"a"

julia> lookup(vocab, 10000)
"[UNK]"

julia> lookup(vocab_unk, 10000)
"xxx"

```
"""
function lookup end

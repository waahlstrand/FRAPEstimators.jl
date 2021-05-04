using Distributions

function unirand(lower_bound::T, upper_bound::T) where {T<:Real}

    r = rand(Uniform(lower_bound, upper_bound))

    return convert(T, r)

end

function logunirand(lower_bound::T, upper_bound::T) where {T<:Real}

    lower_bound = log10(lower_bound)
    upper_bound = log10(upper_bound)

    r = 10 ^(rand(Uniform(lower_bound, upper_bound)))

    return convert(T, r)
end

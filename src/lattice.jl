## linear indexing for a 2D lattice of dimension dims
function linear_ind(p::NTuple{D,Int}, dims::NTuple{D,Int}) where{D}
    res = mapreduce(+,enumerate(p[2:end]), init = p[1]) do (jj, pp)
        (pp-1)*prod(dims[1:jj])
    end
    return res
end

## coordinates on a 2D lattice of dimension dims given the linear index
function _coordinate_simple_lattice(p::Int, dims::NTuple{D,Int}) where {D}
    p_vec = Vector{Int64}(undef, D)
    p_r = p
    for jj in D:-1:2
        modifier = prod(dims[1:jj-1])
        p_vec[jj] = div(p_r - 1, modifier) + 1
        p_r = mod1(p_r, modifier)
    end
    p_vec[1] = p_r
    return Tuple(p_vec)
end

## Vector of nearest neighbours pairs
function nearest_neighbours(L, mapping::Vector{Int}; periodic_x::Bool = true, periodic_y::Bool = true)
    prod_it = Iterators.product(UnitRange.(1, L)...)

    iter = map(prod_it) do pos
        map(enumerate(L)) do (dir,dim)
            (!(periodic_x) && pos[1] >= L[1] && dir == 1) && return
            (!(periodic_y) && pos[2] >= L[2] && dir == 2) && return
            unit_vec = zeros(Int64, 2)
            unit_vec[dir] = 1
            nextpos =  Tuple(map(pp -> mod(pp-1, dim)+1, pos .+ unit_vec))

            return (mapping[linear_ind(pos, L)], mapping[linear_ind(nextpos, L)]) 
        end
    end

    return Vector{Tuple{Vararg{Int}}}(filter(!isnothing, vcat(vec(iter)...))) 
end


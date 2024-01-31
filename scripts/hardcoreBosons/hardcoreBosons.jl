function move_wall_right(conf::Tuple{Vararg{Vector{Int64}}}, l::Int)
    return Tuple(map(enumerate(conf)) do (layer,chain)
        if layer == l
            if chain[end] == 1
                error("domain wall is already at the right end!")
            end
            return vcat(chain[end], chain[1:end-1]...)
        else
            return chain
        end
    end)
end

function move_wall_left(conf::Tuple{Vararg{Vector{Int64}}}, l::Int)
    return Tuple(map(enumerate(conf)) do (layer,chain)
        if layer == l
            if chain[1] == 1
                error("domain wall is already at the left end!")
            end
            return vcat(chain[2:end]..., chain[1])
        else
            return chain
        end
    end)
end

    
function build_matrix(domain_walls, table, param)
    (L,J,g,h) = param

    rows = Vector{Int}(); 
    columns = Vector{Int}(); 
    values = Vector{Float64}();

    for dw in domain_walls
        dw_int = get(table, dw, false)
        dw_int == false && continue

        #hopping_term
        for layer in 1:L[2]
            #right
            if dw[layer][end] == 0
                dw_hopr = move_wall_right(dw, layer)
                dw_hopr_int = get(table, dw_hopr, false)
                dw_hopr_int == false && continue
            
                append!(rows, [dw_int])
                append!(columns, [dw_hopr_int])
                append!(values, [g])
            end
            #left
            if dw[layer][1] == 0
                dw_hopl = move_wall_left(dw, layer)
                dw_hopl_int = get(table, dw_hopl, false)
                dw_hopl_int == false && continue
            
                append!(rows, [dw_int])
                append!(columns, [dw_hopl_int])
                append!(values, [g])
            end
        end
        
        #domain_wall-term
        domain_wall_val = mapreduce(+, 1:L[2]) do layer
            site1 = findall(x->x==1, dw[layer])[1]
            next_layer = layer+1 >L[2] ? 1 : layer+1
            site2 = findall(x->x==1, dw[next_layer])[1]

            if (length(dw[layer])==L[1]+1) && (site1==1 || site1==L[1]+1)
                return J*((2*L[1]-1)-2*abs(site1-site2))
            else
                return J*((2*L[1]-3)-2*abs(site1-site2))
            end
        end
        append!(rows, [dw_int])
        append!(columns, [dw_int])
        append!(values, [domain_wall_val])
    end
    
    return sparse(rows,columns,values)
end;

function toSpinGrid(conf::Tuple{Vararg{Vector{Int64}}}, L, n)
    return map(conf) do chain
        chainC = deepcopy(chain)
        append!(prepend!(chainC, fill(0, div(L[1]+1-n,2))), fill(0, div(L[1]+1-n,2)))
        posWall = (findall(x->x==1, chainC)[1]-1)
        return vcat(fill(1, posWall), fill(-1, L[1]-posWall))
    end
end

function imbalance(conf::Tuple{Vararg{Vector{Int64}}}, L, n)
    spins = toSpinGrid(conf, L, n)
    return mapreduce(+, Iterators.product(1:L[1], 1:L[2])) do (i,j)
        j <= L[1]/2 && return spins[i][j]
        return -spins[i][j]
    end/prod(L)
end

function pol(conf::Tuple{Vararg{Vector{Int64}}}, L, n)
    spins = toSpinGrid(conf, L, n)
    return mapreduce(+, Iterators.product(1:L[1], 1:L[2])) do (i,j)
        return spins[i][j]
    end
end

## domain wall length operator
function domainWallL(dw, param)
    (L,J,g,h) = param

    return mapreduce(+, 1:L[2]) do layer
        site1 = findall(x->x==1, dw[layer])[1]
        next_layer = layer+1 >L[2] ? 1 : layer+1
        site2 = findall(x->x==1, dw[next_layer])[1]

        if (length(dw[layer])==L[1]+1) && (site1==1 || site1==L[1]+1)
            return abs(site1-site2)
        else
            return 1+abs(site1-site2)
        end
    end
end;

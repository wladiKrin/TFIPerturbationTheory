include("../../src/PertTheory.jl")

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
        domain_wall_val = 0 #h*pol(dw)

        for layer in 1:L[2]

            site1 = findall(x->x==1, dw[layer])[1]
            next_layer = layer+1 >L[2] ? 1 : layer+1
            site2 = findall(x->x==1, dw[next_layer])[1]

            # if (site1 == 1 || site1 == L[1]+1)
            #   domain_wall_val += (-(2*L[1]-1) + 2*abs(site1-site2))
            # else
            domain_wall_val += (-(2*L[1]-3)+2*abs(site1-site2))
            # end
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

## domain wall length operator;;; not very efficient just use H0 for that
function domainWallL(spins::Tuple{Vararg{Int64}}, L, neigh)
    spins = reshape([s for s in spins], L)

    D = 0
    for (i,j) in neigh
        pos1 = _coordinate_simple_lattice(i, L)
        pos2 = _coordinate_simple_lattice(j, L)
        s1 = spins[pos1[1],pos1[2]] == 1 ? 1 : -1
        s2 = spins[pos2[1],pos2[2]] == 1 ? 1 : -1
        D += (1-s1*s2)/2
    end
    return trunc(Int,D)
end;

gs = [-0.25,-0.5,-0.75,-1.0,-1.25,-1.5,-1.75]

let
    #parameters
    J = -1;
    # g = -1.;
    g = gs[parse(Int, ARGS[1])]
    h = 0.;
    L= (8,8)
    N = prod(L);

    n = 5
    dt = 0.1
    T = 2e2

    single_dw_basis = map(1:n) do l
        s = fill(0,n)
        s[l] = 1
        return s
    end

	#initial state
    dw_basis = vec(collect(Iterators.product(fill(single_dw_basis,L[2])...)));
    @show size(dw_basis)
    init_idx = findall(x->x==Tuple(fill(single_dw_basis[div(n+1,2)], L[2])), dw_basis)[1]

	psi=zeros(length(dw_basis))
	psi[init_idx]=1

    imb_precalc = map(dw_basis) do dw
        return imbalance(dw, L, n)
    end

    basis_table = Dict(
        map(enumerate(dw_basis)) do (i, spin)
            return (spin, i)
        end
    );

	H = build_matrix(dw_basis, basis_table, (L,J,g,h))

	# Compute time evolution
	data = Any[]
	t = 0
	while t < T
        @time begin
            println(t)
            imb = mapreduce(+,enumerate(psi)) do (i,psi_i)
                α = abs(psi_i)^2
                return α * imb_precalc[i]
            end

            append!(data, [[t, imb]])#xpol]]) #, dw]])
            
            # Propagate state
            psi,_ = exponentiate(H, -1im*dt, psi)
            t += dt
        end
	end

  df = DataFrame(t = [d[1] for d in data], imb = [d[2] for d in data]) #, pol = [d[3] for d in data])#, dw = [d[4] for d in data])
  CSV.write("../../data/obs_Eff_Bound_hardcoreBosons_n=$(n)_L=($(L[1])_$(L[2]))_g=$(g).csv", df);
end

include("../../src/PertTheory.jl")
using Statistics
using FastExpm

gs = [-0.1,-0.2,-0.3,-0.5,-0.7,-0.75,-0.9,-1.0,-1.25,-1.5,-1.75,-2.0]

## parity operator
function build_parity(spin_basis, table, param)
    (L,J,g,h) = param

    rows    = Vector{Int}(); 
    columns = Vector{Int}(); 
    values  = Vector{Float64}();
    
    for spin in spin_basis
        (n1,D1) = get(table, spin, (false,false))
        n1 == false && continue

        newSpin = abs.(deepcopy(spin) .- 1)
        (n2,D2) = get(table, newSpin, (false,false))
        n2 == false && continue

        append!(rows, [n1])
        append!(columns, [n2])
        append!(values, [1])
    end
    
    return dropzeros(sparse(rows,columns,values, length(spin_basis), length(spin_basis)))
end;

## total polarization
function build_pol(spin_basis, table, param)
    (L,J,g,h) = param

    rows    = Vector{Int}(); 
    columns = Vector{Int}(); 
    values  = Vector{Float64}();
    
    for spin in spin_basis
        (n1,D1) = get(table, spin, (false,false))
        n1 == false && continue

        append!(rows, [n1])
        append!(columns, [n1])
        append!(values, [sum(2 .* spin .- 1)])
    end
    
    return dropzeros(sparse(rows,columns,values, length(spin_basis), length(spin_basis)))
end;

## lattice inversion
function build_lattice_inversion(spin_basis, table, param)
    (L,J,g,h) = param

    rows    = Vector{Int}(); 
    columns = Vector{Int}(); 
    values  = Vector{Float64}();
    
    for spin in spin_basis
        (n1,D1) = get(table, spin, (false,false))
        n1 == false && continue

        newSpin = Tuple(hcat(rotr90(toSpinMatr(spin, L), 2)...))
        (n2,D2) = get(table, newSpin, (false,false))
        n2 == false && continue

        append!(rows, [n1])
        append!(columns, [n2])
        append!(values, [1])
    end
    
    return dropzeros(sparse(rows,columns,values, length(spin_basis), length(spin_basis)))
end;

## lattice mirroring along x-axis
function build_lattice_mirror(spin_basis, table, param)
    (L,J,g,h) = param

    rows    = Vector{Int}(); 
    columns = Vector{Int}(); 
    values  = Vector{Float64}();
    
    for spin in spin_basis
        (n1,D1) = get(table, spin, (false,false))
        n1 == false && continue

        spinM = toSpinMatr(spin, L)
        newSpinM = copy(spinM)
        newSpinM[:,1] = spinM[:,4]
        newSpinM[:,2] = spinM[:,3]
        newSpinM[:,3] = spinM[:,2]
        newSpinM[:,4] = spinM[:,1]
        val = 1 #(-1)^div(sum(abs.(newSpinM .- spinM)), 2)


        newSpin = Tuple(hcat(newSpinM...))
        (n2,D2) = get(table, newSpin, (false,false))
        n2 == false && continue

        append!(rows, [n1])
        append!(columns, [n2])
        append!(values, [val])
    end
    
    return dropzeros(sparse(rows,columns,values, length(spin_basis), length(spin_basis)))
end;

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

## imbalance correlation
function corrF(spins::Tuple{Vararg{Int64}}, L)
    spins = reshape([s for s in spins], L)

    left = mapreduce(+, Iterators.product(1:L[1], 1:floor(Int, L[2]/2))) do (i,j)
        spin = (spins[i,j] == 1) ? 1 : -1
        return spin
    end/(prod(L)/2)
    right = mapreduce(+, Iterators.product(1:L[1], floor(Int, L[2]/2)+1:L[2])) do (i,j)
        spin = (spins[i,j] == 1) ? 1 : -1
        return spin
    end/(prod(L)/2)
    return left*right
end

L= (4,4)
N = prod(L);
J = -1;
h = -0.

next_neighbours = nearest_neighbours(L, collect(1:prod(L)), periodic_y=false)

spin_basis = vec(collect(Iterators.product(fill([1,0],N)...)));
dw_precalc = map(spin_basis) do spin
    return domainWallL(spin, L, next_neighbours)
end

### sort basis according to domain wall length ###
sorted_spin_basis = sort(collect(zip(dw_precalc, spin_basis)), by = x->x[1])
dw_precalc  = [d[1] for d in sorted_spin_basis]
spin_basis  = [d[2] for d in sorted_spin_basis];

spin_basis_table = Dict(
    map(enumerate(zip(spin_basis,dw_precalc))) do (i, (spin, dw))
        return (spin, (i,dw))
    end
);

# for g in gs
let
    array_id = parse(Int, ARGS[1])
    g = gs[array_id]
    @show g

    ### build Hamiltonians ###
    H0  = build_H0(spin_basis, next_neighbours, spin_basis_table, (L,J,g,h));
    H1, R1 = build_H1_R1(spin_basis, spin_basis_table, (L,J,g,h));
    V1 = H1+R1;
    H = H0+V1

    vals, vecs, dw = readSpec("../../data/spec_ED_Bound_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h)")

    imb_precalc = map(s -> imbalance(s, L), spin_basis)

    imbM = sparse(diagm(imb_precalc))
    parM = build_parity(spin_basis, spin_basis_table, (L,J,g,h))
    polM = build_pol(spin_basis, spin_basis_table, (L,J,g,h))
    lIM = build_lattice_inversion(spin_basis, spin_basis_table, (L,J,g,h))
    lMM = build_lattice_mirror(spin_basis, spin_basis_table, (L,J,g,h))
    oneM = sparse(1.0I, length(spin_basis), length(spin_basis));

    CImb = sparse(sgn.(imbM))
    KImb = 1im/2 * (CImb*lMM-lMM*CImb)

    #=
    commImb1 = commutator(CImb, H)
    commImb2 = commutator(KImb, H)

    CPol = sparse(sgn.(polM))
    KPol = 1im/2 * (CPol*parM-parM*CPol)

    commPol1 = commutator(CPol, H)
    commPol2 = commutator(KPol, H)

    data = map(1:length(spin_basis)) do i
        v = vecs[:,i]
        return dot(v, commImb1*v), dot(v, commImb2*v), dot(v, commPol1*v), dot(v, commPol2*v), dot(v, CImb*v), dot(v, CPol*v)
    end
        
    name = "../../data/symmBreaking_eigenvectors_L=($(L[1])_$(L[2]))_g=$(g)"
    h5open(name*".h5", "w") do file    
        file["commCPol"] = [d[1] for d in data]
        file["commKPol"] = [d[2] for d in data]
        file["commCImb"] = [d[3] for d in data]
        file["commKImb"] = [d[4] for d in data]
        file["CImb"]     = [d[5] for d in data]
        file["CPol"]     = [d[6] for d in data]
    end
    =#

    CImb_precalc = map(1:length(spin_basis)) do i
        v = vecs[:,i]
        return dot(v, CImb*v)
    end

    Ts      = collect(2.5:-0.5:0.5)
    betas   = 1 ./ Ts
    lambdas = (-7.6, 5.7)

    data = map(betas) do beta
        @show beta

        exponent = -beta*H - lambdas[1] * CImb - lambdas[2] * KImb
        @time "exp" res = fastExpm(exponent; threshold=1e-4, nonzero_tol=1e-9)
        
        @time "Z" Z = tr(res)
        @time E_mean = tr(H*res) / Z
        @time imb_mean = tr(imbM*res) / Z
        @time dw_mean = tr(H0*res) / Z
        @time CImb_mean = tr(CImb*res) / Z
        @time KImb_mean = tr(KImb*res) / Z
        @time lI_mean = tr(lMM*res) / Z

        @show (beta, Z, E_mean, imb_mean, dw_mean, CImb_mean, KImb_mean, lI_mean)
        return real.([beta, Z, E_mean, imb_mean, dw_mean, CImb_mean, KImb_mean, lI_mean])
    end

    name = "../../data/mixedState_GGE_L=($(L[1])_$(L[2]))_g=$(g)"
    h5open(name*".h5", "w") do file    
        file["beta"] = [d[1] for d in data]
        file["Z"] = [d[2] for d in data]
        file["E"] = [d[3] for d in data]
        file["imb"] = [d[4] for d in data]
        file["dw"] = [d[5] for d in data]
        file["C"] = [d[6] for d in data]
        file["K"] = [d[7] for d in data]
        file["lI"] = [d[8] for d in data]
    end
end

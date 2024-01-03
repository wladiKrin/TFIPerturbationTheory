using CSV, DataFrames, HDF5
using Statistics
using Roots
include("../../src/SW.jl");
include("./BWDefinitions.jl")

L= (4,4)
N = prod(L);
J = -1;
g = -0.5
h = -0.

init_states_indices = [145,205,251,285,336,370,416,476]

next_neighbours = nearest_neighbours(L, collect(1:prod(L)))

spin_basis = vec(collect(Iterators.product(fill([1,0],N)...)));
dw_precalc = map(spin_basis) do spin
    return domainWallL(spin, L)
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

domainWall_table = Dict(
    map(enumerate(dw_precalc)) do (i,  dw)
        return (i, dw)
    end
);

# ### build Hamiltonians ###
H0  = build_H0(spin_basis, next_neighbours, spin_basis_table, (L,J,g,h));
H1, R1 = build_H1_R1(spin_basis, spin_basis_table, (L,J,g,h));
V1 = sparse(H1+R1);

### mix the eight important dw states ###
#=
V2 = V1^2
V4 = V2^2
V8 = V4^2

V8_red = map(Iterators.product(init_states_indices,init_states_indices)) do (i,j)
    return V8[i,j]
end

vals,vecs = eigen(Matrix(V8_red))

transf = sparse(Matrix(1.0I, 2^16,2^16))
for (i,j) in Iterators.product(enumerate(init_states_indices),enumerate(init_states_indices))
    transf[i[2],j[2]] = vecs[i[1],j[1]]
end

transf = sparse(transf)
=#

gs = [-0.1,-0.2,-0.3,-0.4,-0.5]

dataInds = []

Threads.@threads for nState in init_states_indices
    println(nState)
    g = -0.5
    H1, R1 = build_H1_R1(spin_basis, spin_basis_table, (L,J,g,h));
    V1 = sparse(H1+R1);
    V = V1
    EInit = -16.1

    f2(x) = x - (H0[nState,nState] + BW_second_order_energy(nState, H0, V, x))
    f4(x) = x - (H0[nState,nState] + BW_second_order_energy(nState, H0, V, x) + BW_fourth_order_energy(nState, H0, V, x))
    f6(x) = x - (H0[nState,nState] + BW_second_order_energy(nState, H0, V, x) + BW_fourth_order_energy(nState, H0, V, x) + BW_sixth_order_energy(nState, H0, V, x))
    f8(x) = x - (H0[nState,nState] + BW_second_order_energy(nState, H0, V, x) + BW_fourth_order_energy(nState, H0, V, x) + BW_sixth_order_energy(nState, H0, V, x) + BW_eighth_order_energy(nState, H0, V, x))
    f8Mod(x) = x - (H0[nState,nState] + BW_eighth_order_energyMod(nState, H0, V, x))

    @time begin
        res = find_zero(f8Mod, EInit) + 16
    end
    @show nState, res
    push!(dataInds, [nState, res])
end

data = DataFrame(inds = [d[1] for d in dataInds], E8=[d[2] for d in dataInds])

# CSV.write("../data/BWPert_states_g=$(g).csv", data)

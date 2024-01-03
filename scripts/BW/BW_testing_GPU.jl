using Pkg; Pkg.activate("..")
using CUDA
using CSV, DataFrames, HDF5
using Statistics, Roots

include("../src/SW.jl");
include("./BWDefinitions.jl");

device!(4)

L= (4,4)
N = prod(L);
J = -1;
g = -0.5
h = -0.

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

# is6 = [i for (i,dw) in enumerate(dw_precalc) if dw == 6];
is8 = [i for (i,dw) in enumerate(dw_precalc) if dw == 8];
is10 = [i for (i,dw) in enumerate(dw_precalc) if dw == 10];
# is12 = [i for (i,dw) in enumerate(dw_precalc) if dw == 12];
# is14 = [i for (i,dw) in enumerate(dw_precalc) if dw == 14];
is = vcat(is8,is10)

spin_basis_red = [spin_basis[i] for i in is];
dw_precalc_red = [dw_precalc[i] for i in is];

# spin_basis_red_table = Dict(
#     map(enumerate(zip(spins_basis_red, dw_precalc_red))) do (i, (spin, dw))
#         return (spin, (i,dw))
#     end
# );

# ### build Hamiltonians ###
H0  = build_H0(spin_basis, next_neighbours, spin_basis_table, (L,J,g,h));
H1, R1 = build_H1_R1(spin_basis, spin_basis_table, (L,J,g,h));
V1 = sparse(H1+R1);

# @time spin_basis_pert = map([99]) do i
#     @show i
#     spin = sparsevec([i], [1], 2^16)
#     @show spin 
#     @show first_order_states(i, spin_basis, H0, V1) 
#     @show second_order_states(i, spin_basis, H0, V1)
#     res = dropzeros(spin + first_order_states(i, spin_basis, H0, V1) + second_order_states(i, spin_basis, H0, V1))
#     return res ./ norm(res)
# end

######### testing order 8 energies ##########

# ### build Hamiltonians ###
H0  = build_H0(spin_basis, next_neighbours, spin_basis_table, (L,J,g,h));
H1, R1 = build_H1_R1(spin_basis, spin_basis_table, (L,J,g,h));
V1 = sparse(H1+R1);

gs = [-0.1,-0.2,-0.3,-0.4,-0.5]
g = -0.5

init_states_indices = [145, 205, 251, 285, 336, 370, 416, 476]
dataInds = map(enumerate(init_states_indices)) do (i, nState)
    @show (i, nState)
    g = -0.5
    H1, R1 = build_H1_R1(spin_basis, spin_basis_table, (L,J,g,h));
    V1 = sparse(H1+R1);
    V = V1
    EInit = -16.1

    # f2(x) = x - (H0[nState,nState] + BW_second_order_energy(nState, H0, V, x))
    # f4(x) = x - (H0[nState,nState] + BW_second_order_energy(nState, H0, V, x) + BW_fourth_order_energy(nState, H0, V, x))
    # f6(x) = x - (H0[nState,nState] + BW_second_order_energy(nState, H0, V, x) + BW_fourth_order_energy(nState, H0, V, x) + BW_sixth_order_energy(nState, H0, V, x))
    # f8(x) = x - (H0[nState,nState] + BW_second_order_energy(nState, H0, V, x) + BW_fourth_order_energy(nState, H0, V, x) + BW_sixth_order_energy(nState, H0, V, x) + BW_eighth_order_energy(nState, H0, V, x))
    f8Mod(x) = x - (H0[nState,nState] + BW_eighth_order_energyModGPU(nState, H0, V, x))
    f8ModOld(x) = x - (H0[nState,nState] + BW_eighth_order_energyModGPUOld(nState, H0, V, x))
    #@time "old" resOld = f8ModOld(EInit)
    @time "new" res = f8Mod(EInit)
    @show res, resOld
    return 0

    #@time res = find_zero(f8Mod, EInit) + 16
    #@show res
    #return res
end

data = DataFrame(inds = init_states_indices, E10=[d[4] for d in dataInds])

@show data
CSV.write("../data/BWPert_8thorder_states.csv", data)

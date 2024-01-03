using CSV, DataFrames, HDF5
using Statistics
include("../../src/SW.jl");
include("./BWDefinitions.jl")


ts = [0.0,0.1]
step = 1.1
tmax = 1e10

### logarithmic timesteps ###
while true
    push!(ts, ts[end]*step)
    ts[end] > tmax && break
end


for g in [-0.1,-0.2,-0.3,-0.4,-0.5]
    L= (4,4)
    N = prod(L);
    J = -1;
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

    ### initial state ###
    init_spin = vcat(fill(1,Int(N/2)),fill(0,Int(N/2)))
    init_idx = first(spin_basis_table[Tuple(init_spin)])

    init_states_indices = [145,205,251,285,336,370,416,476]

    # ### build Hamiltonians ###
    # H0  = build_H0(spin_basis, next_neighbours, spin_basis_table, (L,J,g,h));
    H1f, R1f = build_H1_R1(spin_basis, spin_basis_table, (L,J,g,h));
    V = sparse(H1f+R1f);

    for order in 2:4 
        println("order = $(order)")

        is = find_inds(init_states_indices, order, V)
        println("found indices")
        dw_precalcRed  = [dw_precalc[i] for i in is]
        spin_basisRed  = [spin_basis[i] for i in is];

        spin_basis_tableRed = Dict(
            map(enumerate(zip(spin_basisRed,dw_precalcRed))) do (i, (spin, dw))
                return (spin, (i,dw))
            end
        );

        domainWall_tableRed = Dict(
            map(enumerate(dw_precalcRed)) do (i,  dw)
                return (i, dw)
            end
        );

        ### initial state ###
        init_spin = vcat(fill(1,Int(N/2)),fill(0,Int(N/2)))
        init_idx = first(spin_basis_tableRed[Tuple(init_spin)])

        psi=zeros(length(spin_basisRed))
        psi[init_idx]=1
        psi = sparse(psi)

        ### build Hamiltonians ###
        H0  = build_H0(spin_basisRed, next_neighbours, spin_basis_tableRed, (L,J,g,h));
        H1, R1 = build_H1_R1(spin_basisRed, spin_basis_tableRed, (L,J,g,h));
        V1 = H1+R1;
        H = H0+V1

        println("Starting caluclation of eigenvectors/-values")
        @time "eigen: " vals, vecs = eigen(Matrix(H))

        psi = Transpose(vecs) * psi

        imb_precalc = map(s -> imbalance(s, L), spin_basisRed)
        data = Any[]

        for (t, tf) in zip(ts[1:end-1], ts[2:end])
            dt = tf-t

            psi_prime = vecs*psi

            imb = mapreduce(+, enumerate(psi_prime)) do (i,psi_i)
                α = abs2(psi_i)
                return α * imb_precalc[i]
            end

            dwObs = real.(dot(psi_prime, H0 * psi_prime))

            append!(data, [[t, imb, dwObs]])

            # Propagate state
            U = exp.(-1im*dt .* vals)
            psi = U .* psi
        end

        df = DataFrame(t = [real(d[1]) for d in data], imb = [real(d[2]) for d in data], N = [real(d[3]) for d in data])
        CSV.write("../data/obs_Eff_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h).csv", df)
    end
end

include("../../src/PertTheory.jl");

L= (4,4)
N = prod(L);
J = - 1;
g = - 0.4
h = - 0.

gs = [-0.1,-0.2,-0.3,-0.4,-0.5,-0.7,-0.9,-1.0,-1.25,-2.]

function domainWallLTest(spins::Tuple{Vararg{Int64}}, L)
    conf = toSpinMatr(spins)

    res = map(1:L[2]) do layer
        s = conf[layer,:]
        @show s
        mapreduce(+, zip(s, vcat(s[2:end],s[1]))) do (i,j)
            return i==j ? 0 : 1
        end
    end

    return sum(res .<= 2) == 4 ? true : false
end;

next_neighbours = nearest_neighbours(L, collect(1:prod(L)))

spin_basis = vec(collect(Iterators.product(fill([1,0],N)...)));
dw_precalc = map(spin_basis) do spin
    return domainWallL(spin, L, next_neighbours)
end

### sort basis according to domain wall length ###
sorted_spin_basis = sort(filter(x -> domainWallLTest(x[2], L), collect(zip(dw_precalc, spin_basis))), by = x->x[1])
dw_precalc  = [d[1] for d in sorted_spin_basis]
spin_basis  = [d[2] for d in sorted_spin_basis];
@show length(spin_basis)

ts = [0.0,0.1]
step = 1.1
tmax = 1e10

### logarithmic timesteps ###
while true
    push!(ts, ts[end]*step)
    ts[end] > tmax && break
end

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

psi=zeros(length(spin_basis))
psi[init_idx]=1
psi = sparse(psi)

### build Hamiltonians ###
H0  = build_H0(spin_basis, next_neighbours, spin_basis_table, (L,J,g,h));
H1, R1 = build_H1_R1(spin_basis, spin_basis_table, (L,J,g,h));
V1 = H1+R1;
H = H0+V1

println("Starting caluclation of eigenvectors/-values")

@time "eigen: " vals, vecs = eigen(Matrix(H))
dw = map(1:size(vecs)[2]) do i
    absVals = abs2.(vecs[:,i])
    return sum(absVals .* dw_precalc)
end

dfSpec = DataFrame(en = real.(vals), dw = dw, occ = real.(vecs[init_idx,:]))
CSV.write("../../data/spec_Eff_twoDWStates_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h).csv", dfSpec)

psi = Transpose(vecs) * psi

imb_precalc = map(s -> imbalance(s, L), spin_basis)
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
    global psi = U .* psi
end

df = DataFrame(t = [real(d[1]) for d in data], imb = [real(d[2]) for d in data], N = [real(d[3]) for d in data])
CSV.write("../../data/obs_Eff_twoDWStates_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h).csv", df)
